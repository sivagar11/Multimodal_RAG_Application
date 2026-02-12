"""
PDF Processing Service
Handles PDF document processing, vectorization, and question answering
"""
import os
import uuid
import base64
import tempfile
from typing import Tuple, List, Optional
from pathlib import Path

from unstructured.partition.pdf import partition_pdf
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.schema.document import Document
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from backend.core.config import Config
from backend.core.logger import setup_logger

logger = setup_logger(__name__)


PROMPT_TEMPLATE = """
You are document analyst tasking with providing insights from documents.
Answer the question based only on the following context, which can include text, images and tables:
{context}
Question: {question}
Just return the helpful answer in as much as detailed possible.
Answer:
"""


class PDFService:
    """Service for processing PDFs and answering questions"""
    
    def __init__(self):
        """Initialize PDF service"""
        self.config = Config
        self.temp_image_dir = self.config.OUTPUT_PATH / "temp_images"
        self.temp_image_dir.mkdir(parents=True, exist_ok=True)
    
    def process_all_pdfs_to_vector_db(self, directory_path: str) -> List:
        """
        Process all PDFs in a directory
        
        Args:
            directory_path: Path to directory containing PDFs
            
        Returns:
            List of raw PDF elements
        """
        raw_pdf_elements = []
        
        for filename in os.listdir(directory_path):
            if filename.endswith(".pdf"):
                pdf_file_path = os.path.join(directory_path, filename)
                
                elements = partition_pdf(
                    filename=pdf_file_path,
                    extract_images_in_pdf=True,
                    infer_table_structure=True,
                    chunking_strategy="by_title",
                    max_characters=4000,
                    new_after_n_chars=3800,
                    combine_text_under_n_chars=2000,
                    extract_image_block_output_dir=str(self.temp_image_dir),
                )
                raw_pdf_elements.extend(elements)
                logger.info(f"Processed PDF: {filename}")
        
        return raw_pdf_elements
    
    def categorize_elements(self, raw_pdf_elements: List) -> Tuple[List[str], List[str], List[str]]:
        """
        Categorize extracted elements from PDFs into tables and texts
        
        Args:
            raw_pdf_elements: List of unstructured.documents.elements
            
        Returns:
            Tuple of (texts_4k_token, tables, texts)
        """
        tables = []
        texts = []
        
        for element in raw_pdf_elements:
            if "unstructured.documents.elements.Table" in str(type(element)):
                tables.append(str(element))
            elif "unstructured.documents.elements.CompositeElement" in str(type(element)):
                texts.append(str(element))
        
        # Enforce specific token size for texts
        text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=4000, chunk_overlap=0
        )
        joined_texts = " ".join(texts)
        texts_4k_token = text_splitter.split_text(joined_texts)
        
        logger.info(f"Categorized elements: {len(texts_4k_token)} texts, {len(tables)} tables")
        return texts_4k_token, tables, texts
    
    def generate_text_summaries(
        self,
        texts: List[str],
        tables: List[str],
        summarize_texts: bool = False
    ) -> Tuple[List[str], List[str]]:
        """
        Summarize text and table elements
        
        Args:
            texts: List of text strings
            tables: List of table strings
            summarize_texts: Whether to summarize texts
            
        Returns:
            Tuple of (text_summaries, table_summaries)
        """
        prompt_text = """You are an assistant tasked with summarizing tables and text for retrieval. \
        These summaries will be embedded and used to retrieve the raw text or table elements. \
        Give a concise summary of the table or text that is well optimized for retrieval. Table or text: {element} """
        
        prompt = ChatPromptTemplate.from_template(prompt_text)
        model = ChatOpenAI(
            temperature=0,
            model=self.config.DEFAULT_LLM_MODEL,
            api_key=self.config.OPENAI_API_KEY
        )
        summarize_chain = {"element": lambda x: x} | prompt | model | StrOutputParser()
        
        text_summaries = []
        table_summaries = []
        
        if texts and summarize_texts:
            text_summaries = summarize_chain.batch(texts, {"max_concurrency": 5})
        elif texts:
            text_summaries = texts
        
        if tables:
            table_summaries = summarize_chain.batch(tables, {"max_concurrency": 5})
        
        logger.info("Text and table summaries generated")
        return text_summaries, table_summaries
    
    def encode_image(self, image_path: str) -> str:
        """Encode image to base64"""
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    
    def summarize_image(self, encoded_image: str) -> str:
        """
        Summarize image using GPT-4 Vision
        
        Args:
            encoded_image: Base64 encoded image
            
        Returns:
            Image summary text
        """
        prompt = [
            SystemMessage(
                content="You are document analyst tasking with providing insights from documents"
            ),
            HumanMessage(
                content=[
                    {
                        "type": "text",
                        "text": "You are document analyst tasking with providing insights from documents.\n"
                        "You will be given a mixed of text, tables, and image(s) usually of charts or graphs.\n"
                        "Use this information to provide insights related to the user question. \n",
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"},
                    },
                ]
            ),
        ]
        
        response = ChatOpenAI(
            model="gpt-4-vision-preview",
            openai_api_key=self.config.OPENAI_API_KEY,
            max_tokens=1024
        ).invoke(prompt)
        
        return response.content
    
    def process_pdf_to_vector_db(self, directory_path: str) -> FAISS:
        """
        Process PDF directory to create FAISS vector database
        
        Args:
            directory_path: Path to directory containing PDFs
            
        Returns:
            FAISS vectorstore
        """
        raw_elements = self.process_all_pdfs_to_vector_db(directory_path)
        texts_4k_token, tables, texts = self.categorize_elements(raw_elements)
        text_summaries, table_summaries = self.generate_text_summaries(
            texts_4k_token, tables, summarize_texts=True
        )
        
        # Process images
        image_elements = []
        image_summaries = []
        
        for filename in os.listdir(self.temp_image_dir):
            if filename.endswith((".png", ".jpg", ".jpeg")):
                image_path = os.path.join(self.temp_image_dir, filename)
                encoded_image = self.encode_image(image_path)
                image_elements.append(encoded_image)
                summary = self.summarize_image(encoded_image)
                image_summaries.append(summary)
        
        # Create documents for vectorstore
        documents = []
        
        for e, s in zip(texts, text_summaries):
            doc = Document(
                page_content=s,
                metadata={"id": str(uuid.uuid4()), "type": "text", "original_content": e}
            )
            documents.append(doc)
        
        for e, s in zip(tables, table_summaries):
            doc = Document(
                page_content=s,
                metadata={"id": str(uuid.uuid4()), "type": "table", "original_content": e}
            )
            documents.append(doc)
        
        for e, s in zip(image_elements, image_summaries):
            doc = Document(
                page_content=s,
                metadata={"id": str(uuid.uuid4()), "type": "image", "original_content": e}
            )
            documents.append(doc)
        
        # Create vectorstore
        vectorstore = FAISS.from_documents(
            documents=documents,
            embedding=OpenAIEmbeddings(openai_api_key=self.config.OPENAI_API_KEY)
        )
        
        logger.info(f"Created vectorstore with {len(documents)} documents")
        return vectorstore
    
    def answer_pdf(
        self,
        question: str,
        directory_path: str
    ) -> Tuple[str, Optional[List[str]]]:
        """
        Answer question based on PDFs in directory
        
        Args:
            question: User question
            directory_path: Path to directory containing PDFs
            
        Returns:
            Tuple of (answer, relevant_images)
        """
        logger.info(f"Processing PDFs from {directory_path}")
        vectorstore = self.process_pdf_to_vector_db(directory_path)
        
        relevant_docs = vectorstore.similarity_search(question)
        context = ""
        relevant_images = []
        
        qa_chain = LLMChain(
            llm=ChatOpenAI(
                model=self.config.DEFAULT_LLM_MODEL,
                openai_api_key=self.config.OPENAI_API_KEY,
                max_tokens=self.config.MAX_TOKENS
            ),
            prompt=PromptTemplate.from_template(PROMPT_TEMPLATE),
        )
        
        for d in relevant_docs:
            doc_type = d.metadata.get("type", "text")
            if doc_type == "text":
                context += "[text]" + d.metadata["original_content"]
            elif doc_type == "table":
                context += "[table]" + d.metadata["original_content"]
            elif doc_type == "image":
                context += "[image]" + d.page_content
                relevant_images.append(d.metadata["original_content"])
        
        result = qa_chain.run({"context": context, "question": question})
        
        logger.info("Question answered successfully")
        return result, relevant_images if relevant_images else None
    
    def answer_only(self, question: str) -> Tuple[str, Optional[List[str]]]:
        """
        Answer question using pre-loaded FAISS index
        
        Args:
            question: User question
            
        Returns:
            Tuple of (answer, relevant_images)
        """
        embeddings = OpenAIEmbeddings(openai_api_key=self.config.OPENAI_API_KEY)
        vectorstore = FAISS.load_local(
            str(self.config.FAISS_INDEX_PATH.parent / "faiss_index_chat"),
            embeddings,
            allow_dangerous_deserialization=True
        )
        
        relevant_docs = vectorstore.similarity_search(question)
        context = ""
        relevant_images = []
        
        qa_chain = LLMChain(
            llm=ChatOpenAI(
                model=self.config.DEFAULT_LLM_MODEL,
                openai_api_key=self.config.OPENAI_API_KEY,
                max_tokens=2048
            ),
            prompt=PromptTemplate.from_template(PROMPT_TEMPLATE),
        )
        
        for d in relevant_docs:
            doc_type = d.metadata.get("type", "text")
            if doc_type == "text":
                context += "[text]" + d.metadata["original_content"]
            elif doc_type == "table":
                context += "[table]" + d.metadata["original_content"]
            elif doc_type == "image":
                context += "[image]" + d.page_content
                relevant_images.append(d.metadata["original_content"])
        
        result = qa_chain.run({"context": context, "question": question})
        
        return result, relevant_images if relevant_images else None


# Create singleton instance
pdf_service = PDFService()

