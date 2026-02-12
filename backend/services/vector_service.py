"""
Vector Store Service
Handles injection of documents into FAISS and Pinecone vector databases
"""
import os
import uuid
import base64
import shutil
from typing import List, Tuple
from pathlib import Path

from unstructured.partition.pdf import partition_pdf
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.schema.document import Document
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pinecone import Pinecone
from openai import OpenAI
import numpy as np

from backend.core.config import Config
from backend.core.logger import setup_logger

logger = setup_logger(__name__)


class VectorService:
    """Service for managing vector databases"""
    
    def __init__(self):
        """Initialize vector service"""
        self.config = Config
        self.temp_image_dir = self.config.OUTPUT_PATH / "temp_images_vector"
        self.temp_image_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize Pinecone
        self.pc = Pinecone(api_key=self.config.PINECONE_API_KEY)
        self.pinecone_index = self.pc.Index(self.config.PINECONE_INDEX_NAME)
        
        # Initialize OpenAI client
        self.openai_client = OpenAI(api_key=self.config.OPENAI_API_KEY)
    
    def categorize_elements(self, raw_pdf_elements: List) -> Tuple[List[str], List[str]]:
        """
        Categorize extracted elements from a PDF into tables and texts
        
        Args:
            raw_pdf_elements: List of unstructured.documents.elements
            
        Returns:
            Tuple of (texts, tables)
        """
        tables = []
        texts = []
        
        for element in raw_pdf_elements:
            if "unstructured.documents.elements.Table" in str(type(element)):
                tables.append(str(element))
            elif "unstructured.documents.elements.CompositeElement" in str(type(element)):
                texts.append(str(element))
        
        logger.info(f"Categorized {len(texts)} texts and {len(tables)} tables")
        return texts, tables
    
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
                content="You are financial analyst tasking with providing investment advice"
            ),
            HumanMessage(
                content=[
                    {
                        "type": "text",
                        "text": "You are financial analyst tasking with providing investment advice.\n"
                        "You will be given a mixed of text, tables, and image(s) usually of charts or graphs.\n"
                        "Use this information to provide investment advice related to the user question. \n",
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
    
    def creating_summaries(self, file_path: str) -> Tuple:
        """
        Create summaries for text, tables, and images in PDF
        
        Args:
            file_path: Path to PDF file
            
        Returns:
            Tuple of (texts, text_summaries, tables, table_summaries, image_elements, image_summaries)
        """
        logger.info(f"Creating summaries for {file_path}")
        
        # Clean up old images
        for filename in os.listdir(self.temp_image_dir):
            file_path_to_delete = os.path.join(self.temp_image_dir, filename)
            try:
                if os.path.isfile(file_path_to_delete):
                    os.unlink(file_path_to_delete)
            except Exception as e:
                logger.error(f"Error deleting {file_path_to_delete}: {e}")
        
        # Process PDF
        raw_pdf_elements = partition_pdf(
            filename=file_path,
            extract_images_in_pdf=True,
            infer_table_structure=True,
            chunking_strategy="by_title",
            max_characters=4000,
            new_after_n_chars=3800,
            combine_text_under_n_chars=2000,
            extract_image_block_output_dir=str(self.temp_image_dir),
        )
        
        texts, tables = self.categorize_elements(raw_pdf_elements)
        
        # Split texts
        text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=4000, chunk_overlap=0
        )
        joined_texts = " ".join(texts)
        texts_4k_token = text_splitter.split_text(joined_texts)
        
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
        
        logger.info("All summaries created")
        
        return (
            texts,
            text_summaries,
            tables,
            table_summaries,
            image_elements,
            image_summaries,
        )
    
    def add_new_vectors_to_faiss(
        self,
        texts: List[str],
        text_summaries: List[str],
        tables: List[str],
        table_summaries: List[str],
        image_elements: List[str],
        image_summaries: List[str]
    ) -> FAISS:
        """
        Add new vectors to FAISS database
        
        Args:
            texts: List of text strings
            text_summaries: List of text summaries
            tables: List of table strings
            table_summaries: List of table summaries
            image_elements: List of encoded images
            image_summaries: List of image summaries
            
        Returns:
            Updated FAISS database
        """
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
        
        # Create new vectorstore
        embeddings = OpenAIEmbeddings(openai_api_key=self.config.OPENAI_API_KEY)
        vectorstore = FAISS.from_documents(documents=documents, embedding=embeddings)
        
        # Save temporarily
        temp_path = self.config.VECTOR_STORE_PATH / "faiss" / "faiss_index_new"
        temp_path.parent.mkdir(parents=True, exist_ok=True)
        vectorstore.save_local(str(temp_path))
        
        # Load and merge with existing database
        db_new = FAISS.load_local(str(temp_path), embeddings, allow_dangerous_deserialization=True)
        db = FAISS.load_local(
            str(self.config.FAISS_INDEX_PATH),
            embeddings,
            allow_dangerous_deserialization=True
        )
        db.merge_from(db_new)
        db.save_local(str(self.config.FAISS_INDEX_PATH))
        
        logger.info(f"Added {len(documents)} documents to FAISS")
        return db
    
    def inject_to_faiss(self, file_path: str) -> FAISS:
        """
        Inject PDF into FAISS database
        
        Args:
            file_path: Path to PDF file
            
        Returns:
            Updated FAISS database
        """
        logger.info(f"Injecting {file_path} to FAISS")
        
        texts, text_summaries, tables, table_summaries, image_elements, image_summaries = (
            self.creating_summaries(file_path)
        )
        
        db = self.add_new_vectors_to_faiss(
            texts, text_summaries, tables, table_summaries, image_elements, image_summaries
        )
        
        logger.info("Successfully injected to FAISS")
        return db
    
    def generate_embeddings_and_upsert(self, texts: List[str]):
        """
        Generate embeddings and upsert to Pinecone
        
        Args:
            texts: List of texts to embed and upsert
        """
        batch_size = 10
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            
            # Generate embeddings
            res = self.openai_client.embeddings.create(
                input=batch_texts,
                model=self.config.DEFAULT_EMBEDDING_MODEL
            )
            embeddings = [np.array(record.embedding) for record in res.data]
            
            # Prepare vectors for upsert
            vectors = []
            for text, embedding in zip(batch_texts, embeddings):
                vectors.append(
                    {
                        "id": str(uuid.uuid4()),
                        "values": embedding.tolist(),
                        "metadata": {"text": text},
                    }
                )
            
            self.pinecone_index.upsert(vectors=vectors)
        
        logger.info(f"Upserted {len(texts)} vectors to Pinecone")
    
    def inject_to_pinecone(self, file_path: str):
        """
        Inject PDF into Pinecone database
        
        Args:
            file_path: Path to PDF file
        """
        logger.info(f"Injecting {file_path} to Pinecone")
        
        texts, text_summaries, tables, table_summaries, image_elements, image_summaries = (
            self.creating_summaries(file_path)
        )
        
        all_summaries = table_summaries + text_summaries + image_summaries
        self.generate_embeddings_and_upsert(all_summaries)
        
        logger.info("Successfully injected to Pinecone")


# Create singleton instance
vector_service = VectorService()

