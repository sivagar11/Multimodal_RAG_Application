"""
Data Ingestion Service
Handles ingestion of CSV files and URLs into vector databases
"""
import os
import uuid
from typing import List, Tuple

import numpy as np
from bs4 import BeautifulSoup
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema.document import Document
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders.csv_loader import UnstructuredCSVLoader
from langchain_community.document_loaders import SeleniumURLLoader
from openai import OpenAI
from pinecone import Pinecone

from backend.core.config import Config
from backend.core.logger import setup_logger

logger = setup_logger(__name__)


class DataIngestionService:
    """Service for ingesting various data formats into vector databases"""
    
    def __init__(self):
        """Initialize data ingestion service"""
        self.config = Config
        
        # Initialize Pinecone
        self.pc = Pinecone(api_key=self.config.PINECONE_API_KEY)
        self.pinecone_index = self.pc.Index(self.config.PINECONE_INDEX_NAME)
        
        # Initialize OpenAI client
        self.openai_client = OpenAI(api_key=self.config.OPENAI_API_KEY)
    
    def clean_text_from_csv_loader(self, data: List) -> str:
        """
        Clean HTML text from CSV loader output
        
        Args:
            data: List of documents from CSV loader
            
        Returns:
            Cleaned text
        """
        clean_data = []
        
        for document in data:
            page_content = document.page_content
            soup = BeautifulSoup(page_content, "html.parser")
            text_content = soup.get_text(separator="\n", strip=True)
            clean_document = {"page_content": text_content}
            clean_data.append(clean_document)
        
        if clean_data:
            return clean_data[0]["page_content"]
        return ""
    
    def generate_text_summaries(self, texts: List[str]) -> List[str]:
        """
        Generate summaries for texts
        
        Args:
            texts: List of text strings
            
        Returns:
            List of summaries
        """
        prompt_text = """You are an assistant tasked with summarizing text for retrieval. \
        These summaries will be embedded and used to retrieve the raw texts. \
        Give a concise summary of the text that is well optimized for retrieval. Table or text: {element} """
        
        prompt = ChatPromptTemplate.from_template(prompt_text)
        model = ChatOpenAI(
            temperature=0,
            model=self.config.DEFAULT_LLM_MODEL,
            api_key=self.config.OPENAI_API_KEY
        )
        summarize_chain = {"element": lambda x: x} | prompt | model | StrOutputParser()
        
        text_summaries = []
        if texts:
            text_summaries = summarize_chain.batch(texts, {"max_concurrency": 5})
        
        logger.info(f"Generated {len(text_summaries)} summaries")
        return text_summaries
    
    def create_faiss_db(
        self,
        cleaned_data: List[str],
        text_summaries: List[str]
    ) -> FAISS:
        """
        Create FAISS database from cleaned data
        
        Args:
            cleaned_data: List of cleaned text data
            text_summaries: List of text summaries
            
        Returns:
            FAISS vectorstore
        """
        documents = []
        
        for e, s in zip(cleaned_data, text_summaries):
            doc = Document(
                page_content=s,
                metadata={"id": str(uuid.uuid4()), "type": "text", "original_content": e}
            )
            documents.append(doc)
        
        embeddings = OpenAIEmbeddings(openai_api_key=self.config.OPENAI_API_KEY)
        vectorstore = FAISS.from_documents(documents=documents, embedding=embeddings)
        
        # Merge with existing database
        db = FAISS.load_local(
            str(self.config.FAISS_INDEX_PATH),
            embeddings,
            allow_dangerous_deserialization=True
        )
        db.merge_from(vectorstore)
        db.save_local(str(self.config.FAISS_INDEX_PATH))
        
        logger.info("Merged with existing FAISS database")
        return db
    
    def create_summaries(self, docs: List) -> Tuple[List[str], str]:
        """
        Create summaries from documents
        
        Args:
            docs: List of documents
            
        Returns:
            Tuple of (text_summaries, cleaned_data)
        """
        cleaned_data = self.clean_text_from_csv_loader(docs)
        
        text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=2000, chunk_overlap=0
        )
        
        texts_4k_token = text_splitter.split_text(cleaned_data)
        text_summaries = self.generate_text_summaries(texts_4k_token)
        
        return text_summaries, texts_4k_token
    
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
    
    def inject_to_faiss_csv(self, file_path: str) -> FAISS:
        """
        Inject CSV file into FAISS database
        
        Args:
            file_path: Path to CSV file
            
        Returns:
            Updated FAISS database
        """
        logger.info(f"Injecting CSV {file_path} to FAISS")
        
        loader = UnstructuredCSVLoader(file_path, mode="elements")
        docs = loader.load()
        text_summaries, cleaned_data = self.create_summaries(docs)
        faiss_db = self.create_faiss_db(cleaned_data, text_summaries)
        
        logger.info("Successfully injected CSV to FAISS")
        return faiss_db
    
    def inject_to_pinecone_csv(self, file_path: str):
        """
        Inject CSV file into Pinecone database
        
        Args:
            file_path: Path to CSV file
        """
        logger.info(f"Injecting CSV {file_path} to Pinecone")
        
        loader = UnstructuredCSVLoader(file_path, mode="elements")
        docs = loader.load()
        text_summaries, _ = self.create_summaries(docs)
        self.generate_embeddings_and_upsert(text_summaries)
        
        logger.info("Successfully injected CSV to Pinecone")
    
    def inject_csv(self, file_path: str, vector_db: str):
        """
        Inject CSV into specified vector database
        
        Args:
            file_path: Path to CSV file
            vector_db: Vector database name ('faiss' or 'pinecone')
            
        Returns:
            Updated vector database (for FAISS)
        """
        if vector_db == "faiss":
            return self.inject_to_faiss_csv(file_path)
        else:
            self.inject_to_pinecone_csv(file_path)
            return None
    
    def inject_url(self, url: str, vector_db: str):
        """
        Inject URL content into specified vector database
        
        Args:
            url: URL to scrape
            vector_db: Vector database name ('faiss' or 'pinecone')
            
        Returns:
            Updated vector database (for FAISS)
        """
        logger.info(f"Injecting URL {url} to {vector_db}")
        
        urls = [url]
        loader = SeleniumURLLoader(urls=urls)
        data = loader.load()
        text_summaries, cleaned_data = self.create_summaries(data)
        
        if vector_db == "faiss":
            vector_database = self.create_faiss_db(cleaned_data, text_summaries)
        else:
            self.generate_embeddings_and_upsert(text_summaries)
            vector_database = None
        
        logger.info(f"Successfully injected URL to {vector_db}")
        return vector_database


# Create singleton instance
data_ingestion_service = DataIngestionService()

