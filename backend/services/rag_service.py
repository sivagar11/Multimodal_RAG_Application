"""
RAG (Retrieval Augmented Generation) Service
Handles question answering using vector databases and LLMs
"""
from typing import Tuple, List, Optional
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.vectorstores import FAISS
from langchain_community.chat_models import ChatAnthropic
from langchain_pinecone import PineconeVectorStore
from langchain_community.llms import Together
from pinecone import Pinecone
from googlesearch import search

from backend.core.config import Config
from backend.core.logger import setup_logger

logger = setup_logger(__name__)


# Prompt template for RAG
PROMPT_TEMPLATE = """
You are financial analyst tasking with providing investment advice.
Answer the question based only on the following context, which can include text, images and tables:
{context}
Question: {question}
Just return the helpful answer in as much as detailed possible.
Answer:
"""


class RAGService:
    """Service for RAG-based question answering"""
    
    def __init__(self):
        """Initialize RAG service with vector stores"""
        self.config = Config
        self._faiss_db = None
        self._pinecone_vectorstore = None
        
    @property
    def faiss_db(self):
        """Lazy load FAISS database"""
        if self._faiss_db is None:
            try:
                embeddings = OpenAIEmbeddings(openai_api_key=self.config.OPENAI_API_KEY)
                self._faiss_db = FAISS.load_local(
                    str(self.config.FAISS_INDEX_PATH),
                    embeddings,
                    allow_dangerous_deserialization=True
                )
                logger.info(f"FAISS database loaded from {self.config.FAISS_INDEX_PATH}")
            except Exception as e:
                logger.error(f"Error loading FAISS database: {e}")
                raise
        return self._faiss_db
    
    @property
    def pinecone_vectorstore(self):
        """Lazy load Pinecone vectorstore"""
        if self._pinecone_vectorstore is None:
            try:
                pc = Pinecone(api_key=self.config.PINECONE_API_KEY)
                index = pc.Index(self.config.PINECONE_INDEX_NAME)
                embeddings = OpenAIEmbeddings(
                    model=self.config.DEFAULT_EMBEDDING_MODEL,
                    openai_api_key=self.config.OPENAI_API_KEY
                )
                text_field = "text"
                self._pinecone_vectorstore = PineconeVectorStore(
                    index, embeddings, text_field
                )
                logger.info(f"Pinecone vectorstore connected to {self.config.PINECONE_INDEX_NAME}")
            except Exception as e:
                logger.error(f"Error connecting to Pinecone: {e}")
                raise
        return self._pinecone_vectorstore
    
    def initialize_model(
        self,
        model_name: str,
        api_key: Optional[str] = None,
        max_tokens: Optional[int] = None
    ) -> LLMChain:
        """
        Initialize LLM model based on name
        
        Args:
            model_name: Name of the model to initialize
            api_key: Optional API key (for custom models)
            max_tokens: Optional max tokens override
            
        Returns:
            Configured LLMChain
        """
        max_tokens = max_tokens or self.config.MAX_TOKENS
        openai_key = api_key or self.config.OPENAI_API_KEY
        
        prompt = PromptTemplate.from_template(PROMPT_TEMPLATE)
        
        if model_name == "gpt-4":
            llm = ChatOpenAI(
                model="gpt-4",
                openai_api_key=openai_key,
                max_tokens=max_tokens,
                temperature=self.config.TEMPERATURE
            )
        elif model_name == "gpt-3.5-turbo":
            llm = ChatOpenAI(
                model="gpt-3.5-turbo",
                openai_api_key=openai_key,
                max_tokens=max_tokens,
                temperature=self.config.TEMPERATURE
            )
        elif model_name == "claude-2":
            llm = ChatAnthropic(
                temperature=self.config.TEMPERATURE,
                anthropic_api_key=self.config.ANTHROPIC_API_KEY,
                model_name="claude-2"
            )
        elif model_name == "mistral":
            llm = Together(
                model="mistralai/Mixtral-8x7B-Instruct-v0.1",
                temperature=self.config.TEMPERATURE,
                max_tokens=500,
                top_k=3,
                together_api_key=self.config.TOGETHER_API_KEY,
            )
        elif model_name == "llama2-7b":
            llm = Together(
                model="meta-llama/Llama-2-7b-chat-hf",
                temperature=self.config.TEMPERATURE,
                max_tokens=500,
                top_k=3,
                together_api_key=self.config.TOGETHER_API_KEY,
            )
        elif model_name == "gemma-7b":
            llm = Together(
                model="google/gemma-7b-it",
                temperature=self.config.TEMPERATURE,
                max_tokens=500,
                top_k=3,
                together_api_key=self.config.TOGETHER_API_KEY,
            )
        else:
            # Support for custom fine-tuned models
            llm = ChatOpenAI(
                model=model_name,
                openai_api_key=openai_key,
                max_tokens=max_tokens,
                temperature=self.config.TEMPERATURE
            )
        
        return LLMChain(llm=llm, prompt=prompt)
    
    def answer_question(
        self,
        question: str,
        model: str,
        database_name: str,
        api_key: Optional[str] = None
    ) -> Tuple[str, Optional[List[str]]]:
        """
        Answer question using RAG approach
        
        Args:
            question: User question
            model: Model name to use
            database_name: Vector database to query ('faiss' or 'pinecone')
            api_key: Optional API key for custom models
            
        Returns:
            Tuple of (answer, relevant_images)
        """
        logger.info(f"Answering question with model={model}, db={database_name}")
        
        # Get vector database
        if database_name == "faiss":
            vector_db = self.faiss_db
        elif database_name == "pinecone":
            vector_db = self.pinecone_vectorstore
        else:
            raise ValueError(f"Unsupported database_name: {database_name}")
        
        # Initialize model
        qa_chain = self.initialize_model(model, api_key=api_key)
        
        # Search for relevant documents
        relevant_docs = vector_db.similarity_search(question)
        
        # Build context from retrieved documents
        context = ""
        relevant_images = []
        
        for d in relevant_docs:
            if database_name == "faiss":
                doc_type = d.metadata.get("type", "text")
                if doc_type == "text":
                    context += "[text]" + d.metadata["original_content"]
                elif doc_type == "table":
                    context += "[table]" + d.metadata["original_content"]
                elif doc_type == "image":
                    context += "[image]" + d.page_content
                    relevant_images.append(d.metadata["original_content"])
            elif database_name == "pinecone":
                context += "[text]" + d.page_content
        
        # Generate answer
        result = qa_chain.run({"context": context, "question": question})
        
        logger.info("Question answered successfully")
        return result, relevant_images if relevant_images else None
    
    def get_related_urls(self, query: str, max_results: int = 5) -> List[str]:
        """
        Get related URLs from Google search
        
        Args:
            query: Search query
            max_results: Maximum number of URLs to return
            
        Returns:
            List of URLs
        """
        try:
            enhanced_query = query + " ceylon tea brokers plc"
            logger.info(f"Searching for URLs: {enhanced_query}")
            
            results = search(enhanced_query)
            urls = []
            
            for i, url in enumerate(results):
                if i >= max_results:
                    break
                urls.append(url)
            
            logger.info(f"Found {len(urls)} URLs")
            return urls
        except Exception as e:
            logger.error(f"Error fetching URLs: {e}")
            return []


# Create singleton instance
rag_service = RAGService()

