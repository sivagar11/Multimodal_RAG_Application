"""
Evaluation Service
Handles RAG evaluation using RAGAS metrics
"""
import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    answer_correctness,
    context_recall,
    context_precision,
)
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

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


class EvaluationService:
    """Service for evaluating RAG performance"""
    
    def __init__(self):
        """Initialize evaluation service"""
        self.config = Config
    
    def evaluate_rag_using_ragas(
        self,
        openai_api_key: str,
        model: str,
        file_path: str,
    ) -> dict:
        """
        Evaluate RAG using RAGAS metrics
        
        Args:
            openai_api_key: OpenAI API key
            model: Model name to evaluate
            file_path: Path to test CSV file with questions and ground truth
            
        Returns:
            Dictionary with evaluation results
        """
        logger.info(f"Starting RAGAS evaluation with model={model}")
        
        # Load test data
        test_df = pd.read_csv(file_path)
        test_questions = test_df["question"].values.tolist()
        test_groundtruths = test_df["ground_truth"].values.tolist()
        
        # Initialize model
        qa_chain = LLMChain(
            llm=ChatOpenAI(
                model=model,
                openai_api_key=openai_api_key,
                max_tokens=1024,
                temperature=0
            ),
            prompt=PromptTemplate.from_template(PROMPT_TEMPLATE),
        )
        
        # Load Pinecone vectorstore
        embeddings = OpenAIEmbeddings(
            model=self.config.DEFAULT_EMBEDDING_MODEL,
            openai_api_key=openai_api_key
        )
        pc = Pinecone(api_key=self.config.PINECONE_API_KEY)
        index = pc.Index(self.config.PINECONE_INDEX_NAME)
        vector_db = PineconeVectorStore(index, embeddings, "text")
        
        # Generate answers and collect contexts
        answers = []
        contexts = []
        
        for question in test_questions:
            relevant_docs = vector_db.similarity_search(question, k=3)
            context = [d.page_content for d in relevant_docs]
            
            response = qa_chain.run({"context": context, "question": question})
            answers.append(response)
            contexts.append(context)
        
        # Create dataset for evaluation
        response_dataset = Dataset.from_dict(
            {
                "question": test_questions,
                "answer": answers,
                "contexts": contexts,
                "ground_truth": test_groundtruths,
            }
        )
        
        # Define metrics
        metrics = [
            faithfulness,
            answer_relevancy,
            context_recall,
            context_precision,
            answer_correctness,
        ]
        
        # Evaluate
        results = evaluate(response_dataset, metrics, raise_exceptions=False)
        
        logger.info(f"Evaluation completed: {results}")
        return results


# Create singleton instance
evaluation_service = EvaluationService()

