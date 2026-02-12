"""
Fine-tuning Service
Handles OpenAI model fine-tuning
"""
import json
import random
import time
from typing import Optional
import pandas as pd
from openai import OpenAI

from backend.core.config import Config
from backend.core.logger import setup_logger

logger = setup_logger(__name__)


DEFAULT_SYSTEM_PROMPT = (
    "You are financial analyst tasking with providing investment advice and insights "
    "and designed to handle inquiries related finance. Your task is to assist the user "
    "in addressing their queries effectively."
)


class FinetuningService:
    """Service for fine-tuning OpenAI models"""
    
    def __init__(self):
        """Initialize finetuning service"""
        self.config = Config
    
    def create_dataset(self, question: str, answer: str) -> dict:
        """
        Create a training example in the correct format
        
        Args:
            question: User question
            answer: Expected answer
            
        Returns:
            Formatted training example
        """
        return {
            "messages": [
                {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer},
            ]
        }
    
    def generate_dataset(self, file_path: str) -> str:
        """
        Generate JSONL dataset from CSV file
        
        Args:
            file_path: Path to CSV file with Question and Answer columns
            
        Returns:
            Path to generated JSONL file
        """
        df = pd.read_csv(file_path)
        
        output_file = self.config.OUTPUT_PATH / "finetune.jsonl"
        
        with open(output_file, "w") as f:
            for _, row in df.iterrows():
                example_str = json.dumps(
                    self.create_dataset(row["Question"], row["Answer"])
                )
                f.write(example_str + "\n")
        
        logger.info(f"Generated dataset with {len(df)} examples: {output_file}")
        return str(output_file)
    
    def split_train_validation(
        self,
        input_file: str,
        train_ratio: float = 0.8
    ) -> tuple[list[str], list[str]]:
        """
        Split JSONL file into train and validation sets
        
        Args:
            input_file: Path to JSONL file
            train_ratio: Ratio of training data (default 0.8)
            
        Returns:
            Tuple of (train_set, validation_set)
        """
        with open(input_file, "r") as f:
            lines = f.readlines()
        
        random.shuffle(lines)
        
        train_size = int(len(lines) * train_ratio)
        train_set = lines[:train_size]
        validation_set = lines[train_size:]
        
        logger.info(f"Split dataset: {len(train_set)} train, {len(validation_set)} validation")
        return train_set, validation_set
    
    def fine_tuning(
        self,
        train_set: list[str],
        validation_set: list[str],
        suffix_name: str,
        api_key: str
    ) -> str:
        """
        Execute fine-tuning job
        
        Args:
            train_set: Training data lines
            validation_set: Validation data lines
            suffix_name: Suffix for the fine-tuned model
            api_key: OpenAI API key
            
        Returns:
            Fine-tuned model ID
        """
        train_file = self.config.OUTPUT_PATH / "train_set.jsonl"
        validation_file = self.config.OUTPUT_PATH / "validation_set.jsonl"
        
        # Write files
        with open(train_file, "w") as f:
            f.writelines(train_set)
        
        with open(validation_file, "w") as f:
            f.writelines(validation_set)
        
        client = OpenAI(api_key=api_key)
        
        # Upload training file
        with open(train_file, "rb") as training_file:
            training_response = client.files.create(
                file=training_file,
                purpose="fine-tune"
            )
        training_file_id = training_response.id
        logger.info(f"Training file uploaded: {training_file_id}")
        
        # Upload validation file
        with open(validation_file, "rb") as validation_file:
            validation_response = client.files.create(
                file=validation_file,
                purpose="fine-tune"
            )
        validation_file_id = validation_response.id
        logger.info(f"Validation file uploaded: {validation_file_id}")
        
        # Create fine-tuning job
        response = client.fine_tuning.jobs.create(
            training_file=training_file_id,
            validation_file=validation_file_id,
            model="gpt-3.5-turbo",
            suffix=suffix_name,
        )
        
        job_id = response.id
        logger.info(f"Fine-tuning job created: {job_id}")
        
        # Wait for job to complete (5 minutes)
        time.sleep(300)
        
        # Get the latest model
        list_models = client.models.list()
        last_model = None
        for model in list_models:
            last_model = model
        
        if last_model:
            last_model_id = last_model.id
            logger.info(f"Fine-tuned model ID: {last_model_id}")
            return last_model_id
        else:
            raise Exception("No models found after fine-tuning")
    
    def fine_tune_model(
        self,
        file_path: str,
        suffix_name: str,
        api_key: str
    ) -> str:
        """
        Complete fine-tuning workflow
        
        Args:
            file_path: Path to CSV file with training data
            suffix_name: Suffix for the fine-tuned model
            api_key: OpenAI API key
            
        Returns:
            Fine-tuned model ID
        """
        logger.info(f"Starting fine-tuning workflow for {file_path}")
        
        # Generate dataset
        dataset_file = self.generate_dataset(file_path)
        
        # Split into train/validation
        train_set, validation_set = self.split_train_validation(dataset_file)
        
        # Execute fine-tuning
        model_id = self.fine_tuning(train_set, validation_set, suffix_name, api_key)
        
        logger.info(f"Fine-tuning completed: {model_id}")
        return model_id


# Create singleton instance
finetuning_service = FinetuningService()

