"""
Configuration management for the application.
Loads settings from environment variables with validation.
"""
import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Config:
    """Base configuration class"""
    
    # Application Settings
    FLASK_ENV: str = os.getenv("FLASK_ENV", "development")
    FLASK_HOST: str = os.getenv("FLASK_HOST", "0.0.0.0")
    FLASK_PORT: int = int(os.getenv("FLASK_PORT", "5000"))
    DEBUG: bool = FLASK_ENV == "development"
    
    # API Keys
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
    TOGETHER_API_KEY: str = os.getenv("TOGETHER_API_KEY", "")
    
    # Pinecone Configuration
    PINECONE_API_KEY: str = os.getenv("PINECONE_API_KEY", "")
    PINECONE_INDEX_NAME: str = os.getenv("PINECONE_INDEX_NAME", "crylon-tea")
    
    # Paths - using absolute paths
    BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent
    VECTOR_STORE_PATH: Path = BASE_DIR / os.getenv("VECTOR_STORE_PATH", "vector_stores")
    DATA_PATH: Path = BASE_DIR / os.getenv("DATA_PATH", "data")
    OUTPUT_PATH: Path = BASE_DIR / os.getenv("OUTPUT_PATH", "outputs")
    FEEDBACK_PATH: Path = OUTPUT_PATH / "feedback"
    REPORTS_PATH: Path = OUTPUT_PATH / "reports"
    
    # FAISS Configuration
    FAISS_INDEX_PATH: Path = VECTOR_STORE_PATH / "faiss" / "faiss_index_tea"
    
    # Model Configuration
    DEFAULT_EMBEDDING_MODEL: str = "text-embedding-ada-002"
    DEFAULT_LLM_MODEL: str = "gpt-4"
    MAX_TOKENS: int = int(os.getenv("MAX_TOKENS", "1024"))
    TEMPERATURE: float = float(os.getenv("TEMPERATURE", "0"))
    
    # CORS Settings
    CORS_ORIGINS: list = os.getenv("CORS_ORIGINS", "*").split(",")
    
    # File Upload Settings
    MAX_FILE_SIZE: int = int(os.getenv("MAX_FILE_SIZE", "10485760"))  # 10MB default
    ALLOWED_EXTENSIONS: set = {"pdf", "csv"}
    
    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    
    @classmethod
    def validate(cls) -> bool:
        """Validate that all required environment variables are set"""
        required_vars = [
            "OPENAI_API_KEY",
        ]
        
        missing_vars = []
        for var in required_vars:
            if not getattr(cls, var):
                missing_vars.append(var)
        
        if missing_vars:
            raise ValueError(
                f"Missing required environment variables: {', '.join(missing_vars)}\n"
                f"Please set them in your .env file or environment."
            )
        
        return True
    
    @classmethod
    def ensure_directories(cls):
        """Ensure all required directories exist"""
        directories = [
            cls.VECTOR_STORE_PATH,
            cls.DATA_PATH,
            cls.OUTPUT_PATH,
            cls.FEEDBACK_PATH,
            cls.REPORTS_PATH,
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)


class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True


class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False
    LOG_LEVEL = "WARNING"


# Select configuration based on environment
config_by_name = {
    "development": DevelopmentConfig,
    "production": ProductionConfig,
}

# Get current configuration
current_config = config_by_name.get(Config.FLASK_ENV, DevelopmentConfig)

