"""
Flask Application Entry Point
"""
from flask import Flask
from flask_cors import CORS

from backend.api.routes import api
from backend.core.config import Config, current_config
from backend.core.logger import setup_logger

logger = setup_logger(__name__)


def create_app(config_class=None):
    """
    Application factory
    
    Args:
        config_class: Configuration class to use
        
    Returns:
        Flask application instance
    """
    app = Flask(__name__)
    
    # Load configuration
    if config_class is None:
        config_class = current_config
    
    app.config.from_object(config_class)
    
    # Validate configuration
    try:
        Config.validate()
        Config.ensure_directories()
        logger.info("Configuration validated successfully")
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        raise
    
    # Enable CORS
    CORS(app, origins=Config.CORS_ORIGINS)
    
    # Register blueprints
    app.register_blueprint(api)
    
    # Log configuration
    logger.info(f"Application started in {Config.FLASK_ENV} mode")
    logger.info(f"Flask host: {Config.FLASK_HOST}:{Config.FLASK_PORT}")
    
    return app


def main():
    """Main entry point"""
    app = create_app()
    app.run(
        host=Config.FLASK_HOST,
        port=Config.FLASK_PORT,
        debug=Config.DEBUG
    )


if __name__ == "__main__":
    main()

