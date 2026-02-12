"""
API Routes for the Flask application
"""
import os
import tempfile
import json
from flask import Blueprint, jsonify, request, send_file

from backend.services.rag_service import rag_service
from backend.services.pdf_service import pdf_service
from backend.services.vector_service import vector_service
from backend.services.finetuning_service import finetuning_service
from backend.services.evaluation_service import evaluation_service
from backend.services.report_service import report_service
from backend.services.data_ingestion_service import data_ingestion_service
from backend.utils.feedback import save_feedback_to_csv, save_report_to_csv
from backend.core.config import Config
from backend.core.logger import setup_logger

logger = setup_logger(__name__)

# Create blueprint
api = Blueprint('api', __name__, url_prefix='/api')


@api.route('/hello', methods=['GET'])
def hello():
    """Health check endpoint"""
    return jsonify({"message": "Hello, World!"})


@api.route('/answer_question', methods=['POST'])
def api_answer_question():
    """
    Answer question using RAG
    
    Request JSON:
        - question: User question
        - model: Model name
        - database: Database name ('faiss' or 'pinecone')
        - apikey: Optional API key for custom models
    """
    try:
        data = request.json
        logger.info(f"Answer question request: {data}")
        
        question = data.get('question')
        model = data.get('model')
        database_name = data.get('database')
        api_key = data.get('apikey', None)
        
        # Validate required fields
        if not all([question, model, database_name]):
            return jsonify({"error": "Missing required fields"}), 400
        
        # Get answer
        result, relevant_images = rag_service.answer_question(
            question, model, database_name, api_key
        )
        
        # Get related URLs
        try:
            urls = rag_service.get_related_urls(question)
        except Exception as e:
            logger.warning(f"Error fetching URLs: {e}")
            urls = []
        
        return jsonify({
            "result": result,
            "relevant_images": relevant_images,
            "urls": urls
        })
    
    except Exception as e:
        logger.error(f"Error in answer_question: {e}")
        return jsonify({"error": str(e)}), 500


@api.route('/finetune', methods=['POST'])
def finetune():
    """
    Fine-tune a model
    
    Form data:
        - file: Training data file
        - suffixName: Model suffix
        - apiKey: OpenAI API key
    """
    try:
        logger.info("Finetuning request received")
        
        file = request.files.get('file')
        suffix = request.form.get('suffixName')
        api_key = request.form.get('apiKey')
        
        if not all([file, suffix, api_key]):
            return jsonify({"error": "Missing required fields"}), 400
        
        # Save file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp_file:
            file.save(tmp_file.name)
            tmp_path = tmp_file.name
        
        # Fine-tune
        model_id = finetuning_service.fine_tune_model(tmp_path, suffix, api_key)
        
        # Clean up
        os.unlink(tmp_path)
        
        return jsonify({"model_id": model_id})
    
    except Exception as e:
        logger.error(f"Error in finetune: {e}")
        return jsonify({"error": str(e)}), 500


@api.route('/inject_to_vector_db', methods=['POST'])
def inject_to_vector_db():
    """
    Inject data to vector database
    
    Form data:
        - file: Optional PDF or CSV file
        - database: Database choice ('faiss' or 'pinecone')
        - url: Optional URL to scrape
    """
    try:
        file = request.files.get('file')
        database_choice = request.form.get('database')
        url = request.form.get('url')
        
        logger.info(f"Inject request: database={database_choice}, url={url}")
        
        if not database_choice:
            return jsonify({"error": "Database choice is required"}), 400
        
        if file:
            # Handle file upload
            temp_dir = tempfile.TemporaryDirectory()
            temp_file_path = os.path.join(temp_dir.name, file.filename)
            file.save(temp_file_path)
            
            file_extension = temp_file_path.split(".")[-1].lower()
            
            if file_extension == "pdf":
                if database_choice == "faiss":
                    vector_service.inject_to_faiss(temp_file_path)
                else:
                    vector_service.inject_to_pinecone(temp_file_path)
            else:
                data_ingestion_service.inject_csv(temp_file_path, database_choice)
            
            temp_dir.cleanup()
            
            return jsonify({
                "message": f"Data has been successfully injected into {database_choice}."
            })
        
        elif url:
            # Handle URL
            data_ingestion_service.inject_url(url, database_choice)
            
            return jsonify({
                "message": f"Data has been successfully injected into {database_choice}."
            })
        
        else:
            return jsonify({"error": "Either file or URL is required"}), 400
    
    except Exception as e:
        logger.error(f"Error in inject_to_vector_db: {e}")
        return jsonify({"error": str(e)}), 500


@api.route('/chat_with_pdfs', methods=['POST'])
def chat_with_pdfs():
    """
    Chat with uploaded PDFs
    
    Form data:
        - question: User question
        - file: Optional PDF file
    """
    try:
        question = request.form.get('question')
        file = request.files.get('file')
        
        logger.info(f"Chat with PDFs: question={question}, file={file.filename if file else None}")
        
        if not question:
            return jsonify({"error": "Question is required"}), 400
        
        if file:
            # Process uploaded PDF
            temp_dir = tempfile.TemporaryDirectory()
            temp_pdf_path = os.path.join(temp_dir.name, file.filename)
            file.save(temp_pdf_path)
            
            answer, images_base64 = pdf_service.answer_pdf(question, temp_dir.name)
            
            temp_dir.cleanup()
        else:
            # Use existing index
            answer, images_base64 = pdf_service.answer_only(question)
        
        return jsonify({
            "result": answer,
            "relevant_images": images_base64
        })
    
    except Exception as e:
        logger.error(f"Error in chat_with_pdfs: {e}")
        return jsonify({"error": str(e)}), 500


@api.route('/submit_feedback', methods=['POST'])
def submit_feedback():
    """
    Submit user feedback
    
    Request JSON:
        - message_id: Message identifier
        - response: Model response
        - feedback: User feedback
    """
    try:
        data = request.get_json()
        logger.info(f"Feedback submission: {data}")
        
        if "message_id" not in data or "response" not in data or "feedback" not in data:
            return jsonify({"error": "Invalid data format"}), 400
        
        message_id = data["message_id"]
        response = data["response"]
        feedback = data["feedback"]
        
        save_feedback_to_csv(message_id, response, feedback)
        
        return jsonify({"success": True})
    
    except Exception as e:
        logger.error(f"Error in submit_feedback: {e}")
        return jsonify({"error": str(e)}), 500


@api.route('/submit_report', methods=['POST'])
def submit_report():
    """
    Submit report data
    
    Request JSON:
        - message_id: Message identifier
        - response: Model response
    """
    try:
        data = request.get_json()
        logger.info(f"Report submission: {data}")
        
        if "message_id" not in data or "response" not in data:
            return jsonify({"error": "Invalid data format"}), 400
        
        message_id = data["message_id"]
        response = data["response"]
        
        save_report_to_csv(message_id, response)
        
        return jsonify({"success": True})
    
    except Exception as e:
        logger.error(f"Error in submit_report: {e}")
        return jsonify({"error": str(e)}), 500


@api.route('/report_generation', methods=['POST'])
def report_generate():
    """
    Generate report from last N responses
    
    Request JSON:
        - number: Number of last responses to include
    """
    try:
        data = request.get_json()
        number = data.get("number")
        
        if not number:
            return jsonify({"error": "Missing 'number' field in request"}), 400
        
        no_of_msg = min(int(number), 5)
        
        report_service.report_generation(no_of_msg)
        
        return jsonify({
            "message": "Report was successfully generated",
            "docx_file": "economic_analysis_report.docx",
            "pdf_file": "report.pdf",
        })
    
    except ValueError:
        return jsonify({"error": "'number' must be a valid integer"}), 400
    except Exception as e:
        logger.error(f"Error in report_generation: {e}")
        return jsonify({"error": str(e)}), 500


@api.route('/download_docx/<path:filename>')
def download_docx(filename):
    """Download generated DOCX file"""
    try:
        file_path = Config.REPORTS_PATH / filename
        return send_file(file_path, as_attachment=True)
    except Exception as e:
        logger.error(f"Error downloading DOCX: {e}")
        return jsonify({"error": str(e)}), 404


@api.route('/download_pdf/<path:filename>')
def download_pdf(filename):
    """Download generated PDF file"""
    try:
        file_path = Config.REPORTS_PATH / filename
        return send_file(file_path, as_attachment=True)
    except Exception as e:
        logger.error(f"Error downloading PDF: {e}")
        return jsonify({"error": str(e)}), 404


@api.route('/evaluation', methods=['POST'])
def evaluate_model():
    """
    Evaluate RAG model using RAGAS
    
    Request JSON:
        - model: Model name
        - apikey: OpenAI API key
    """
    try:
        data = request.json
        logger.info(f"Evaluation request: {data}")
        
        model = data.get("model")
        api_key = data.get("apikey")
        
        if not all([model, api_key]):
            return jsonify({"error": "Missing required fields"}), 400
        
        # Use test file from config or default path
        file_path = Config.DATA_PATH / "test_data.csv"
        if not file_path.exists():
            return jsonify({"error": "Test data file not found"}), 404
        
        results = evaluation_service.evaluate_rag_using_ragas(api_key, model, str(file_path))
        results_json = json.dumps(results, default=str)
        
        return jsonify(results_json)
    
    except Exception as e:
        logger.error(f"Error in evaluation: {e}")
        return jsonify({"error": str(e)}), 500

