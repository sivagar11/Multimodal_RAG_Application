"""
Streamlit Frontend Application
Connects to Flask backend API
"""
import os
import streamlit as st
import requests
from PIL import Image
import io
import base64

# Get backend API URL from environment variable
BACKEND_API_URL = os.getenv("BACKEND_API_URL", "http://localhost:5000")


def call_api(endpoint, method="POST", data=None, files=None):
    """
    Generic API call helper
    
    Args:
        endpoint: API endpoint path
        method: HTTP method
        data: JSON data or form data
        files: Files to upload
        
    Returns:
        API response
    """
    url = f"{BACKEND_API_URL}/api/{endpoint}"
    
    try:
        if method == "POST":
            if files:
                response = requests.post(url, data=data, files=files)
            else:
                response = requests.post(url, json=data)
        elif method == "GET":
            response = requests.get(url, params=data)
        else:
            raise ValueError(f"Unsupported method: {method}")
        
        response.raise_for_status()
        return response.json()
    
    except requests.exceptions.RequestException as e:
        st.error(f"API Error: {e}")
        return None


def chat_with_pdfs_tab():
    """Chat with PDFs tab"""
    st.header("💬 Chat with PDFs")
    
    question_input = st.text_input("Enter your question")
    pdf_input = st.file_uploader("Upload PDF (optional)", type="pdf")
    
    if st.button("Submit", key="chat_submit"):
        if question_input:
            with st.spinner("Processing..."):
                files = None
                if pdf_input is not None:
                    files = {"file": (pdf_input.name, pdf_input, "application/pdf")}
                
                data = {"question": question_input}
                result = call_api("chat_with_pdfs", data=data, files=files)
                
                if result:
                    st.subheader("Answer:")
                    st.write(result.get("result", "No answer generated"))
                    
                    # Display relevant images
                    if result.get("relevant_images"):
                        st.subheader("Relevant Images:")
                        for img_base64 in result["relevant_images"]:
                            try:
                                image = Image.open(io.BytesIO(base64.b64decode(img_base64)))
                                st.image(image, caption="Relevant Image")
                            except Exception as e:
                                st.warning(f"Could not display image: {e}")
        else:
            st.warning("Please enter a question")


def inject_to_vector_db_tab():
    """Inject to Vector DB tab"""
    st.header("📚 Inject to Vector Database")
    
    input_type = st.radio("Input Type", ["File", "URL"])
    
    if input_type == "File":
        pdf_input_db = st.file_uploader("Upload PDF or CSV", type=["pdf", "csv"])
        url_input = None
    else:
        pdf_input_db = None
        url_input = st.text_input("Enter URL")
    
    database_choice = st.radio("Database Choice", ["faiss", "pinecone"])
    
    if st.button("Inject Data", key="inject_submit"):
        with st.spinner("Injecting data..."):
            files = None
            data = {"database": database_choice}
            
            if pdf_input_db is not None:
                files = {"file": (pdf_input_db.name, pdf_input_db)}
                data["url"] = ""
            elif url_input:
                data["url"] = url_input
            else:
                st.warning("Please provide a file or URL")
                return
            
            result = call_api("inject_to_vector_db", data=data, files=files)
            
            if result:
                st.success(result.get("message", "Data injected successfully"))


def rag_application_tab():
    """RAG Application tab"""
    st.header("🤖 RAG Application")
    
    question_rag = st.text_input("Enter your question", key="rag_question")
    
    use_own_model = st.radio(
        "Model Option",
        ["Use Existing Model", "Use Own Model"],
        key="model_option"
    )
    
    database_name = st.radio("Database Name", ["faiss", "pinecone"], key="db_name")
    
    if use_own_model == "Use Existing Model":
        model_selection_rag = st.radio(
            "Model Name",
            ["gpt-4", "gpt-3.5-turbo", "claude-2", "mistral", "llama2-7b", "gemma-7b"],
            key="model_selection"
        )
        own_model_name = ""
        api_key = ""
    else:
        model_selection_rag = ""
        own_model_name = st.text_input("Own Model Name", key="own_model")
        api_key = st.text_input("API Key", type="password", key="api_key")
    
    if st.button("Submit", key="rag_submit"):
        if question_rag:
            with st.spinner("Generating answer..."):
                model = model_selection_rag if use_own_model == "Use Existing Model" else own_model_name
                
                data = {
                    "question": question_rag,
                    "model": model,
                    "database": database_name,
                    "apikey": api_key if api_key else None
                }
                
                result = call_api("answer_question", data=data)
                
                if result:
                    st.subheader("Answer:")
                    st.write(result.get("result", "No answer generated"))
                    
                    # Display relevant images
                    if result.get("relevant_images"):
                        st.subheader("Relevant Images:")
                        for img_base64 in result["relevant_images"]:
                            try:
                                image = Image.open(io.BytesIO(base64.b64decode(img_base64)))
                                st.image(image, caption="Relevant Image")
                            except Exception as e:
                                st.warning(f"Could not display image: {e}")
                    
                    # Display URLs
                    if result.get("urls"):
                        st.subheader("Related URLs:")
                        for url in result["urls"]:
                            st.write(f"- {url}")
        else:
            st.warning("Please enter a question")


def finetuning_tab():
    """Finetuning tab"""
    st.header("🔧 Fine-tuning")
    
    fine_tune_file_input = st.file_uploader("File for Fine-tuning", type=["csv"])
    fine_tune_suffix_name = st.text_input("Suffix Name for Model")
    fine_tune_api_key = st.text_input("Your API Key", type="password", key="finetune_api")
    
    if st.button("Start Fine-tuning", key="finetune_submit"):
        if all([fine_tune_file_input, fine_tune_suffix_name, fine_tune_api_key]):
            with st.spinner("Fine-tuning in progress (this may take several minutes)..."):
                files = {"file": (fine_tune_file_input.name, fine_tune_file_input)}
                data = {
                    "suffixName": fine_tune_suffix_name,
                    "apiKey": fine_tune_api_key
                }
                
                result = call_api("finetune", data=data, files=files)
                
                if result:
                    if result.get("error"):
                        st.error(f"Error: {result['error']}")
                    else:
                        st.success(f"Fine-tuned Model ID: {result.get('model_id')}")
        else:
            st.warning("Please fill in all fields")


def report_generation_tab():
    """Report Generation tab"""
    st.header("📄 Report Generation")
    
    number_of_responses = st.number_input(
        "Number of last responses to include (max 5)",
        min_value=1,
        max_value=5,
        value=3
    )
    
    if st.button("Generate Report", key="report_submit"):
        with st.spinner("Generating report..."):
            data = {"number": number_of_responses}
            result = call_api("report_generation", data=data)
            
            if result:
                if result.get("error"):
                    st.error(f"Error: {result['error']}")
                else:
                    st.success(result.get("message", "Report generated successfully"))
                    
                    # Provide download links
                    st.markdown("### Download Report")
                    st.markdown(f"[Download DOCX]({BACKEND_API_URL}/api/download_docx/economic_analysis_report.docx)")
                    st.markdown(f"[Download PDF]({BACKEND_API_URL}/api/download_pdf/report.pdf)")


def evaluation_tab():
    """Evaluation tab"""
    st.header("📊 Model Evaluation")
    
    model_to_evaluate = st.text_input("Model Name", value="gpt-3.5-turbo")
    eval_api_key = st.text_input("OpenAI API Key", type="password", key="eval_api")
    
    if st.button("Run Evaluation", key="eval_submit"):
        if all([model_to_evaluate, eval_api_key]):
            with st.spinner("Running evaluation (this may take a while)..."):
                data = {
                    "model": model_to_evaluate,
                    "apikey": eval_api_key
                }
                
                result = call_api("evaluation", data=data)
                
                if result:
                    st.subheader("Evaluation Results:")
                    st.json(result)
        else:
            st.warning("Please fill in all fields")


def main():
    """Main Streamlit app"""
    st.set_page_config(
        page_title="Multimodal LLM Application",
        page_icon="🤖",
        layout="wide"
    )
    
    st.title("🤖 Multimodal LLM Application")
    st.markdown("---")
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    tab = st.sidebar.radio(
        "Select Feature",
        [
            "Chat with PDFs",
            "Inject to Vector DB",
            "RAG Application",
            "Fine-tuning",
            "Report Generation",
            "Evaluation"
        ]
    )
    
    # Display selected tab
    if tab == "Chat with PDFs":
        chat_with_pdfs_tab()
    elif tab == "Inject to Vector DB":
        inject_to_vector_db_tab()
    elif tab == "RAG Application":
        rag_application_tab()
    elif tab == "Fine-tuning":
        finetuning_tab()
    elif tab == "Report Generation":
        report_generation_tab()
    elif tab == "Evaluation":
        evaluation_tab()
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.info(
        f"**Backend API:** {BACKEND_API_URL}\n\n"
        "This application connects to a Flask backend API."
    )


if __name__ == "__main__":
    main()

