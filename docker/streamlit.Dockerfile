# Streamlit Dockerfile
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY streamlit_requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r streamlit_requirements.txt

# Copy Streamlit app
COPY streamlit_app.py .

# Expose port
EXPOSE 8501

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV BACKEND_API_URL=http://backend:5000

# Run Streamlit
CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]

