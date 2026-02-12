# API Documentation

Complete REST API documentation for the Multimodal RAG Application.

## Base URL

- Local: `http://localhost:5000`
- Docker: `http://backend:5000` (internal), `http://localhost:5000` (external)

## Authentication

Currently, the API does not require authentication for most endpoints. API keys for external services (OpenAI, etc.) are provided in request bodies or configured via environment variables.

## Endpoints

### Health Check

#### GET `/api/hello`

Health check endpoint to verify the API is running.

**Response:**
```json
{
  "message": "Hello, World!"
}
```

---

### RAG Question Answering

#### POST `/api/answer_question`

Answer a question using RAG with vector database context.

**Request Body:**
```json
{
  "question": "What was the revenue in 2023?",
  "model": "gpt-4",
  "database": "faiss",
  "apikey": "optional-custom-api-key"
}
```

**Parameters:**
- `question` (string, required): User question
- `model` (string, required): Model name (gpt-4, gpt-3.5-turbo, claude-2, mistral, llama2-7b, gemma-7b, or custom model ID)
- `database` (string, required): Vector database (faiss or pinecone)
- `apikey` (string, optional): Custom API key for own models

**Response:**
```json
{
  "result": "The revenue in 2023 was...",
  "relevant_images": ["base64_encoded_image1", "base64_encoded_image2"],
  "urls": ["https://example.com/related1", "https://example.com/related2"]
}
```

---

### Chat with PDFs

#### POST `/api/chat_with_pdfs`

Chat with uploaded PDFs or use existing index.

**Request (multipart/form-data):**
- `question` (string, required): User question
- `file` (file, optional): PDF file to process

**Response:**
```json
{
  "result": "Based on the document...",
  "relevant_images": ["base64_encoded_image1"]
}
```

---

### Inject to Vector Database

#### POST `/api/inject_to_vector_db`

Inject documents into vector database from PDF, CSV, or URL.

**Request (multipart/form-data):**
- `file` (file, optional): PDF or CSV file
- `url` (string, optional): URL to scrape
- `database` (string, required): Vector database (faiss or pinecone)

**Response:**
```json
{
  "message": "Data has been successfully injected into faiss."
}
```

---

### Fine-tuning

#### POST `/api/finetune`

Fine-tune a GPT-3.5-turbo model on custom dataset.

**Request (multipart/form-data):**
- `file` (file, required): CSV file with Question and Answer columns
- `suffixName` (string, required): Suffix for the fine-tuned model
- `apiKey` (string, required): OpenAI API key

**Response:**
```json
{
  "model_id": "ft:gpt-3.5-turbo-0125:org:suffix:id"
}
```

**Note:** Fine-tuning takes approximately 5 minutes. The endpoint waits for completion.

---

### Evaluation

#### POST `/api/evaluation`

Evaluate RAG performance using RAGAS metrics.

**Request Body:**
```json
{
  "model": "gpt-3.5-turbo",
  "apikey": "your-openai-api-key"
}
```

**Response:**
```json
{
  "faithfulness": 0.95,
  "answer_relevancy": 0.92,
  "context_recall": 0.88,
  "context_precision": 0.90,
  "answer_correctness": 0.87
}
```

**Note:** Requires test data CSV at `data/test_data.csv` with columns: question, ground_truth

---

### Feedback

#### POST `/api/submit_feedback`

Submit user feedback on a response.

**Request Body:**
```json
{
  "message_id": "unique-message-id",
  "response": "The model's response...",
  "feedback": "positive"
}
```

**Response:**
```json
{
  "success": true
}
```

---

### Report Data

#### POST `/api/submit_report`

Save report data for later aggregation.

**Request Body:**
```json
{
  "message_id": "unique-message-id",
  "response": "Financial analysis shows..."
}
```

**Response:**
```json
{
  "success": true
}
```

---

### Report Generation

#### POST `/api/report_generation`

Generate a financial report from last N responses.

**Request Body:**
```json
{
  "number": 3
}
```

**Parameters:**
- `number` (integer, required): Number of last responses to include (max 5)

**Response:**
```json
{
  "message": "Report was successfully generated",
  "docx_file": "economic_analysis_report.docx",
  "pdf_file": "report.pdf"
}
```

---

### Download Reports

#### GET `/api/download_docx/<filename>`

Download generated DOCX report.

**Example:**
```
GET /api/download_docx/economic_analysis_report.docx
```

#### GET `/api/download_pdf/<filename>`

Download generated PDF report.

**Example:**
```
GET /api/download_pdf/report.pdf
```

---

## Error Responses

All endpoints return standard error responses:

**400 Bad Request:**
```json
{
  "error": "Missing required fields"
}
```

**404 Not Found:**
```json
{
  "error": "Resource not found"
}
```

**500 Internal Server Error:**
```json
{
  "error": "Detailed error message"
}
```

---

## Rate Limiting

Currently, there is no rate limiting implemented. Consider adding rate limiting in production.

---

## CORS

CORS is enabled for all origins by default. Configure `CORS_ORIGINS` in `.env` to restrict access:

```bash
CORS_ORIGINS=http://localhost:8501,https://yourdomain.com
```

---

## Examples

### Python Example

```python
import requests

# Answer a question
response = requests.post(
    "http://localhost:5000/api/answer_question",
    json={
        "question": "What was the profit margin?",
        "model": "gpt-4",
        "database": "faiss"
    }
)

data = response.json()
print(data["result"])
```

### cURL Example

```bash
# Upload and inject PDF
curl -X POST http://localhost:5000/api/inject_to_vector_db \
  -F "file=@document.pdf" \
  -F "database=faiss"

# Ask a question
curl -X POST http://localhost:5000/api/answer_question \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is the revenue?",
    "model": "gpt-4",
    "database": "faiss"
  }'
```

### JavaScript Example

```javascript
// Fetch answer
const response = await fetch('http://localhost:5000/api/answer_question', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    question: 'What are the key findings?',
    model: 'gpt-4',
    database: 'faiss'
  })
});

const data = await response.json();
console.log(data.result);
```

---

## Webhook Support

Currently, webhooks are not supported. All operations are synchronous.

---

## Versioning

Current API version: v1 (implicit, no version prefix in URLs)

Future versions may use URL prefixes like `/api/v2/...`

