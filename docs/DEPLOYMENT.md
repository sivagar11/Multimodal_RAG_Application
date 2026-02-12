# Deployment Guide

Complete guide for deploying the Multimodal RAG Application in various environments.

## Table of Contents

1. [Local Development](#local-development)
2. [Docker Deployment](#docker-deployment)
3. [Cloud Deployment](#cloud-deployment)
4. [Production Considerations](#production-considerations)
5. [Troubleshooting](#troubleshooting)

---

## Local Development

### Prerequisites

- Python 3.10 or higher
- pip package manager
- Virtual environment tool (venv, conda, etc.)
- At least 4GB RAM
- OpenAI API key

### Setup Steps

1. **Create Virtual Environment**
   ```bash
   python3.10 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install Dependencies**
   ```bash
   # Backend
   cd backend
   pip install -r requirements.txt
   
   # Frontend
   cd ../frontend
   pip install -r requirements.txt
   ```

3. **Configure Environment**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

4. **Initialize Vector Stores**
   ```bash
   mkdir -p vector_stores/faiss
   mkdir -p outputs/reports outputs/feedback
   ```

5. **Run Backend**
   ```bash
   cd backend
   python main.py
   ```
   Backend runs on http://localhost:5000

6. **Run Frontend (separate terminal)**
   ```bash
   cd frontend
   streamlit run streamlit_app.py
   ```
   Frontend runs on http://localhost:8501

---

## Docker Deployment

### Prerequisites

- Docker 20.10+
- Docker Compose 2.0+
- 8GB RAM recommended
- API keys configured in `.env`

### Quick Start

1. **Setup Environment**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

2. **Build and Run**
   ```bash
   cd docker
   docker-compose up --build
   ```

3. **Access Application**
   - Frontend: http://localhost:8501
   - Backend API: http://localhost:5000

### Production Build

```bash
# Build images
docker-compose build --no-cache

# Run in detached mode
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

### Persist Data

Volumes are automatically created for:
- `vector_stores/` - Vector database indexes
- `outputs/` - Generated reports and feedback
- `logs/` - Application logs

To backup:
```bash
docker-compose down
tar -czf backup.tar.gz vector_stores outputs logs
```

To restore:
```bash
tar -xzf backup.tar.gz
docker-compose up -d
```

---

## Cloud Deployment

### AWS Deployment

#### Using EC2

1. **Launch EC2 Instance**
   - Instance type: t3.large or larger
   - OS: Ubuntu 22.04 LTS
   - Storage: 50GB+ EBS volume
   - Security group: Allow ports 5000, 8501, 22

2. **Install Docker**
   ```bash
   sudo apt update
   sudo apt install -y docker.io docker-compose
   sudo usermod -aG docker ubuntu
   ```

3. **Deploy Application**
   ```bash
   git clone <repository>
   cd hackathon
   cp .env.example .env
   # Configure .env with AWS Secrets Manager or directly
   
   cd docker
   sudo docker-compose up -d
   ```

4. **Setup Nginx Reverse Proxy**
   ```nginx
   server {
       listen 80;
       server_name your-domain.com;
       
       location / {
           proxy_pass http://localhost:8501;
           proxy_http_version 1.1;
           proxy_set_header Upgrade $http_upgrade;
           proxy_set_header Connection "upgrade";
       }
       
       location /api {
           proxy_pass http://localhost:5000;
       }
   }
   ```

#### Using ECS (Elastic Container Service)

1. **Push Images to ECR**
   ```bash
   aws ecr create-repository --repository-name hackathon-backend
   aws ecr create-repository --repository-name hackathon-frontend
   
   # Build and push
   docker build -f docker/backend.Dockerfile -t hackathon-backend .
   docker tag hackathon-backend:latest <account-id>.dkr.ecr.<region>.amazonaws.com/hackathon-backend:latest
   docker push <account-id>.dkr.ecr.<region>.amazonaws.com/hackathon-backend:latest
   
   # Repeat for frontend
   ```

2. **Create ECS Task Definition**
   - Define backend and frontend services
   - Set environment variables
   - Configure networking

3. **Deploy to ECS Cluster**
   - Create ECS cluster
   - Create service from task definition
   - Configure Application Load Balancer

### Google Cloud Platform (GCP)

#### Using Cloud Run

1. **Build and Push to GCR**
   ```bash
   gcloud builds submit --config cloudbuild.yaml
   ```

2. **Deploy Backend**
   ```bash
   gcloud run deploy hackathon-backend \
     --image gcr.io/PROJECT_ID/hackathon-backend \
     --platform managed \
     --region us-central1 \
     --allow-unauthenticated \
     --set-env-vars FLASK_ENV=production
   ```

3. **Deploy Frontend**
   ```bash
   gcloud run deploy hackathon-frontend \
     --image gcr.io/PROJECT_ID/hackathon-frontend \
     --platform managed \
     --region us-central1 \
     --allow-unauthenticated \
     --set-env-vars BACKEND_API_URL=https://backend-url
   ```

### Azure Deployment

#### Using Container Instances

1. **Create Container Registry**
   ```bash
   az acr create --resource-group myResourceGroup \
     --name hackathonRegistry --sku Basic
   ```

2. **Build and Push Images**
   ```bash
   az acr build --registry hackathonRegistry \
     --image hackathon-backend:latest \
     --file docker/backend.Dockerfile .
   ```

3. **Deploy Container**
   ```bash
   az container create \
     --resource-group myResourceGroup \
     --name hackathon-backend \
     --image hackathonRegistry.azurecr.io/hackathon-backend:latest \
     --dns-name-label hackathon-backend \
     --ports 5000
   ```

---

## Production Considerations

### Security

1. **Environment Variables**
   - Never commit `.env` to version control
   - Use secrets management (AWS Secrets Manager, Azure Key Vault, etc.)
   - Rotate API keys regularly

2. **HTTPS**
   - Use SSL/TLS certificates (Let's Encrypt)
   - Configure reverse proxy (Nginx, Caddy)
   - Redirect HTTP to HTTPS

3. **API Security**
   - Implement rate limiting
   - Add authentication/authorization
   - Enable request validation
   - Use API keys for external access

4. **CORS**
   - Restrict CORS origins in production
   - Use specific domains instead of `*`

### Performance

1. **Scaling**
   - Use load balancer for multiple instances
   - Implement caching (Redis)
   - Consider CDN for static assets

2. **Database**
   - Use Pinecone for production vector database
   - Implement connection pooling
   - Add database backups

3. **Monitoring**
   - Setup application monitoring (Datadog, New Relic)
   - Configure log aggregation (ELK, CloudWatch)
   - Set up alerts for errors

4. **Resource Limits**
   - Set memory limits in Docker
   - Configure request timeouts
   - Implement job queues for long-running tasks

### Backup and Recovery

1. **Vector Databases**
   ```bash
   # Backup FAISS indexes
   tar -czf vector_stores_backup_$(date +%Y%m%d).tar.gz vector_stores/
   
   # Restore
   tar -xzf vector_stores_backup_YYYYMMDD.tar.gz
   ```

2. **Outputs and Logs**
   - Regular backups to S3/Cloud Storage
   - Implement log rotation
   - Archive old reports

3. **Configuration**
   - Version control for configuration
   - Document environment variables
   - Keep backup of `.env` (securely)

### Optimization

1. **Docker Images**
   ```dockerfile
   # Use multi-stage builds
   FROM python:3.10-slim as builder
   # ... build dependencies
   
   FROM python:3.10-slim
   COPY --from=builder /app /app
   ```

2. **Caching**
   - Cache vector database connections
   - Implement response caching
   - Use CDN for static assets

3. **Resource Management**
   - Limit concurrent requests
   - Implement request queuing
   - Use connection pooling

---

## Troubleshooting

### Common Issues

**Port Already in Use**
```bash
# Find process using port
lsof -i :5000
# Kill process
kill -9 <PID>
```

**Docker Build Fails**
```bash
# Clean Docker cache
docker system prune -a
# Rebuild without cache
docker-compose build --no-cache
```

**Out of Memory**
```bash
# Increase Docker memory limit
# Docker Desktop -> Settings -> Resources -> Memory

# Or in docker-compose.yml
services:
  backend:
    mem_limit: 4g
```

**Vector Database Not Found**
```bash
# Ensure vector stores directory exists
mkdir -p vector_stores/faiss
# Check permissions
chmod -R 755 vector_stores/
```

### Logs and Debugging

**View Application Logs**
```bash
# Docker logs
docker-compose logs -f backend
docker-compose logs -f frontend

# Local logs
tail -f logs/app.log
```

**Debug Mode**
```bash
# Enable debug in .env
FLASK_ENV=development
LOG_LEVEL=DEBUG

# Restart services
docker-compose restart
```

### Health Checks

**Check Backend**
```bash
curl http://localhost:5000/api/hello
```

**Check Frontend**
```bash
curl http://localhost:8501
```

**Check Docker Services**
```bash
docker-compose ps
docker stats
```

---

## Maintenance

### Updates

```bash
# Pull latest code
git pull origin main

# Rebuild and restart
docker-compose down
docker-compose up --build -d
```

### Cleanup

```bash
# Remove unused Docker resources
docker system prune -a --volumes

# Clean old logs
find logs/ -name "*.log" -mtime +30 -delete

# Archive old reports
tar -czf reports_archive_$(date +%Y%m).tar.gz outputs/reports/
```

---

## Support

For deployment issues:
1. Check logs first
2. Verify environment variables
3. Ensure all required services are running
4. Check network connectivity
5. Review resource usage

For additional help, open an issue in the repository.

