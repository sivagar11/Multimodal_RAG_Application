# Vector Stores

This directory contains the vector database indexes used by the application.

## Structure

- `faiss/` - FAISS vector database indexes
  - `faiss_index_tea/` - Primary index for Ceylon Tea Brokers documents

## FAISS Index

The FAISS index stores embeddings of:
- Text chunks from annual reports
- Table summaries
- Image descriptions from financial charts and graphs

Each document in the index has metadata:
- `id`: Unique identifier
- `type`: Document type (text, table, or image)
- `original_content`: Full original content or base64-encoded image

## Usage

The vector databases are automatically loaded by the backend services when needed. No manual intervention is required.

## Rebuilding Indexes

To rebuild or add documents to the indexes, use the "Inject to Vector DB" feature in the application.

## Storage

FAISS indexes are stored locally as binary files. The main index files are:
- `index.faiss` - The FAISS index
- `index.pkl` - Metadata pickle file

**Note:** These files are excluded from version control due to their size. You'll need to rebuild them or restore from backup when setting up a new environment.

