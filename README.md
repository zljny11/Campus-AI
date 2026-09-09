# Campus-AI

**Completed RAG microservice for UniTicket campus-event discovery.**

Python · FastAPI · LangChain/LCEL · Ollama · FAISS · DeepSeek · MySQL

## Overview

Campus-AI converts structured campus-event records into a versioned FAISS index and exposes a REST API used by the [UniTicket](https://github.com/zljny11/uniticket) Spring Boot platform.

~~~mermaid
flowchart TD
    subgraph OFFLINE["Offline Indexing"]
        DB[("MySQL Event Records")] --> ING["Incremental Ingestion"]
        ING --> META["Content Hash + Metadata"]
        META --> EMB["Ollama Embedding"]
        EMB --> IDX[("Versioned FAISS Index")]
    end

    subgraph ONLINE["Online RAG Request"]
        UNI["UniTicket Spring Boot"] -->|"REST"| API["FastAPI /ai/ask"]
        API --> QE["Query Embedding"]
        QE --> RET["Top-K Retrieval"]
        IDX --> CHECK["Index / Model Validation"]
        CHECK --> RET
        RET --> FILTER["Metadata Filtering + Reranking"]
        FILTER --> CTX["Structured Context Selection"]
        CTX --> LLM["DeepSeek-V3 Generation"]
        LLM --> RES["Answer + Sources + Metadata"]
        RES --> UNI
    end

    subgraph EVALUATION["Evaluation"]
        TEST["Fixed Evaluation Set"] --> PIPE["RAG Pipeline"]
        PIPE --> RAGAS["Ragas Faithfulness"]
    end
~~~

## Features

- Asynchronous FastAPI health and question-answering endpoints.
- Ollama mxbai-embed-large embeddings and FAISS retrieval.
- Incremental ingestion with content hashes and versioned index snapshots.
- Metadata-aware reranking using event name, category, venue, time, and tags.
- Structured context selection before DeepSeek generation.
- Index/model compatibility validation and operational response metadata.

On a fixed evaluation set, metadata-aware context selection improved **Ragas Faithfulness from 0.62 to 0.81**.

## API

~~~bash
curl http://localhost:8000/health

curl -X POST http://localhost:8000/ai/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"Which music events are held this week?","top_k":5}'
~~~

Interactive API documentation is available at http://localhost:8000/docs.

## Quick Start

~~~bash
git clone https://github.com/zljny11/Campus-AI.git
cd Campus-AI

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env

ollama pull mxbai-embed-large
python ingest.py
uvicorn main:app --host 0.0.0.0 --port 8000
~~~

Configure MySQL, Ollama, and DeepSeek values in .env before ingestion.

## Project Structure

~~~text
main.py          FastAPI endpoints and RAG pipeline
ingest.py        incremental ingestion and index versioning
vector_store.py  FAISS loading and compatibility checks
schemas.py       request and response models
config.py        environment-based configuration
.env.example     configuration template
~~~

