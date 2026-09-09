# Campus-AI

**Completed RAG microservice for UniTicket campus-event discovery.**

Python · FastAPI · LangChain/LCEL · Ollama · FAISS · DeepSeek · MySQL

## Overview

Campus-AI converts structured campus-event records into a versioned FAISS index and exposes a REST API used by the [UniTicket](https://github.com/zljny11/uniticket) Spring Boot platform.

~~~mermaid
flowchart LR
    subgraph INDEX["Index Pipeline"]
        direction TB
        DB[("MySQL Events")] --> ING["Incremental Ingestion"]
        ING --> EMB["Hash, Metadata and Embedding"]
        EMB --> FAISS[("Versioned FAISS Index")]
    end

    subgraph SERVICE["Retrieval Service"]
        direction TB
        APP["UniTicket Platform"] --> API["FastAPI REST API"]
        API --> QUERY["Query Embedding"]
        QUERY --> RET["Top-K Retrieval"]
        RET --> RANK["Metadata Filter and Reranking"]
    end

    subgraph GENERATION["Answer Generation"]
        direction TB
        CONTEXT["Structured Context"]
        CONTEXT --> LLM["DeepSeek-V3"]
        LLM --> RESULT["Answer, Sources and Metadata"]
    end

    FAISS --> RET
    RANK --> CONTEXT
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

