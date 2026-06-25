# Advanced Medical Chatbot System

A highly scalable, microservices-based AI medical chatbot designed to provide accurate, structured, and compassionate medical information. 

This project is a significantly enhanced and refactored version of a foundational medical chatbot, upgraded with a modern technology stack, advanced Retrieval-Augmented Generation (RAG) capabilities, and robust enterprise-grade observability.

## Key Features & Architectural Enhancements

- **Modern & Interactive UI**: Upgraded the user interface from a basic Flask template to **Chainlit**, delivering a premium, seamless conversational AI experience with streaming support.
- **Microservices Architecture**: Decoupled the monolithic design into modular services:
  - **gRPC Backend**: Handles the intensive LangChain RAG pipeline, ensuring high-performance and low-latency internal communication.
  - **FastAPI Gateway**: Acts as a lightweight API layer streaming Server-Sent Events (SSE) to the frontend.
- **Advanced RAG Pipeline**: 
  - Integrated **Cohere Rerank** to optimize document relevance post-retrieval.
  - Utilizes **Pinecone Vector Store** for rapid similarity search.
  - Implemented an intelligent **History-Aware Retriever** to seamlessly translate and contextualize user queries.
- **Persistent Memory**: Integrated **PostgreSQL** (`langchain-postgres`) to maintain persistent and reliable chat histories across user sessions.
- **Observability & Tracing**: Fully integrated with **Langfuse** to trace LLM executions, monitor token usage, and debug the RAG pipeline in real time.

## Technology Stack

- **Frontend**: Chainlit
- **Backend/API**: FastAPI, gRPC
- **AI/LLM**: OpenAI (GPT-4o-mini), LangChain
- **Database & Retrieval**: Pinecone (VectorDB), PostgreSQL (Memory), Cohere (Reranking)
- **Observability**: Langfuse

## Setup & Installation

### 1. Prerequisites
Ensure you have Python 3.10+ installed and create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Environment Variables
Create a `.env` file in the root directory and configure the following credentials:
```ini
OPENAI_API_KEY="your_openai_key"
PINECONE_API_KEY="your_pinecone_key"
COHERE_API_KEY="your_cohere_key"
DB_URI="your_postgresql_connection_string"

# Langfuse Observability
LANGFUSE_SECRET_KEY="sk-lf-..."
LANGFUSE_PUBLIC_KEY="pk-lf-..."
LANGFUSE_BASE_URL="https://cloud.langfuse.com"

# Chainlit Auth (Optional)
CHAINLIT_AUTH_SECRET="your_chainlit_secret"
```

### 3. Data Ingestion
Process the raw medical CSV datasets and populate the Pinecone vector database:
```bash
python scripts/store_index.py
```

### 4. Running the Services
Because of the microservices architecture, you will need to start the services in separate terminal instances:

**Terminal 1: Start the gRPC AI Backend**
```bash
python backend/grpc_server.py
```

**Terminal 2: Start the FastAPI Streaming Gateway**
```bash
uvicorn backend.fastapi_app:app --port 8080
```

**Terminal 3: Start the Chainlit UI**
```bash
cd frontend
chainlit run app.py -w
```
Your chatbot will now be accessible at `http://localhost:8000`!

## Future Roadmap

In the next phase of development, the system will be containerized and deployed to the cloud for production readiness:
- **Dockerization**: Wrap the Chainlit frontend, FastAPI gateway, and gRPC backend into dedicated Docker containers using `docker-compose`.
- **Cloud Deployment**: Push the Docker images to AWS ECR and deploy the cluster on **AWS ECS (Elastic Container Service)** to guarantee high availability and scalable infrastructure.

## Acknowledgements

Original Repository: [Build-a-Complete-Medical-Chatbot-with-LLMs-LangChain-Pinecone-Flask-AWS](https://github.com/entbappy/Build-a-Complete-Medical-Chatbot-with-LLMs-LangChain-Pinecone-Flask-AWS)
