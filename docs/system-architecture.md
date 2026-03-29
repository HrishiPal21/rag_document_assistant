# System Architecture

This project is organized into three cooperating pipelines:

1. Ingestion and vector indexing on service startup.
2. Online query serving through the `/ask` API endpoint.
3. Offline LLM-as-judge evaluation for quality benchmarking.

```mermaid
flowchart LR
    subgraph Ingestion["Ingestion & Indexing Pipeline (Startup)"]
        A["Text Corpus (data/*.txt)"]
        B["DirectoryLoader + TextLoader"]
        C["RecursiveCharacterTextSplitter<br/>chunk_size=1000, overlap=100"]
        D["HuggingFaceEmbeddings<br/>all-MiniLM-L6-v2"]
        E["FAISS Vector Store (in-memory)"]
        A --> B --> C --> D --> E
    end

    subgraph Serving["Query Serving Pipeline (/ask)"]
        F["Client Query (POST /ask)"]
        G["FastAPI Request Validation<br/>(Pydantic models)"]
        H["get_answer(query)"]
        I["Similarity Search<br/>k=5 + L2 distance scores"]
        J["Context Assembly"]
        K["Gemini 2.5 Flash QA Chain<br/>(LangChain prompt + documents)"]
        L["Structured API Response<br/>answer + confidence_scores + token_usage + latency"]
        F --> G --> H --> I --> J --> K --> L
    end

    subgraph Eval["Offline Evaluation Pipeline"]
        M["eval/llm_as_judge.py"]
        N["Fixed Evaluation Query Set"]
        O["RAG Answer Generation"]
        P["Gemini Judge Prompt<br/>(0-100 relevance scoring)"]
        Q["Per-query + Average Scores"]
        M --> N --> O --> P --> Q
    end

    E --> I
    H --> O
```

## Component Roles

### 1) Ingestion and Indexing

- Reads `.txt` files from `data/`.
- Splits documents into retrieval-friendly chunks.
- Builds local semantic embeddings.
- Stores vectors in FAISS for fast nearest-neighbor search.

### 2) Query Serving

- Accepts user query at `/ask`.
- Validates payload with Pydantic request schema.
- Retrieves top-k semantically similar chunks from FAISS.
- Sends query + context to Gemini through LangChain chain.
- Returns answer and observability fields in one JSON response.

### 3) Evaluation

- Runs fixed benchmark queries.
- Reuses the same RAG answer pipeline.
- Uses a separate judge prompt to assign 0-100 relevance scores.
- Reports per-query metrics and aggregate average.

## Data and Control Flow Notes

- Retrieval confidence values are FAISS distances (lower is generally better).
- Token usage in responses is heuristic and intended for trend monitoring.
- The vector store is in-memory in this version (ephemeral across restarts).
- LLM calls require `GOOGLE_API_KEY` in environment configuration.
