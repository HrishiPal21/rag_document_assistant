import time
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from typing import List, Dict, Any
from app.rag import get_answer, init_vector_store

app = FastAPI(
    title="RAG Document Assistant API",
    description="A minimal Retrieval-Augmented Generation API to query documents."
)

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    answer: str
    confidence_scores: List[float] = []
    token_usage: Dict[str, Any] = {}
    latency: float = 0.0

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    print(f"INFO:    Request: {request.url.path} | API Latency: {process_time:.4f}s")
    return response

@app.on_event("startup")
async def startup_event():
    # Initialize the RAG system and load documents on startup
    init_vector_store()

@app.post("/ask", response_model=QueryResponse)
async def ask_question(request: QueryRequest):
    try:
        if not request.query.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty.")
            
        result = get_answer(request.query)
        if isinstance(result, str):
            # Fallback if old format is returned somehow
            return QueryResponse(answer=result)
            
        return QueryResponse(
            answer=result.get("answer", ""),
            confidence_scores=result.get("confidence_scores", []),
            token_usage=result.get("token_usage", {}),
            latency=result.get("latency", 0.0)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
