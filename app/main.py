from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from app.rag import get_answer, init_vector_store

app = FastAPI(
    title="RAG Document Assistant API",
    description="A minimal Retrieval-Augmented Generation API to query documents."
)

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    answer: str

@app.on_event("startup")
async def startup_event():
    # Initialize the RAG system and load documents on startup
    init_vector_store()

@app.post("/ask", response_model=QueryResponse)
async def ask_question(request: QueryRequest):
    try:
        if not request.query.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty.")
            
        answer = get_answer(request.query)
        return QueryResponse(answer=answer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
