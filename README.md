# RAG Document Assistant

A simple beginner-friendly RAG (Retrieval-Augmented Generation) Document Assistant built with FastAPI, LangChain, and FAISS.

## Prerequisites
- Python 3.8+
- OpenAI API Key

## Setup

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Add Documents:**
   Place your `.txt` files in the `data/` folder. A sample document is automatically created on the first start if none exist.

3. **Set your OpenAI API Key:**
   ```bash
   export OPENAI_API_KEY="your-api-key-here"
   ```

## Running the Application

1. **Start the FastAPI server:**
   ```bash
   uvicorn app.main:app --reload
   ```

2. **Access the API Documentation:**
   Open your browser and navigate to: [http://localhost:8000/docs](http://localhost:8000/docs)

## Usage

You can query the assistant via the `/ask` endpoint or using the built-in Swagger UI.

**Example Request (cURL):**
```bash
curl -X POST "http://localhost:8000/ask" \
     -H "Content-Type: application/json" \
     -d '{"query": "What is the fox doing?"}'
```
