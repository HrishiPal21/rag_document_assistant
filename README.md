# RAG Document Assistant

An advanced, production-ready Retrieval-Augmented Generation (RAG) Document Assistant built with FastAPI, LangChain, and FAISS. 

This project demonstrates a fully instrumented document synthesis pipeline capable of ingesting and processing large-scale document datasets, alongside rigorous, automated LLM-as-judge evaluation.

## Key Features
- **Scalable Document Synthesis Pipeline**: Engineered to process **3K+ records** via custom LangChain retrieval chains, seamlessly enabling conversational Q&A from large-scale source contexts.
- **Advanced Semantic Retrieval**: Built customized document chunking strategies and local HuggingFace embedding workflows.
- **LLM-as-Judge Evaluation Framework**: Includes a strict, reproducible evaluation script (`eval/llm_as_judge.py`) that benchmarks answer relevance, objectively tracking quality improvements.
- **FastAPI-Level Monitoring**: Fully instrumented endpoints that track critical metadata per run:
  - Retrieval **confidence scores** (L2 similarity distances from FAISS)
  - End-to-end API **latency**
  - **Token usage** for systematic debugging of response quality and costs.

## Prerequisites
- Python 3.8+
- Google API Key (Gemini)

## Setup

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Generate Test Data (Optional):**
   To simulate a production-scale 3,000+ document dataset locally, run the data synthesis script:
   ```bash
   python data/generate_records.py
   ```

3. **Set your Google API Key:**
   Get a free API key from [Google AI Studio](https://aistudio.google.com/app/apikey).
   ```bash
   export GOOGLE_API_KEY="your-api-key-here"
   ```

## Running the Application

1. **Start the FastAPI server:**
   ```bash
   uvicorn app.main:app --reload
   ```

2. **Access the API Documentation:**
   Navigate to: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) to explore the endpoint and test the real-time monitoring metrics.

## Running the Evaluation Framework

To run the structured LLM-as-judge evaluation suite across test queries:
```bash
python eval/llm_as_judge.py
```
This script queries the pipeline, evaluates the generated answer relevance using Gemini as an impartial judge, and outputs metrics directly in the terminal to measure performance improvements.
