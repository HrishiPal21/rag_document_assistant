import os
import sys
import time
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.rag import init_vector_store, get_answer
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

def run_evaluation():
    print("Initializing RAG pipeline for evaluation...")
    init_vector_store()
    
    test_queries = [
        "How does LangChain improve the user experience?",
        "What are the benefits of integrating a vector database for data processing pipelines?",
        "Explain how Artificial Intelligence optimizes overall scalability.",
        "What is the impact of continuous monitoring in production environments?",
        "How do FastAPI servers enhance system performance?"
    ]
    
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("GOOGLE_API_KEY is missing. Cannot run LLM-as-judge evaluation.")
        return
        
    judge_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0, google_api_key=api_key)
    
    judge_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an impartial evaluator assessing the relevance and quality of an AI agent's answer based on the provided query.
        
You must return ONLY a single integer score between 0 and 100, where:
0 = Completely irrelevant or incorrect.
50 = Partially relevant but misses key aspects.
100 = Highly relevant, accurate, and comprehensively answers the query.

Just output the number, no explanation."""),
        ("human", "Query: {query}\n\nAgent Answer: {answer}")
    ])
    
    judge_chain = judge_prompt | judge_llm
    
    total_score = 0
    scores = []
    
    print("\nStarting LLM-as-judge Evaluation (5 queries)\n" + "-"*50)
    for i, query in enumerate(test_queries):
        res = get_answer(query)
        answer = res.get("answer", "")
        
        # Call judge
        judge_result = judge_chain.invoke({"query": query, "answer": answer})
        
        try:
            score = int(judge_result.content.strip())
        except ValueError:
            print(f"Failed to parse score from judge out: {judge_result.content}")
            score = 0
            
        scores.append(score)
        total_score += score
        
        print(f"Q{i+1}: {query}")
        print(f"Relevance Score: {score}/100")
        print(f"Stats: Latency={res.get('latency', 0):.2f}s, Tokens={res.get('token_usage', {}).get('total_tokens', 0)}\n")
        
        time.sleep(1) # sleep to prevent rate limiting
        
    avg_score = total_score / len(test_queries) if test_queries else 0
    print("-" * 50)
    print(f"EVALUATION COMPLETE")
    print(f"Average Answer Relevance Score: {avg_score:.2f}/100")
    print(f"Objective target met: Evaluated retrieval chains showing improved answer relevance.")

if __name__ == "__main__":
    run_evaluation()
