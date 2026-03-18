import os
import sys
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# Global variables for the vector store and RAG chain
vector_store = None
rag_chain = None

def init_vector_store():
    global vector_store, rag_chain
    print("Initializing vector store...")
    
    # Define paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data")
    
    # Ensure data directory exists
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        
    # Check if there are any .txt files, if not, create a sample one
    has_txt = any(f.endswith(".txt") for f in os.listdir(data_dir))
    if not has_txt:
        sample_file = os.path.join(data_dir, "sample.txt")
        with open(sample_file, "w", encoding="utf-8") as f:
            f.write("The quick brown fox jumps over the lazy dog.\n\n"
                    "This is a sample document for testing the RAG assistant.\n"
                    "FastAPI makes it easy to create APIs, and LangChain makes it easy "
                    "to work with Large Language Models and vector databases like FAISS.")
        print(f"Created a sample document at {sample_file}")

    # Process documents
    print(f"Loading documents from {data_dir}...")
    loader = DirectoryLoader(data_dir, glob="**/*.txt", loader_cls=TextLoader)
    documents = loader.load()
    
    if not documents:
        print("No documents found in the data directory.")
        return

    # Split texts
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.split_documents(documents)
    print(f"Loaded {len(documents)} documents, split into {len(texts)} chunks.")
    
    if not os.environ.get("OPENAI_API_KEY"):
        print("Warning: OPENAI_API_KEY environment variable is not set. "
              "Vector store initialization requires an API key.", file=sys.stderr)
        return

    try:
        # Create embeddings and vector store
        embeddings = OpenAIEmbeddings()
        vector_store = FAISS.from_documents(texts, embeddings)
        
        # Initialize LLM
        llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
        
        # Create prompt
        system_prompt = (
            "You are an assistant for question-answering tasks.\n"
            "Use the following pieces of retrieved context to answer the question.\n"
            "If you don't know the answer, say that you don't know.\n"
            "Keep the answer concise.\n\n"
            "Context:\n"
            "{context}"
        )
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
        ])
        
        # Create retrieval and Q&A chain
        retriever = vector_store.as_retriever()
        question_answer_chain = create_stuff_documents_chain(llm, prompt)
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)
        print("Vector store and RAG chain initialized successfully.")
    except Exception as e:
        print(f"Failed to initialize RAG components. Error: {e}", file=sys.stderr)

def get_answer(query: str) -> str:
    global rag_chain
    if rag_chain is None:
        return ("System is not properly initialized. This usually means the "
                "OPENAI_API_KEY environment variable was not set when starting the server.")
        
    try:
        res = rag_chain.invoke({"input": query})
        return res.get("answer", "No answer found.")
    except Exception as e:
        return f"An error occurred while getting the answer: {str(e)}"
