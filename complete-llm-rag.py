import pandas as pd
from langchain_chroma import Chroma  # Updated import
from langchain_core.documents import Document
import os
import requests
import json
from langchain_core.embeddings import Embeddings
from typing import List
import shutil
import time
import chromadb

# Custom LM Studio Embeddings class
class LMStudioEmbeddings(Embeddings):
    """LM Studio embeddings wrapper for LangChain."""

    def __init__(
        self,
        model: str = "text-embedding-medical-10-10-1-jinaai_jina-embeddings-v2-small-en-50-gpt-3.5-turbo-01_9062874564-i1",
        base_url: str = "http://127.0.0.1:1234"
    ):
        self.model = model
        self.base_url = base_url
        # Test connection to LM Studio
        try:
            self.embed_query("test")
            print("Successfully connected to LM Studio embeddings API")
        except Exception as e:
            print(f"Warning: Could not connect to LM Studio: {e}")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents using LM Studio API."""
        embeddings = []
        # Process in batches to avoid overwhelming the API
        batch_size = 10
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            batch_embeddings = self._get_embeddings(batch)
            embeddings.extend(batch_embeddings)
            print(
                f"Processed {min(i+batch_size, len(texts))}/{len(texts)} documents")
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        """Embed a query using LM Studio API."""
        return self._get_embeddings([text])[0]

    def _get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings from LM Studio API."""
        url = f"{self.base_url}/v1/embeddings"

        headers = {
            "Content-Type": "application/json"
        }

        # Handle both single string and list of strings
        if isinstance(texts, str):
            texts = [texts]

        data = {
            "model": self.model,
            "input": texts
        }

        try:
            response = requests.post(
                url, headers=headers, data=json.dumps(data))
            response.raise_for_status()
            result = response.json()
            # Extract the embedding data from the response
            embeddings = [item["embedding"] for item in result["data"]]
            return embeddings
        except requests.exceptions.RequestException as e:
            raise ValueError(f"Error getting embeddings from LM Studio: {e}")
        except (KeyError, IndexError) as e:
            raise ValueError(
                f"Error processing LM Studio response: {e}, Response: {response.text}")


def create_vector_store_from_csv(csv_path="urology_patient_data.csv", persist_directory="./chroma_db"):
    """
    Load a CSV file into a ChromaDB vector store using LangChain with LM Studio embeddings.
    
    Args:
        csv_path: Path to the CSV file to load
        persist_directory: Directory to persist the ChromaDB vector store
        
    Returns:
        The ChromaDB vector store object
    """
    print(f"Loading CSV file from {csv_path}...")

    # Load the CSV file into a pandas DataFrame
    df = pd.read_csv(csv_path)

    # Display the first few rows for verification
    print(f"Loaded {len(df)} rows of data. First few rows:")
    print(df.head())

    # Define which column to use as the document content
    # For medical data, we'll use Clinical_Notes as the primary content
    content_column = 'Clinical_Notes'
    
    # If the content column doesn't exist, fall back to the first column
    if content_column not in df.columns:
        content_column = df.columns[0]
        print(f"Warning: Clinical_Notes column not found, using {content_column} instead")

    # Optionally, create metadata from other columns
    def create_metadata(row):
        return {col: row[col] for col in df.columns if col != content_column}

    # Create a list of Document objects from the DataFrame
    documents = []
    for i, row in df.iterrows():
        content = str(row[content_column])
        metadata = create_metadata(row)
        doc = Document(page_content=content, metadata=metadata)
        documents.append(doc)

    print(f"Created {len(documents)} documents")

    # Initialize the LM Studio embedding model
    embeddings = LMStudioEmbeddings()

    # Create the vector store - persistence is automatic in Chroma 0.4.x+
    vector_store = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=persist_directory
    )

    # Note: No need to call vector_store.persist() as Chroma now handles this automatically
    
    print(f"Vector store created and persisted to {persist_directory}")
    return vector_store


def search_vector_store(query, vector_store, n_results=3):
    """
    Search the vector store for documents similar to the query.
    
    Args:
        query: The query string
        vector_store: The ChromaDB vector store
        n_results: Number of results to return
        
    Returns:
        List of (document, score) tuples
    """
    docs = vector_store.similarity_search_with_score(query, k=n_results)
    return docs


def chat_with_lm_studio(messages, model="lmstudio-community/gemma-3-27b-it-GGUF/gemma-3-27b-it-Q4_K_M.gguf", max_tokens=200, temperature=0.7):
    """Sends a list of messages to LM Studio's API and returns the response in Model Context Protocol format."""

    url = "http://localhost:1234/v1/chat/completions"

    headers = {
        "Content-Type": "application/json"
    }

    data = {
        "messages": messages,
        "model": model,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    try:
        response = requests.post(url, headers=headers, data=json.dumps(data))
        response.raise_for_status()
        result = response.json()
        return result["choices"][0]["message"]
    except requests.exceptions.RequestException as e:
        return {"role": "assistant", "content": f"Error: {e}"}
    except (KeyError, IndexError) as e:
        return {"role": "assistant", "content": f"Error processing response: {e}, Response: {response.text}"}


def rag_with_lm_studio(query, vector_store, model="lmstudio-community/gemma-3-27b-it-GGUF/gemma-3-27b-it-Q4_K_M.gguf", max_tokens=500, temperature=0.7, n_results=3):
    """
    Performs RAG using LM Studio and a vector store.
    
    Args:
        query: User query
        vector_store: ChromaDB vector store
        model: LM Studio model to use
        max_tokens: Maximum tokens to generate
        temperature: Generation temperature
        n_results: Number of results to retrieve from vector store
        
    Returns:
        LM Studio response with RAG
    """
    # Search the vector store for relevant documents
    results = search_vector_store(query, vector_store, n_results=n_results)
    
    # Prepare context from retrieved documents
    context = ""
    for i, (doc, score) in enumerate(results):
        context += f"Document {i+1} (Relevance: {score:.4f}):\n"
        context += f"Content: {doc.page_content}\n"
        context += f"Metadata: {json.dumps(doc.metadata, default=str)}\n\n"
    
    # Prepare the messages for the LLM
    messages = [
        {"role": "system", "content": "You are a helpful assistant with access to medical information. When answering, use only the information provided in the context. If the answer cannot be found in the context, say so."},
        {"role": "user", "content": f"Context:\n{context}\n\nQuery: {query}"}
    ]
    
    # Get response from LM Studio
    response = chat_with_lm_studio(messages, model=model, max_tokens=max_tokens, temperature=temperature)
    
    return response


if __name__ == "__main__":
    # Create the vector store
    csv_path = "urology_patient_data.csv"  # Update with your CSV file path
    persist_directory = "./chroma_db"
    
    # Check if vector store exists but may be corrupted
    if os.path.exists(persist_directory) and os.path.isdir(persist_directory):
        # Backup and remove potentially corrupted database
        backup_dir = f"{persist_directory}_backup_{int(time.time())}"
        print(f"Backing up existing database to {backup_dir}")
        shutil.copytree(persist_directory, backup_dir)
        
        # Remove existing database
        print(f"Removing potentially corrupted database at {persist_directory}")
        shutil.rmtree(persist_directory)
        
        # Create new vector store
        print("Creating fresh vector store...")
        vector_store = create_vector_store_from_csv(csv_path=csv_path, persist_directory=persist_directory)
    else:
        # Create new vector store if none exists
        print("No existing database found. Creating new vector store...")
        vector_store = create_vector_store_from_csv(csv_path=csv_path, persist_directory=persist_directory)
    
    # Start interactive RAG session
    print("\nRAG Query System with LM Studio")
    print("Type 'exit' to quit")
    
    while True:
        query = input("\nEnter your query: ")
        if query.lower() in ["exit", "quit", "q"]:
            break
            
        response = rag_with_lm_studio(query, vector_store)
        print("\nResponse:")
        print(response["content"])
