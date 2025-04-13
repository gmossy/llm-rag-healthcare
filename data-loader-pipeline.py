#!/usr/bin/env python3
"""
Data Loader Pipeline for ChromaDB with LM Studio Embeddings

This module implements a data ingestion pipeline that:
1. Processes CSV files for vectorization
2. Generates embeddings via LM Studio API
3. Persists vector representations in ChromaDB
4. Implements error handling and data validation

Author: Technical Research Team
Version: 1.0.0
"""

import argparse
import csv
import os
import sys
import time
import logging
import json
import pandas as pd
import requests
import shutil
from typing import List, Dict, Any, Optional, Union
import chromadb
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("data_loader.log")
    ]
)
logger = logging.getLogger("DataLoader")


class LMStudioEmbeddings(Embeddings):
    """LM Studio embeddings wrapper for vectorization operations."""

    def __init__(
        self,
        model: str = "text-embedding-medical-10-10-1-jinaai_jina-embeddings-v2-small-en-50-gpt-3.5-turbo-01_9062874564-i1",
        base_url: str = "http://127.0.0.1:1234"
    ):
        self.model = model
        self.base_url = base_url
        logger.info(f"Initializing LM Studio Embeddings with model: {model}")
        
        # Test connection to LM Studio
        try:
            self.embed_query("test")
            logger.info("Successfully connected to LM Studio embeddings API")
        except Exception as e:
            logger.error(f"Warning: Could not connect to LM Studio: {e}")
            raise ConnectionError(f"Failed to connect to LM Studio API at {base_url}: {e}")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents using LM Studio API."""
        embeddings = []
        # Process in batches to avoid overwhelming the API
        batch_size = 10
        total_docs = len(texts)
        
        logger.info(f"Embedding {total_docs} documents in batches of {batch_size}")
        
        for i in range(0, total_docs, batch_size):
            batch = texts[i:i+batch_size]
            try:
                batch_embeddings = self._get_embeddings(batch)
                embeddings.extend(batch_embeddings)
                logger.info(f"Processed {min(i+batch_size, total_docs)}/{total_docs} documents")
            except Exception as e:
                logger.error(f"Error embedding batch {i//batch_size}: {e}")
                raise
                
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
            logger.error(f"API request error: {e}")
            raise ValueError(f"Error getting embeddings from LM Studio: {e}")
        except (KeyError, IndexError) as e:
            logger.error(f"Response parsing error: {e}")
            raise ValueError(
                f"Error processing LM Studio response: {e}, Response: {response.text}")


class DataLoader:
    """Data loader pipeline for CSV ingestion into ChromaDB."""
    
    def __init__(self, persist_directory: str = "./chroma_db"):
        """
        Initialize the data loader with configuration parameters.
        
        Args:
            persist_directory: Path to ChromaDB persistence directory
        """
        self.persist_directory = persist_directory
        self.embeddings = None
        logger.info(f"Initialized DataLoader with persistence directory: {persist_directory}")
        
    def validate_csv(self, file_path: str) -> bool:
        """
        Validate CSV file structure and content.
        
        Args:
            file_path: Path to CSV file
            
        Returns:
            Boolean indicating validation success
        """
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return False
            
        try:
            df = pd.read_csv(file_path)
            logger.info(f"CSV validation successful: {len(df)} rows, {len(df.columns)} columns")
            
            # Check if file has content
            if len(df) == 0:
                logger.error("CSV file is empty")
                return False
                
            # Log column names for reference
            logger.info(f"CSV columns: {', '.join(df.columns)}")
            return True
            
        except Exception as e:
            logger.error(f"CSV validation failed: {e}")
            return False
    
    def initialize_embeddings(self) -> None:
        """Initialize the embedding model."""
        try:
            self.embeddings = LMStudioEmbeddings()
            logger.info("Embedding model initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize embedding model: {e}")
            raise
    
    def handle_existing_database(self) -> None:
        """Handle existing database by backing up and removing if needed."""
        if os.path.exists(self.persist_directory) and os.path.isdir(self.persist_directory):
            backup_dir = f"{self.persist_directory}_backup_{int(time.time())}"
            logger.info(f"Found existing database. Creating backup at {backup_dir}")
            
            try:
                shutil.copytree(self.persist_directory, backup_dir)
                logger.info("Backup created successfully")
                
                shutil.rmtree(self.persist_directory)
                logger.info("Removed existing database to prevent compatibility issues")
            except Exception as e:
                logger.error(f"Error during database backup/removal: {e}")
                raise
    
    def load_data(self, csv_path: str, content_column: Optional[str] = None) -> bool:
        """
        Load data from CSV into ChromaDB with embeddings.
        
        Args:
            csv_path: Path to CSV file
            content_column: Primary column to use for embeddings (optional)
            
        Returns:
            Boolean indicating success
        """
        logger.info(f"Starting data loading process for {csv_path}")
        
        # Validate CSV
        if not self.validate_csv(csv_path):
            return False
            
        # Initialize embeddings
        if self.embeddings is None:
            self.initialize_embeddings()
            
        # Handle existing database
        self.handle_existing_database()
        
        # Load CSV
        try:
            df = pd.read_csv(csv_path)
            logger.info(f"Loaded {len(df)} rows from {csv_path}")
            
            # Determine content column
            if content_column is None or content_column not in df.columns:
                # For urology data, prefer clinical notes or fall back to first column
                if 'Clinical_Notes' in df.columns:
                    content_column = 'Clinical_Notes'
                    logger.info("Using 'Clinical_Notes' as content column")
                else:
                    content_column = df.columns[0]
                    logger.info(f"Using '{content_column}' as content column")
            
            # Create document objects
            documents = []
            for i, row in df.iterrows():
                content = str(row[content_column])
                
                # Create metadata from other columns
                metadata = {col: str(row[col]) for col in df.columns if col != content_column}
                
                doc = Document(page_content=content, metadata=metadata)
                documents.append(doc)
                
            logger.info(f"Created {len(documents)} document objects")
            
            # Create and persist vector store
            from langchain_chroma import Chroma
            
            logger.info("Creating vector store with ChromaDB")
            vector_store = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                persist_directory=self.persist_directory
            )
            
            logger.info(f"Successfully created vector store at {self.persist_directory}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return False
            
    def interactive_mode(self) -> None:
        """Run interactive mode to load data."""
        print("\n=== ChromaDB Data Loader with LM Studio ===\n")
        
        # Get CSV path
        while True:
            csv_path = input("Enter CSV file path (or 'exit' to quit): ")
            
            if csv_path.lower() in ['exit', 'quit', 'q']:
                print("Exiting program.")
                return
                
            if not os.path.exists(csv_path):
                print(f"Error: File not found at {csv_path}")
                continue
                
            break
            
        # Default to urology data if available
        if os.path.basename(csv_path) != "urology_patient_data.csv" and os.path.exists("urology_patient_data.csv"):
            use_default = input("urology_patient_data.csv found. Use it instead? (y/n): ")
            if use_default.lower().startswith('y'):
                csv_path = "urology_patient_data.csv"
                print(f"Using urology_patient_data.csv")
                
        # Get content column (optional)
        content_column = input("Enter the column name to use for document content (press Enter for auto-detection): ")
        if content_column.strip() == "":
            content_column = None
            print("Using auto-detection for content column")
            
        # Confirm operation
        print("\nOperation Summary:")
        print(f"- CSV File: {csv_path}")
        print(f"- Content Column: {'Auto-detect' if content_column is None else content_column}")
        print(f"- Persistence Directory: {self.persist_directory}")
        
        confirm = input("\nProceed with data loading? (y/n): ")
        if not confirm.lower().startswith('y'):
            print("Operation cancelled.")
            return
            
        # Execute data loading
        print("\nStarting data loading process...\n")
        success = self.load_data(csv_path, content_column)
        
        if success:
            print("\n✅ Data loading completed successfully!")
            print(f"Vector store created at: {os.path.abspath(self.persist_directory)}")
        else:
            print("\n❌ Data loading failed. Check logs for details.")
        

def main():
    """Main entry point for the data loader pipeline."""
    parser = argparse.ArgumentParser(description="Data Loader Pipeline for ChromaDB with LM Studio")
    parser.add_argument("--csv", type=str, help="Path to CSV file")
    parser.add_argument("--content-column", type=str, help="Column to use for document content")
    parser.add_argument("--persist-dir", type=str, default="./chroma_db", 
                        help="Directory to persist ChromaDB")
    parser.add_argument("--interactive", action="store_true", 
                        help="Run in interactive mode")
    
    args = parser.parse_args()
    
    loader = DataLoader(persist_directory=args.persist_dir)
    
    if args.interactive or not args.csv:
        loader.interactive_mode()
    else:
        success = loader.load_data(args.csv, args.content_column)
        if success:
            print("Data loading completed successfully!")
        else:
            print("Data loading failed. Check logs for details.")
            sys.exit(1)


if __name__ == "__main__":
    main()
