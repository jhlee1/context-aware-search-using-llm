import os
from typing import List, Dict, Any
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import json
import hashlib
from abc import ABC, abstractmethod
from config import CHROMA_PERSIST_DIRECTORY, EMBEDDING_MODEL

class VectorStore(ABC):
    def __init__(self):
        # Create directory if it doesn't exist
        os.makedirs(CHROMA_PERSIST_DIRECTORY, exist_ok=True)
        
        # Initialize ChromaDB
        self.client = chromadb.Client(Settings(
            persist_directory=CHROMA_PERSIST_DIRECTORY,
            anonymized_telemetry=False
        ))
        
        # Create or get collection
        self.collection = self.client.get_or_create_collection(
            name="slack_bug_reports",
            metadata={"hnsw:space": "cosine"}
        )
        
        # Initialize embedding model
        self.model = SentenceTransformer(EMBEDDING_MODEL)
    
    def _create_document_id(self, message: Dict[str, Any]) -> str:
        """Create a unique ID for each message"""
        unique_string = f"{message['ts']}-{message.get('channel_id', '')}"
        return hashlib.md5(unique_string.encode()).hexdigest()
    
    @abstractmethod
    def _prepare_document(self, message: Dict[str, Any]) -> str:
        pass
        
    def add_messages(self, messages: List[Dict[str, Any]], channel_id: str = None):
        """Add messages to the vector store"""
        if not messages:
            return
        
        documents = []
        ids = []
        metadatas = []
        
        for message in messages:
            # Add channel_id to message if provided
            if channel_id:
                message["channel_id"] = channel_id
            
            doc_id = self._create_document_id(message)
            document = self._prepare_document(message)
            
            # Store original message as metadata
            metadata = {
                "timestamp": message["ts"],
                "date": message["date"],
                "user": message["user"]["real_name"],
                "original_message": json.dumps(message)
            }
            
            documents.append(document)
            ids.append(doc_id)
            metadatas.append(metadata)
        
        # Generate embeddings and add to collection
        embeddings = self.model.encode(documents).tolist()
        
        self.collection.add(
            documents=documents,
            embeddings=embeddings,
            ids=ids,
            metadatas=metadatas
        )
    
    @abstractmethod
    def search_similar(self, query: str, n_results: int):
        pass