import os
from typing import List, Dict, Any
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import json
import hashlib

from vector_store import VectorStore

from config import CHROMA_PERSIST_DIRECTORY, EMBEDDING_MODEL

class MiniLmVectorStore(VectorStore):
    def __init__(self):
        super().__init__()
    
    def _prepare_document(self, message: Dict[str, Any]) -> str:
        """Format message into a document for embedding"""
        doc = f"Bug Report\n"
        doc += f"Date: {message['date']}\n"
        doc += f"Reported by: {message['user']['real_name']}\n"
        doc += f"Description: {message['text']}\n"
        
        # Add replies if they exist
        if "replies" in message and message["replies"]:
            doc += "\nFollow-up Comments:\n"
            for reply in message["replies"]:
                doc += f"- {reply['user']['real_name']}: {reply['text']}\n"
        
        return doc
    
    def search_similar(self, query: str, n_results: int):
        """Search for similar bug reports"""
        query_embedding = self.model.encode([query]).tolist()
        
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=n_results
        )
        
        return results 