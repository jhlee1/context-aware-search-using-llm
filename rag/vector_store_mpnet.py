from typing import List, Dict, Any

from vector_store import VectorStore


class MPNetVectorStore(VectorStore):
    def __init__(self):
        super().__init__()
        
    def _prepare_document(self, message: Dict[str, Any]) -> str:
        """Format message into a document optimized for semantic search"""
        # Create a clear, structured document that captures the context
        parts = []
        
        # Add primary issue description
        parts.append(f"ISSUE: {message['text']}")
        
        # Add structured context from thread
        if "thread_context" in message and message["thread_context"]:
            parts.append("CONTEXT:")
            for i, reply in enumerate(message["thread_context"]):
                # Only include substantive messages (not just acknowledgments)
                if len(reply["text"]) > 10:  # Skip very short replies
                    parts.append(f"- {reply['text']}")
        
        # Extract key information using simple NLP rules
        main_text = message['text'].lower()
        
        # Look for reproduction steps
        if any(marker in main_text for marker in ["steps to reproduce", "to reproduce", "reproducing", "how to"]):
            parts.append("REPRODUCTION: " + message['text'])
        
        # Look for error information
        if any(marker in main_text for marker in ["error", "exception", "failed", "crash"]):
            parts.append("ERROR: " + message['text'])
        
        # Combine all parts with clear separation
        return "\n\n".join(parts)
        
    def search_similar(self, query: str, n_results: int = 5):
        """Enhanced semantic search with hybrid retrieval"""
        # Generate embedding for the query
        query_embedding = self.model.encode([query]).tolist()
        
        # Get results based on vector similarity
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=n_results * 2  # Get more results for reranking
        )
        
        # Extract the documents and metadata
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]
        
        # Simple reranking based on keyword presence
        ranked_results = []
        keywords = self._extract_keywords(query)
        
        for i, (doc, meta, dist) in enumerate(zip(documents, metadatas, distances)):
            # Calculate keyword score (simple approach)
            keyword_score = sum(1 for kw in keywords if kw.lower() in doc.lower())
            
            # Combined score (semantic + keyword)
            # Convert distance to similarity (1 - distance) since we're using cosine distance
            semantic_score = 1 - dist
            combined_score = (semantic_score * 0.7) + (keyword_score * 0.3 / max(1, len(keywords)))
            
            ranked_results.append({
                "document": doc,
                "metadata": meta,
                "semantic_score": semantic_score,
                "keyword_score": keyword_score,
                "combined_score": combined_score
            })
        
        # Sort by combined score and take top n
        ranked_results.sort(key=lambda x: x["combined_score"], reverse=True)
        top_results = ranked_results[:n_results]
        
        # Reformat back to chromadb result format
        final_results = {
            "documents": [[r["document"] for r in top_results]],
            "metadatas": [[r["metadata"] for r in top_results]],
            "distances": [[1 - r["combined_score"] for r in top_results]]
        }
        
        return final_results
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract potential keywords from query text"""
        # Simple approach: split by spaces and filter out common words
        stopwords = {"the", "a", "an", "is", "are", "was", "were", "and", "or", "but", "in", "on", "at", "to", "for", "with"}
        words = text.split()
        keywords = [word.strip(".,?!") for word in words if word.lower() not in stopwords and len(word) > 2]
        return keywords