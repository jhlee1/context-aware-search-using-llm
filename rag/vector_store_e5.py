from typing import List, Dict, Any

from vector_store import VectorStore
from sentence_transformers import CrossEncoder
import numpy as np

class E5VectorStore(VectorStore):
    def __init__(self):
        super().__init__()
        # Add a cross-encoder for reranking
        # available options:
        # cross-encoder/ms-marco-MiniLM-L-6-v2
        # cross-encoder/stsb-roberta-base
        # cross-encoder/ms-marco-electra-base
        # cross-encoder/ms-marco-deberta-v3-large
        self.cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
       
        
    def _prepare_document(self, message: Dict[str, Any]) -> List[Dict[str, Any]]:
       """Create multiple sophisticated chunks from each message"""
       chunks = []
       
       # Main issue description
       main_chunk = {
           "text": message['text'],
           "type": "main_issue",
           "metadata": {
               "timestamp": message["ts"],
               "user": message["user"]["real_name"]
           }
       }
       chunks.append(main_chunk)
       
       # Thread context as separate but linked chunks
       if "thread_context" in message and len(message["thread_context"]) > 0:
           # Group responses by topic using simple heuristics
           current_topic = {"texts": [], "users": []}
           
           for reply in message["thread_context"]:
               # Start a new topic if different user or significant time gap
               if (reply["user"]["real_name"] not in current_topic["users"] and 
                   len(current_topic["texts"]) > 0):
                   # Add current topic as a chunk
                   topic_text = "\n".join(current_topic["texts"])
                   chunks.append({
                       "text": topic_text,
                       "type": "thread_response",
                       "metadata": {
                           "parent_ts": message["ts"],
                           "users": current_topic["users"]
                       }
                   })
                   # Reset current topic
                   current_topic = {"texts": [reply["text"]], "users": [reply["user"]["real_name"]]}
               else:
                   # Add to current topic
                   current_topic["texts"].append(reply["text"])
                   if reply["user"]["real_name"] not in current_topic["users"]:
                       current_topic["users"].append(reply["user"]["real_name"])
           
           # Add the last topic
           if len(current_topic["texts"]) > 0:
               topic_text = "\n".join(current_topic["texts"])
               chunks.append({
                   "text": topic_text,
                   "type": "thread_response",
                   "metadata": {
                       "parent_ts": message["ts"],
                       "users": current_topic["users"]
                   }
               })
       
       return chunks
        
    def search_similar(self, query: str, n_results: int = 5, rerank_candidates: int = 25):
        """Two-stage retrieval with cross-encoder reranking"""
        # Stage 1: Semantic search with bi-encoder (your embedding model)
        # This retrieves initial candidates efficiently
        query_embedding = self.model.encode([query]).tolist()
        candidates = self.collection.query(
            query_embeddings=query_embedding,
            n_results=rerank_candidates  # Get more candidates for reranking
        )
        
        # Extract candidate documents
        candidate_docs = candidates.get("documents", [[]])[0]
        candidate_metadata = candidates.get("metadatas", [[]])[0]
        
        if not candidate_docs:
            return {"documents": [[]], "metadatas": [[]], "distances": [[]]}
        
        # Stage 2: Precise reranking with cross-encoder
        # This is computationally intensive but provides better ranking
        query_doc_pairs = [[query, doc] for doc in candidate_docs]
        
        # Get cross-encoder scores
        cross_scores = self.cross_encoder.predict(query_doc_pairs)
        
        # Combine candidates with their scores
        scored_results = [
            {"document": doc, "metadata": meta, "score": score}
            for doc, meta, score in zip(candidate_docs, candidate_metadata, cross_scores)
        ]
        
        # Sort by cross-encoder score (higher is better)
        scored_results.sort(key=lambda x: x["score"], reverse=True)
        
        # Take top n results
        top_results = scored_results[:n_results]
        
        # Format results to match expected return format
        return {
            "documents": [[r["document"] for r in top_results]],
            "metadatas": [[r["metadata"] for r in top_results]],
            "distances": [[1 - min(1, max(0, r["score"]/5)) for r in top_results]]  # Normalize scores to distances
        }
        
    def search_with_explanations(self, query: str, n_results: int = 5):
        """Search with explanations of why matches are relevant"""
        # Get reranked results
        results = self.search_similar(query, n_results)
        
        # Add explanations
        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        scores = results["distances"][0]
        
        explained_results = []
        for doc, meta, score in zip(documents, metadatas, scores):
            # Create explanation based on high cross-encoder score
            similarity = 1 - score  # Convert distance back to similarity
            
            # Highlight matching sections (simplified approach)
            highlighted_doc = self._highlight_matching_sections(query, doc)
            
            explained_results.append({
                "document": doc,
                "highlighted_document": highlighted_doc,
                "metadata": meta,
                "relevance_score": similarity,
                "explanation": f"This bug report is {similarity:.2f} similar to your query.",
            })
            
        return explained_results
    
    def _highlight_matching_sections(self, query, document):
        """Simple function to highlight potentially matching sections"""
        # This is a simplified approach - in production you would use
        # more sophisticated NLP techniques
        
        # Extract key terms from query (simple approach)
        query_terms = [term.lower() for term in query.split() if len(term) > 3]
        
        # Split document into lines
        lines = document.split('\n')
        highlighted_lines = []
        
        for line in lines:
            line_lower = line.lower()
            if any(term in line_lower for term in query_terms):
                highlighted_lines.append(f">> {line} <<")
            else:
                highlighted_lines.append(line)
                
        return '\n'.join(highlighted_lines)