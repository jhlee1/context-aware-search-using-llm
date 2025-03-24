from fastapi import APIRouter, HTTPException, Depends, status
import time
import logging
from dto.request.search_request import SearchQuery
from dto.response.search_response import SearchResponse
from auth.api_key import verify_api_key

# Configure logging
logger = logging.getLogger(__name__)

# Create router with prefix
SearchRouter = APIRouter(
    prefix="/search",
    tags=["search"],
    dependencies=[Depends(verify_api_key)]
)

# The vector store and LLM will be passed from the main app
vector_store = None
ollama = None

def init(vs, llm):
    """Initialize the router with dependencies"""
    global vector_store, ollama
    vector_store = vs
    ollama = llm

@SearchRouter.post("", response_model=SearchResponse)
async def search_similar_bugs(search_request: SearchQuery):
    """Search for similar bug reports based on a query"""
    start_time = time.time()
    
    try:
        # Get similar bug reports
        results = vector_store.search_similar(search_request.query, n_results=search_request.max_results)
        
        # Generate response with Ollama
        analysis = ollama.generate_response(search_request.query, results)
        
        # Format response
        similar_reports = []
        for i, doc in enumerate(results.get("documents", [[]])[0]):
            metadata = results.get("metadatas", [[]])[0][i] if i < len(results.get("metadatas", [[]])[0]) else {}
            similar_reports.append({
                "document": doc,
                "metadata": metadata,
                "score": results.get("distances", [[]])[0][i] if i < len(results.get("distances", [[]])[0]) else None
            })
        
        processing_time = time.time() - start_time
        
        return SearchResponse(
            query=search_request.query,
            similar_reports=similar_reports,
            analysis=analysis,
            processing_time=processing_time
        )
    
    except Exception as e:
        logger.error(f"Error in search: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing search: {str(e)}") 
