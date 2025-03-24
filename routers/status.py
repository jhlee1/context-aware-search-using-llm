from fastapi import APIRouter, HTTPException, Depends
import logging
from auth.api_key import verify_api_key

# Configure logging
logger = logging.getLogger(__name__)

# Create router with prefix
StatusRouter = APIRouter(
    prefix="/status",
    tags=["status"],
    dependencies=[Depends(verify_api_key)]
)

# Dependencies will be passed from the main app
vector_store = None
ollama = None

def init(vs, llm):
    """Initialize the router with dependencies"""
    global vector_store, ollama
    vector_store = vs
    ollama = llm

@StatusRouter.get("")
async def get_status():
    """Get system status"""
    try:
        # Get collection info from vector store
        collection_info = vector_store.collection.count()
        
        return {
            "status": "ok",
            "vector_store": {
                "document_count": collection_info,
                "collection_name": vector_store.collection.name
            },
            "embedding_model": vector_store.model.get_sentence_embedding_dimension(),
            "llm_model": ollama.model,
            "llm_endpoint": ollama.base_url
        }
    
    except Exception as e:
        logger.error(f"Error getting status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting status: {str(e)}") 