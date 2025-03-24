from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, status
from fastapi.security import APIKeyHeader
import logging
import uvicorn
import time
from fastapi.middleware.cors import CORSMiddleware
from dto.request.search_request import SearchQuery
from dto.response.search_response import SearchResponse
from dto.request.ingest_request import IngestRequest
from dto.response.ingest_response import IngestResponse
from config import API_KEY
from ingest.slack import SlackIngest
from rag.vector_store_minilm import MiniLmVectorStore
from llm.ollama import OllamaLLM
from routers import search, ingest, status
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Slack Bug Reports RAG API",
    description="API for searching and analyzing similar bug reports from Slack using RAG",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict to specific domains in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances
vector_store = MiniLmVectorStore()
ollama = OllamaLLM()
extractor = SlackIngest()


# Initialize routers with dependencies
search.init(vector_store, ollama)
ingest.init(extractor, vector_store)
status.init(vector_store, ollama)

# Include routers
app.include_router(search.SearchRouter)
app.include_router(ingest.IngestRouter)
app.include_router(status.StatusRouter)

# Endpoints
@app.get("/")
async def root():
    return {"message": "Slack Bug Reports RAG API is running"}





if __name__ == "__main__":
    uvicorn.run("api_server:app", host="0.0.0.0", port=8000, reload=True) 