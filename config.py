import os
from dotenv import load_dotenv

load_dotenv()

# Slack API credentials
SLACK_BOT_TOKEN = os.getenv("SLACK_BOT_TOKEN")

# Ollama settings
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")

# Vector DB settings
CHROMA_PERSIST_DIRECTORY = os.getenv("CHROMA_PERSIST_DIRECTORY", "./chroma_db")

# Embedding model
# Available models:
# - all-MiniLM-L6-v2
# - all-mpnet-base-v2
# - multi-qa-mpnet-base-dot-v1
# - e5-large-v2
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-mpnet-base-v2")  

# API key
API_KEY = os.getenv("API_KEY")