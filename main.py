import argparse
import logging
from ingest.slack import SlackIngest
from rag.vector_store_minilm import MiniLmVectorStore
from llm.ollama import OllamaLLM

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def ingest_data(channels=None):
    """Ingest data from Slack channels"""
    extractor = SlackIngest()
    vector_store = MiniLmVectorStore()
    
    # Get channels if not specified
    if not channels:
        all_channels = extractor.get_channels()
        channels = [c["id"] for c in all_channels if not c["is_archived"]]
    
    for channel_id in channels:
        logger.info(f"Processing channel: {channel_id}")
        messages = extractor.get_messages(channel_id)
        logger.info(f"Found {len(messages)} bug reports in channel")
        vector_store.add_messages(messages, channel_id)
    
    logger.info("Data ingestion complete")

def search_similar_bugs(query):
    """Search for similar bug reports"""
    vector_store = MiniLmVectorStore()
    ollama = OllamaLLM()
    
    logger.info(f"Searching for similar bug reports to: {query}")
    
    # Get similar bug reports
    results = vector_store.search_similar(query, 3)
    
    # Generate response with Ollama
    response = ollama.generate_response(query, results)
    
    return response

def main():
    parser = argparse.ArgumentParser(description="Slack Bug Reports RAG System")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Ingest data command
    ingest_parser = subparsers.add_parser("ingest", help="Ingest data from Slack")
    ingest_parser.add_argument("--channels", nargs="+", help="Channel IDs to process")
    
    # Search command
    search_parser = subparsers.add_parser("search", help="Search for similar bug reports")
    search_parser.add_argument("query", help="Bug report query to search for")
    
    args = parser.parse_args()
    
    if args.command == "ingest":
        ingest_data(args.channels)
    elif args.command == "search":
        response = search_similar_bugs(args.query)
        print("\nGenerated Response:\n")
        print(response)
    else:
        parser.print_help()

if __name__ == "__main__":
    main() 