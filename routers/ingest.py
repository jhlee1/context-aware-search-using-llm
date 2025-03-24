from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends
import logging
from dto.request.ingest_request import IngestRequest
from dto.response.ingest_response import IngestResponse
from auth.api_key import verify_api_key

# Configure logging
logger = logging.getLogger(__name__)

# Create router with prefix
IngestRouter = APIRouter(
    prefix="/ingest",
    tags=["ingest"],
    dependencies=[Depends(verify_api_key)]
)

# Dependencies will be passed from the main app
extractor = None
vector_store = None

def init(slack_ingest, vs):
    """Initialize the router with dependencies"""
    global extractor, vector_store
    extractor = slack_ingest
    vector_store = vs

# Background task for ingestion
def ingest_data_task(channels=None, limit=1000):
    total_messages = 0
    
    # Get channels if not specified
    if not channels:
        all_channels = extractor.get_channels()
        channels = [c["id"] for c in all_channels if not c["is_archived"]]
    
    for channel_id in channels:
        logger.info(f"Processing channel: {channel_id}")
        messages = extractor.get_messages(channel_id, limit=limit)
        logger.info(f"Found {len(messages)} bug reports in channel")
        vector_store.add_messages(messages, channel_id)
        total_messages += len(messages)
    
    logger.info(f"Data ingestion complete. Processed {len(channels)} channels, {total_messages} messages.")
    return len(channels), total_messages

@IngestRouter.post("", response_model=IngestResponse)
async def ingest_data(ingest_request: IngestRequest, background_tasks: BackgroundTasks):
    """Ingest data from Slack channels (runs in background)"""
    try:
        # Start ingestion as a background task
        background_tasks.add_task(
            ingest_data_task, 
            ingest_request.channels, 
            ingest_request.limit
        )
        
        return IngestResponse(
            status="started",
            message="Data ingestion started in background",
            channels_processed=len(ingest_request.channels) if ingest_request.channels else 0,
            total_messages_ingested=0  # This is unknown until the task completes
        )
    
    except Exception as e:
        logger.error(f"Error starting ingestion: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error starting ingestion: {str(e)}")

@IngestRouter.post("/sync", response_model=IngestResponse)
async def ingest_data_sync(ingest_request: IngestRequest):
    """Ingest data from Slack channels (synchronous, waits for completion)"""
    try:
        channels_count, messages_count = ingest_data_task(
            ingest_request.channels, 
            ingest_request.limit
        )
        
        return IngestResponse(
            status="completed",
            message="Data ingestion completed",
            channels_processed=channels_count,
            total_messages_ingested=messages_count
        )
    
    except Exception as e:
        logger.error(f"Error in ingestion: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error in ingestion: {str(e)}") 