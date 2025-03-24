
from pydantic import BaseModel

class IngestResponse(BaseModel):
    status: str
    message: str
    channels_processed: int
    total_messages_ingested: int