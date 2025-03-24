from pydantic import BaseModel
from typing import Optional, List

class IngestRequest(BaseModel):
    channels: Optional[List[str]] = None
    limit: int = 1000
