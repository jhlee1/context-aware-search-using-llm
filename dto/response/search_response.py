from pydantic import BaseModel
from typing import List, Dict, Any

class SearchResponse(BaseModel):
    query: str
    similar_reports: List[Dict[str, Any]]
    analysis: str
    processing_time: float