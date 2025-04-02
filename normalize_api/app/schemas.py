
from typing import List, Dict, Any
from pydantic import BaseModel

class RecordInput(BaseModel):
    record: Dict[str, Any]
    mapping_key: str = "default"

class BatchInput(BaseModel):
    records: List[Dict[str, Any]]
    mapping_key: str = "default"
