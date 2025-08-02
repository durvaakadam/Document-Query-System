from pydantic import BaseModel
from typing import List

class DocumentQueryRequest(BaseModel):
    documents: List[str]
    questions: List[str]

class DocumentQueryResponse(BaseModel):
    answers: List[str]
