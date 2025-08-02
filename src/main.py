from fastapi import FastAPI, Depends, Header, HTTPException, status
from pydantic import BaseModel
from typing import List  # ✅ Required

app = FastAPI()

# Token validation
def verify_bearer_token(authorization: str = Header(...)):
    VALID_TOKEN = "f5f88484be980678f9c1a168de07125a410ae003e99158d783e36a64810434c4"

    if not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or malformed Authorization header"
        )

    token = authorization.split("Bearer ")[1]

    if token != VALID_TOKEN:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid Bearer token"
        )

# ✅ Corrected input schema
class QueryRequest(BaseModel):
    documents: List[str]
    questions: List[str]

# Endpoint
@app.post("/api/v1/hackrx/run")
async def run_submission(
    request: QueryRequest,
    _: str = Depends(verify_bearer_token)
):
    return {
        "message": "Authorized and processing your query",
        "documents": request.documents,
        "questions": request.questions
    }
