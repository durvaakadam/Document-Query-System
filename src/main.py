import sys
import os
from fastapi import FastAPI, Depends, HTTPException, status, Security
from fastapi.openapi.utils import get_openapi
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from typing import List

# ✅ Ensure project root is in path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# ✅ Import your service
from src.services.document_processing_service import DocumentProcessingService

# ✅ Initialize FastAPI app
app = FastAPI()

# ✅ Setup FastAPI's native Bearer token security

bearer_scheme = HTTPBearer()

def verify_bearer_token(
    credentials: HTTPAuthorizationCredentials = Security(bearer_scheme)
):
    VALID_TOKEN = "f5f88484be980678f9c1a168de07125a410ae003e99158d783e36a64810434c4"

    token = credentials.credentials

    # Fix: Remove any accidental repeated "Bearer " prefix
    if token.lower().startswith("bearer "):
        token = token[7:]  # Remove the inner prefix

    if token != VALID_TOKEN:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid Bearer token"
        )
# ✅ Request body model
class QueryRequest(BaseModel):
    documents: List[str]
    questions: List[str]

# ✅ Protected API endpoint
@app.post("/api/v1/hackrx/run")
async def run_submission(
    request: QueryRequest,
    _: HTTPAuthorizationCredentials = Depends(verify_bearer_token)
):
    service = DocumentProcessingService(cache_dir="src/services/document_cache")

    result = await service.process_documents_and_questions(
        document_identifiers=request.documents,
        questions=request.questions
    )

    return result

# ✅ Customize OpenAPI to show Bearer Auth in Swagger UI
def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema

    openapi_schema = get_openapi(
        title="Document Query System API",
        version="1.0.0",
        description="LLM-powered intelligent document QA system",
        routes=app.routes,
    )

    openapi_schema["components"]["securitySchemes"] = {
        "BearerAuth": {
            "type": "http",
            "scheme": "bearer",
            "bearerFormat": "JWT"
        }
    }

    for path in openapi_schema["paths"].values():
        for method in path.values():
            method["security"] = [{"BearerAuth": []}]

    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi
