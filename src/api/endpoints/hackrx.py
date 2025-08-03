from fastapi import APIRouter, HTTPException, Depends
from schemas.models import DocumentQueryRequest, DocumentQueryResponse
from services.document_processing_service import DocumentProcessingService
from typing import List
from auth.bearer import verify_bearer_token

router = APIRouter()

@router.post("/run", response_model=DocumentQueryResponse)
async def run_document_query(
    request: DocumentQueryRequest,
    _: str = Depends(verify_bearer_token)
):
    """
    Process documents and questions to generate answers.
    """
    try:
        print("=== DEBUGGING START ===")
        print(f"Received documents: {request.documents}")
        print(f"Received questions: {request.questions}")
        
        # Validate input
        if not request.documents:
            raise HTTPException(status_code=400, detail="At least one document must be provided")
        if not request.questions:
            raise HTTPException(status_code=400, detail="At least one question must be provided")
        
        print("Validation passed, initializing DocumentProcessingService...")
        
        processing_service = DocumentProcessingService(cache_dir="./services/document_cache")
        print("Service initialized!")
        
        print("Processing documents and questions...")
        result = await processing_service.process_documents_and_questions(
            document_identifiers=request.documents,
            questions=request.questions
        )
        print(f"Processing result: {result}")
        
        print("=== DEBUGGING END ===")
        return DocumentQueryResponse(answers=result["answers"])
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
