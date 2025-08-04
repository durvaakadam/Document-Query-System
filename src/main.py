import sys
import os
from dotenv import load_dotenv
from fastapi import FastAPI, Depends, HTTPException, status, Security
from fastapi.openapi.utils import get_openapi
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from typing import List

# Load environment variables from .env file
load_dotenv()

# Ensure project root is in path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import your services
from src.services.document_processing.document_processing_service import DocumentTextProcessor
from src.services.vector_engine.config import EmbeddingConfig, PineconeConfig
from src.services.vector_engine.vectorizer import DocumentVectorizer # Initialize FastAPI app
app = FastAPI()

# Setup FastAPI's native Bearer token security

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
# Request body model
class QueryRequest(BaseModel):
    documents: List[str]
    questions: List[str]

# Protected API endpoint
@app.post("/api/v1/hackrx/run")
async def run_submission(
    request: QueryRequest,
    _: HTTPAuthorizationCredentials = Depends(verify_bearer_token)
):
    doc_processor = DocumentTextProcessor()
    
    # Convert document names to full paths and detect file types
    document_infos = []
    for doc in request.documents:
        # Build full path (assuming documents are in cache)
        file_path = f"src/services/document_cache/{doc}"
        
        # Detect file type from extension
        if doc.endswith('.pdf'):
            file_type = "pdf"
        elif doc.endswith('.docx'):
            file_type = "docx"
        elif doc.endswith('.eml'):
            file_type = "email"
        else:
            file_type = "pdf"  # default fallback
        
        document_infos.append((file_path, file_type))

    try:
        # Step 1: Process documents to extract text and create chunks
        extracted_documents = await doc_processor.process_multiple_documents(document_infos)
        
        # Step 2: Try to use vector search, but fallback to simple text search if it fails
        try:
            # Setup vector engine configuration
            embedding_config = EmbeddingConfig()
            pinecone_config = PineconeConfig()
            
            # Initialize vectorizer for question answering
            vectorizer = DocumentVectorizer(embedding_config, pinecone_config)
            await vectorizer.initialize()
            
            # Vectorize documents if not already done
            for extracted_doc in extracted_documents:
                await vectorizer.vectorize_document(extracted_doc)
            
            # Answer questions using semantic search
            answers = []
            for question in request.questions:
                # Search for relevant chunks across all processed documents
                similar_chunks = await vectorizer.search_similar_chunks(
                    query_text=question,
                    top_k=5  # Get top 5 most relevant chunks
                )
                
                if similar_chunks:
                    # Combine the most relevant chunks to form an answer
                    context_chunks = []
                    for chunk in similar_chunks[:3]:  # Use top 3 chunks
                        context_chunks.append({
                            "content": chunk["content"],
                            "document_id": chunk["document_id"],
                            "similarity_score": chunk["similarity_score"],
                            "page_number": chunk.get("page_number"),
                            "section_title": chunk.get("section_title")
                        })
                    
                    # Create a comprehensive answer based on found chunks
                    answer = {
                        "question": question,
                        "relevant_chunks": context_chunks,
                        "summary": f"Found {len(similar_chunks)} relevant sections. The most relevant information comes from {context_chunks[0]['document_id']} with {context_chunks[0]['similarity_score']:.3f} similarity score.",
                        "search_method": "vector_search"
                    }
                else:
                    answer = {
                        "question": question,
                        "relevant_chunks": [],
                        "summary": "No relevant information found for this question in the provided documents.",
                        "search_method": "vector_search"
                    }
                
                answers.append(answer)
                
        except Exception as vector_error:
            # Fallback to simple text-based search
            print(f"Vector search failed, falling back to text search: {vector_error}")
            
            answers = []
            for question in request.questions:
                # Simple keyword-based search through all chunks
                relevant_chunks = []
                question_words = question.lower().split()
                
                for extracted_doc in extracted_documents:
                    for i, chunk in enumerate(extracted_doc.chunks):
                        chunk_text = chunk.content.lower()
                        # Calculate simple relevance score based on keyword matches
                        matches = sum(1 for word in question_words if word in chunk_text)
                        if matches > 0:
                            relevance_score = matches / len(question_words)
                            relevant_chunks.append({
                                "content": chunk.content[:500] + "..." if len(chunk.content) > 500 else chunk.content,
                                "document_id": extracted_doc.document_id,
                                "chunk_index": i,
                                "relevance_score": relevance_score,
                                "section_title": getattr(chunk, 'section_title', None),
                                "page_number": getattr(chunk, 'page_number', None)
                            })
                
                # Sort by relevance score and take top 3
                relevant_chunks.sort(key=lambda x: x["relevance_score"], reverse=True)
                top_chunks = relevant_chunks[:3]
                
                if top_chunks:
                    answer = {
                        "question": question,
                        "relevant_chunks": top_chunks,
                        "summary": f"Found {len(relevant_chunks)} relevant sections using text search. Best match has {top_chunks[0]['relevance_score']:.2f} relevance score.",
                        "search_method": "text_search_fallback"
                    }
                else:
                    answer = {
                        "question": question,
                        "relevant_chunks": [],
                        "summary": "No relevant information found for this question in the provided documents.",
                        "search_method": "text_search_fallback"
                    }
                
                answers.append(answer)
        
        return {
            "answers": answers,
            "processed_documents": [
                {
                    "document_id": doc.document_id,
                    "chunks_count": len(doc.chunks),
                    "total_characters": sum(len(chunk.content) for chunk in doc.chunks)
                }
                for doc in extracted_documents
            ]
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing documents or answering questions: {str(e)}"
        )

# Customize OpenAPI to show Bearer Auth in Swagger UI
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)