import logging
from typing import Dict, Any
from .config import EmbeddingConfig, PineconeConfig
from .vectorizer import DocumentVectorizer
from services.document_processing.ingestion_service import DocumentIngestionService
from services.document_processing.document_processing_service import DocumentTextProcessor

logger = logging.getLogger(__name__)

async def complete_pipeline_with_free_vectorization(document_url: str, embedding_config: EmbeddingConfig, pinecone_config: PineconeConfig) -> Dict[str, Any]:
    """Complete pipeline using free models: Ingest → Extract → Vectorize"""
    
    results = {
        "document_url": document_url,
        "ingestion_success": False,
        "extraction_success": False,
        "vectorization_success": False,
        "document_id": None,
        "total_chunks": 0,
        "errors": []
    }
    
    try:
        # Step 1: Document Ingestion
        logger.info("Step 1: Document Ingestion")
        async with DocumentIngestionService() as ingestion_service:
            doc_info = await ingestion_service.ingest_document(document_url)
            
        if not doc_info.is_valid:
            results["errors"].extend(doc_info.validation_errors)
            return results
        
        results["ingestion_success"] = True
        
        # Step 2: Text Extraction
        logger.info("Step 2: Text Extraction")
        processor = DocumentTextProcessor()
        
        # Map MIME type to file type
        mime_to_type = {
            'application/pdf': 'pdf',
            'application/vnd.openxmlformats-officedocument.wordprocessingml.document': 'docx',
            'message/rfc822': 'email'
        }
        
        file_type = mime_to_type.get(doc_info.mime_type, 'pdf')
        extracted_doc = await processor.process_document(doc_info.file_path, file_type)
        
        results["extraction_success"] = True
        results["document_id"] = extracted_doc.document_id
        results["total_chunks"] = len(extracted_doc.chunks)
        
        # Step 3: Vectorization with free models
        logger.info("Step 3: Vectorization using free Hugging Face models")
        vectorizer = DocumentVectorizer(embedding_config, pinecone_config)
        
        # Initialize vectorizer
        await vectorizer.initialize()
        
        # Vectorize document
        vectorization_success = await vectorizer.vectorize_document(extracted_doc)
        results["vectorization_success"] = vectorization_success
        
        # Get final statistics
        results["vectorization_stats"] = vectorizer.get_vectorization_stats()
        
        logger.info(f"Complete free pipeline finished for document: {extracted_doc.document_id}")
        
    except Exception as e:
        error_msg = f"Pipeline failed: {str(e)}"
        logger.error(error_msg)
        results["errors"].append(error_msg)
    
    return results
