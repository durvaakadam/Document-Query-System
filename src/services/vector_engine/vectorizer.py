import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
import asyncio

from .embedding_generator import HuggingFaceEmbeddingGenerator
from .pinecone_manager import PineconeManager
from .config import EmbeddingConfig, PineconeConfig
from .types import VectorMetadata
from services.document_processing.document_processing_service import ExtractedDocument

logger = logging.getLogger(__name__)


class DocumentVectorizer:
    """Main service to vectorize documents using free Hugging Face models"""
    
    def __init__(self, 
                 embedding_config: EmbeddingConfig,
                 pinecone_config: PineconeConfig):
        
        self.embedding_generator = HuggingFaceEmbeddingGenerator(embedding_config)
        self.pinecone_manager = PineconeManager(pinecone_config)
        self.embedding_config = embedding_config
        
    async def initialize(self, delete_existing_index: bool = False) -> bool:
        """Initialize the vectorization service"""
        logger.info("Initializing Document Vectorizer with free Hugging Face models...")
        
        success = await self.pinecone_manager.initialize_index(delete_if_exists=delete_existing_index)
        if success:
            logger.info("Document Vectorizer initialized successfully")
        else:
            logger.error("Failed to initialize Document Vectorizer")
        
        return success
    
    async def vectorize_document(self, extracted_doc: ExtractedDocument) -> bool:
        """Vectorize an extracted document and store in Pinecone"""
        logger.info(f"Vectorizing document: {extracted_doc.document_id}")
        
        if not extracted_doc.chunks:
            logger.warning(f"No chunks found in document: {extracted_doc.document_id}")
            return False
        
        try:
            # Prepare texts for embedding
            texts = []
            metadatas = []
            
            for chunk in extracted_doc.chunks:
                # Create enhanced text for embedding (include section title if available)
                embedding_text = chunk.content
                if chunk.section_title and chunk.section_title not in chunk.content:
                    embedding_text = f"{chunk.section_title}\n\n{chunk.content}"
                
                texts.append(embedding_text)
                
                # Create metadata
                metadata = VectorMetadata(
                    document_id=extracted_doc.document_id,
                    chunk_id=chunk.chunk_id,
                    chunk_type=chunk.chunk_type,
                    content=chunk.content[:1000],  # Truncate content for metadata storage
                    page_number=chunk.page_number,
                    section_title=chunk.section_title,
                    word_count=chunk.word_count,
                    file_type=extracted_doc.file_type,
                    source_file=extracted_doc.source_file,
                    created_at=datetime.now().isoformat()
                )
                metadatas.append(metadata)
            
            # Generate embeddings using free Hugging Face model
            logger.info(f"Generating embeddings for {len(texts)} chunks using free model...")
            embeddings = await self.embedding_generator.generate_embeddings_batch(texts)
            
            # Store in Pinecone
            logger.info(f"Storing embeddings in Pinecone...")
            success = await self.pinecone_manager.upsert_vectors(embeddings, metadatas)
            
            if success:
                logger.info(f"Successfully vectorized document: {extracted_doc.document_id}")
                
                # Log usage statistics
                usage_stats = self.embedding_generator.get_usage_stats()
                logger.info(f"Embedding usage: {usage_stats}")
                
                return True
            else:
                logger.error(f"Failed to store vectors for document: {extracted_doc.document_id}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to vectorize document {extracted_doc.document_id}: {e}")
            return False
    
    async def vectorize_multiple_documents(self, 
                                         extracted_docs: List[ExtractedDocument],
                                         max_concurrent: int = 1) -> List[bool]:
        """Vectorize multiple documents (reduced concurrency for free models)"""
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def bounded_vectorize(doc):
            async with semaphore:
                return await self.vectorize_document(doc)
        
        logger.info(f"Starting batch vectorization of {len(extracted_docs)} documents")
        
        tasks = [bounded_vectorize(doc) for doc in extracted_docs]
        results = await asyncio.gather(*tasks, return_exceptions=False)
        
        successful = sum(1 for r in results if r)
        logger.info(f"Batch vectorization completed: {successful}/{len(extracted_docs)} successful")
        
        return results
    
    async def search_similar_chunks(self, 
                                  query_text: str,
                                  top_k: int = 10,
                                  document_filter: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Search for similar text chunks using semantic similarity"""
        logger.info(f"Searching for similar chunks: '{query_text[:100]}...'")
        
        try:
            # Generate embedding for query using free model
            query_embedding = await self.embedding_generator.generate_embedding(query_text)
            
            # Search in Pinecone
            results = await self.pinecone_manager.query_vectors(
                query_vector=query_embedding,
                top_k=top_k,
                filter_dict=document_filter,
                include_metadata=True
            )
            
            # Enhance results with readable format
            enhanced_results = []
            for result in results:
                enhanced_result = {
                    "chunk_id": result["id"],
                    "similarity_score": result["score"],
                    "content": result["metadata"].get("content", ""),
                    "section_title": result["metadata"].get("section_title"),
                    "page_number": result["metadata"].get("page_number"),
                    "document_id": result["metadata"].get("document_id"),
                    "chunk_type": result["metadata"].get("chunk_type"),
                    "metadata": result["metadata"]
                }
                enhanced_results.append(enhanced_result)
            
            logger.info(f"Found {len(enhanced_results)} similar chunks")
            return enhanced_results
            
        except Exception as e:
            logger.error(f"Failed to search similar chunks: {e}")
            return []
    
    async def delete_document_vectors(self, document_id: str) -> bool:
        """Delete all vectors for a specific document"""
        logger.info(f"Deleting vectors for document: {document_id}")
        
        try:
            filter_dict = {"document_id": {"$eq": document_id}}
            success = await self.pinecone_manager.delete_by_filter(filter_dict)
            
            if success:
                logger.info(f"Successfully deleted vectors for document: {document_id}")
            else:
                logger.error(f"Failed to delete vectors for document: {document_id}")
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to delete document vectors: {e}")
            return False
    
    def get_vectorization_stats(self) -> Dict[str, Any]:
        """Get comprehensive vectorization statistics"""
        embedding_stats = self.embedding_generator.get_usage_stats()
        index_stats = self.pinecone_manager.get_index_stats()
        
        return {
            "embedding_usage": embedding_stats,
            "index_stats": index_stats,
            "configuration": {
                "embedding_model": self.embedding_config.model_name,
                "embedding_dimensions": self.embedding_config.dimensions,
                "max_sequence_length": self.embedding_config.max_length,
                "device": self.embedding_generator.device
            }
        }
