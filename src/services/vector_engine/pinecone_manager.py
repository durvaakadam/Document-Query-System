import asyncio
import logging
import pinecone
from pinecone import Pinecone,ServerlessSpec
from .config import PineconeConfig
from .types import VectorMetadata
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

class PineconeManager:
    """Manage Pinecone vector database operations (same as before but updated for new dimensions)"""
    
    def __init__(self, config: PineconeConfig):
        self.config = config
        self.pc = Pinecone(api_key=config.api_key)

    
    async def initialize_index(self, delete_if_exists: bool = False) -> bool:
        """Initialize Pinecone index"""
        try:
            # Check if index exists
            existing_indexes = self.pc.list_indexes()
            index_exists = any(idx.name == self.config.index_name for idx in existing_indexes)
            
            if index_exists and delete_if_exists:
                logger.warning(f"Deleting existing index: {self.config.index_name}")
                self.pc.delete_index(self.config.index_name)
                index_exists = False
            
            if not index_exists:
                logger.info(f"Creating new index: {self.config.index_name}")
                
                self.pc.create_index(
                    name=self.config.index_name,
                    dimension=self.config.dimension,
                    metric=self.config.metric,
                    spec=ServerlessSpec(
                        cloud=self.config.cloud,
                        region=self.config.region
                    )
                )
                
                # Wait for index to be ready
                logger.info("Waiting for index to be ready...")
                await asyncio.sleep(10)
            
            # Connect to index
            self.index = self.pc.Index(self.config.index_name)
            
            # Get index stats
            stats = self.index.describe_index_stats()
            logger.info(f"Index ready. Current stats: {stats}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize index: {e}")
            return False
    
    async def upsert_vectors(self, 
                           vectors: list[list[float]], 
                           metadatas: list[VectorMetadata],
                           batch_size: int = 100) -> bool:
        """Upsert vectors to Pinecone"""
        if not self.index:
            raise RuntimeError("Index not initialized. Call initialize_index() first.")
        
        if len(vectors) != len(metadatas):
            raise ValueError("Number of vectors must match number of metadatas")
        
        try:
            # Prepare vectors for upsert
            vectors_to_upsert = []
            
            for i, (vector, metadata) in enumerate(zip(vectors, metadatas)):
                vector_data = {
                    "id": metadata.chunk_id,
                    "values": vector,
                    "metadata": metadata.to_dict()
                }
                vectors_to_upsert.append(vector_data)
            
            # Upsert in batches
            for i in range(0, len(vectors_to_upsert), batch_size):
                batch = vectors_to_upsert[i:i + batch_size]
                
                logger.info(f"Upserting batch {i//batch_size + 1}/{(len(vectors_to_upsert)-1)//batch_size + 1}")
                
                upsert_response = self.index.upsert(vectors=batch)
                
                if upsert_response.upserted_count != len(batch):
                    logger.warning(f"Expected to upsert {len(batch)} vectors, but upserted {upsert_response.upserted_count}")
                
                # Small delay between batches
                await asyncio.sleep(0.5)
            
            logger.info(f"Successfully upserted {len(vectors_to_upsert)} vectors")
            return True
            
        except Exception as e:
            logger.error(f"Failed to upsert vectors: {e}")
            return False
    
    async def query_vectors(self, 
                          query_vector: list[float],
                          top_k: int = 10,
                          filter_dict: Optional[Dict[str, Any]] = None,
                          include_metadata: bool = True) -> list[Dict[str, Any]]:
        """Query vectors from Pinecone"""
        if not self.index:
            raise RuntimeError("Index not initialized. Call initialize_index() first.")
        
        try:
            query_response = self.index.query(
                vector=query_vector,
                top_k=top_k,
                filter=filter_dict,
                include_metadata=include_metadata,
                include_values=False
            )
            
            # Extract results
            results = []
            for match in query_response.matches:
                result = {
                    "id": match.id,
                    "score": match.score,
                    "metadata": match.metadata if include_metadata else None
                }
                results.append(result)
            
            logger.info(f"Query returned {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Failed to query vectors: {e}")
            return []
    
    async def delete_vectors(self, ids: list[str]) -> bool:
        """Delete vectors from Pinecone"""
        if not self.index:
            raise RuntimeError("Index not initialized. Call initialize_index() first.")
        
        try:
            delete_response = self.index.delete(ids=ids)
            logger.info(f"Deleted vectors: {ids}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete vectors: {e}")
            return False
    
    async def delete_by_filter(self, filter_dict: Dict[str, Any]) -> bool:
        """Delete vectors by metadata filter"""
        if not self.index:
            raise RuntimeError("Index not initialized. Call initialize_index() first.")
        
        try:
            delete_response = self.index.delete(filter=filter_dict)
            logger.info(f"Deleted vectors with filter: {filter_dict}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete vectors by filter: {e}")
            return False
    
    def get_index_stats(self) -> Dict[str, Any]:
        """Get current index statistics"""
        if not self.index:
            return {}
        
        try:
            stats = self.index.describe_index_stats()
            return {
                "total_vector_count": stats.total_vector_count,
                "dimension": stats.dimension,
                "index_fullness": stats.index_fullness,
                "namespaces": dict(stats.namespaces) if stats.namespaces else {}
            }
        except Exception as e:
            logger.error(f"Failed to get index stats: {e}")
            return {}
