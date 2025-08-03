from dataclasses import dataclass
from typing import Optional, List, Dict, Any


class EmbeddingConfig:
    """Configuration for embedding generation"""
    model: str = "text-embedding-3-small"  # OpenAI embedding model
    dimensions: int = 1536  # Embedding dimensions
    max_tokens: int = 8000  # Max tokens per chunk
    batch_size: int = 100  # Batch size for API calls
    
@dataclass
class PineconeConfig:
    """Configuration for Pinecone vector database"""
    api_key: str
    environment: str = "us-west1-gcp-free"  # Free tier environment
    index_name: str = "hackrx-documents"
    dimension: int = 1536
    metric: str = "cosine"
    cloud: str = "gcp"
    region: str = "us-west1"


