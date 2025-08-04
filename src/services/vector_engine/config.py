from dataclasses import dataclass
from typing import Optional, List, Dict, Any
import os


@dataclass
class EmbeddingConfig:
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    dimensions: int = 384
    max_length: int = 512
    batch_size: int = 32
    device: str = "auto"
    
@dataclass
class PineconeConfig:
    region: str = "us-east-1"  # Changed from environment to region
    api_key: str = os.getenv("PINECONE_API_KEY")
    index_name: str = "hackrx-insurace-docs"
    dimension: int = 384
    metric: str = "cosine"
    cloud: str = "aws"  

    




