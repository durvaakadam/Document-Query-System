import logging
import asyncio
from typing import List, Dict, Any
from openai import AsyncOpenAI
from .utils import TokenCounter
from .config import EmbeddingConfig

logger = logging.getLogger(__name__)

class EmbeddingGenerator:
    """Generate embeddings using OpenAI API"""
    
    def __init__(self, config: EmbeddingConfig, openai_api_key: str):
        self.config = config
        self.client = AsyncOpenAI(api_key=openai_api_key)
        self.token_counter = TokenCounter(config.model)
        
        # Track usage statistics
        self.total_tokens_used = 0
        self.total_requests = 0
    
    async def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for a single text"""
        try:
            # Truncate text if necessary
            processed_text = self.token_counter.truncate_text(text, self.config.max_tokens)
            
            response = await self.client.embeddings.create(
                model=self.config.model,
                input=processed_text,
                dimensions=self.config.dimensions
            )
            
            # Update usage statistics
            self.total_tokens_used += response.usage.total_tokens
            self.total_requests += 1
            
            return response.data[0].embedding
            
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            raise
    
    async def generate_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts in batches"""
        embeddings = []
        
        # Process in batches to respect API limits
        for i in range(0, len(texts), self.config.batch_size):
            batch = texts[i:i + self.config.batch_size]
            
            # Truncate texts in batch
            processed_batch = [
                self.token_counter.truncate_text(text, self.config.max_tokens) 
                for text in batch
            ]
            
            try:
                logger.info(f"Processing embedding batch {i//self.config.batch_size + 1}/{(len(texts)-1)//self.config.batch_size + 1}")
                
                response = await self.client.embeddings.create(
                    model=self.config.model,
                    input=processed_batch,
                    dimensions=self.config.dimensions
                )
                
                # Extract embeddings
                batch_embeddings = [item.embedding for item in response.data]
                embeddings.extend(batch_embeddings)
                
                # Update usage statistics
                self.total_tokens_used += response.usage.total_tokens
                self.total_requests += 1
                
                # Small delay to respect rate limits
                await asyncio.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Failed to generate batch embeddings: {e}")
                # Return zero vectors for failed batch
                failed_embeddings = [[0.0] * self.config.dimensions] * len(batch)
                embeddings.extend(failed_embeddings)
        
        logger.info(f"Generated {len(embeddings)} embeddings using {self.total_tokens_used} tokens")
        return embeddings
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get embedding generation statistics"""
        return {
            "total_tokens_used": self.total_tokens_used,
            "total_requests": self.total_requests,
            "estimated_cost_usd": self.total_tokens_used * 0.00002  # Approximate cost per token
        }
