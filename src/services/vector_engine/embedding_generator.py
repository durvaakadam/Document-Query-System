import torch
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
import torch.nn.functional as F
import numpy as np
import logging
import asyncio
import threading
from .config import EmbeddingConfig
import logging

logger = logging.getLogger(__name__)

class HuggingFaceEmbeddingGenerator:
    """Generate embeddings using free Hugging Face models"""
    
    def __init__(self, config: EmbeddingConfig):
        self.config = config
        self.device = self._setup_device()
        self.model = None
        self.tokenizer = None
        self._model_lock = threading.Lock()
        
        # Track usage statistics
        self.total_texts_processed = 0
        self.total_batches = 0
        
        # Initialize model
        self._load_model()
    
    def _setup_device(self) -> str:
        """Setup computing device"""
        if self.config.device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
                logger.info("Using GPU (CUDA) for embeddings")
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = "mps"  # Apple Silicon
                logger.info("Using Apple Silicon GPU (MPS) for embeddings")
            else:
                device = "cpu"
                logger.info("Using CPU for embeddings")
        else:
            device = self.config.device
            logger.info(f"Using specified device: {device}")
        
        return device
    
    def _load_model(self):
        """Load the embedding model"""
        try:
            logger.info(f"Loading model: {self.config.model_name}")
            
            # Try to use SentenceTransformers first (easier and more efficient)
            if "sentence-transformers" in self.config.model_name:
                self.model = SentenceTransformer(self.config.model_name, device=self.device)
                self.model_type = "sentence_transformer"
                logger.info("Loaded SentenceTransformer model")
            else:
                # Fallback to transformers library
                self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
                self.model = AutoModel.from_pretrained(self.config.model_name)
                self.model.to(self.device)
                self.model.eval()
                self.model_type = "transformer"
                logger.info("Loaded Transformer model")
                
        except Exception as e:
            logger.error(f"Failed to load model {self.config.model_name}: {e}")
            # Fallback to a smaller, guaranteed-to-work model
            logger.info("Falling back to basic model: all-MiniLM-L6-v2")
            self.model = SentenceTransformer('all-MiniLM-L6-v2', device=self.device)
            self.model_type = "sentence_transformer"
            self.config.dimensions = 384
    
    def _mean_pooling(self, model_output, attention_mask):
        """Mean pooling for transformer models"""
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
    async def generate_embedding(self, text: str) -> list[float]:
        """Generate embedding for a single text"""
        try:
            with self._model_lock:
                if self.model_type == "sentence_transformer":
                    # Use SentenceTransformer (easier)
                    embedding = self.model.encode([text], convert_to_tensor=False, show_progress_bar=False)[0]
                else:
                    # Use raw transformer model
                    encoded_input = self.tokenizer(
                        text,
                        padding=True,
                        truncation=True,
                        max_length=self.config.max_length,
                        return_tensors='pt'
                    ).to(self.device)
                    
                    with torch.no_grad():
                        model_output = self.model(**encoded_input)
                        embedding = self._mean_pooling(model_output, encoded_input['attention_mask'])
                        embedding = F.normalize(embedding, p=2, dim=1)
                        embedding = embedding.cpu().numpy()[0]
            
            self.total_texts_processed += 1
            return embedding.tolist()
            
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            # Return zero vector as fallback
            return [0.0] * self.config.dimensions
    
    async def generate_embeddings_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts efficiently"""
        embeddings = []
        
        try:
            # Process in batches
            for i in range(0, len(texts), self.config.batch_size):
                batch = texts[i:i + self.config.batch_size]
                
                logger.info(f"Processing embedding batch {i//self.config.batch_size + 1}/{(len(texts)-1)//self.config.batch_size + 1}")
                
                with self._model_lock:
                    if self.model_type == "sentence_transformer":
                        # Use SentenceTransformer batch processing
                        batch_embeddings = self.model.encode(
                            batch,
                            convert_to_tensor=False,
                            show_progress_bar=False,
                            batch_size=len(batch)
                        )
                    else:
                        # Use transformer model with batching
                        encoded_input = self.tokenizer(
                            batch,
                            padding=True,
                            truncation=True,
                            max_length=self.config.max_length,
                            return_tensors='pt'
                        ).to(self.device)
                        
                        with torch.no_grad():
                            model_output = self.model(**encoded_input)
                            batch_embeddings = self._mean_pooling(model_output, encoded_input['attention_mask'])
                            batch_embeddings = F.normalize(batch_embeddings, p=2, dim=1)
                            batch_embeddings = batch_embeddings.cpu().numpy()
                
                # Convert to list of lists
                if isinstance(batch_embeddings, np.ndarray):
                    batch_embeddings = batch_embeddings.tolist()
                
                embeddings.extend(batch_embeddings)
                self.total_batches += 1
                
                # Small delay to prevent overheating
                await asyncio.sleep(0.1)
        
        except Exception as e:
            logger.error(f"Failed to generate batch embeddings: {e}")
            # Return zero vectors for failed batches
            failed_embeddings = [[0.0] * self.config.dimensions] * len(texts)
            embeddings.extend(failed_embeddings)
        
        self.total_texts_processed += len(texts)
        logger.info(f"Generated {len(embeddings)} embeddings (total processed: {self.total_texts_processed})")
        return embeddings
    
    def get_usage_stats(self) -> dict[str, any]:
        """Get embedding generation statistics"""
        return {
            "total_texts_processed": self.total_texts_processed,
            "total_batches": self.total_batches,
            "model_name": self.config.model_name,
            "model_type": self.model_type,
            "device": self.device,
            "embedding_dimensions": self.config.dimensions,
            "estimated_cost_usd": 0.0  # Free!
        }
