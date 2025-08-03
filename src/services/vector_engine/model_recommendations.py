# Free alternative models (choose based on your needs)
class ModelRecommendations:
    """Recommended free models for different use cases"""
    
    # Best balance of speed and quality (default)
    FAST_AND_GOOD = "sentence-transformers/all-MiniLM-L6-v2"  # 384 dim
    
    # Higher quality, slower
    HIGH_QUALITY = "sentence-transformers/all-mpnet-base-v2"  # 768 dim
    
    # Fastest, smaller
    FASTEST = "sentence-transformers/all-MiniLM-L12-v1"  # 384 dim
    
    # Domain-specific (legal/financial documents)
    LEGAL_DOCUMENTS = "nlpaueb/legal-bert-base-uncased"  # 768 dim
    
    # Multilingual support
    MULTILINGUAL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"  # 384 dim
