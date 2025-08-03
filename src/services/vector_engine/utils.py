import tiktoken

class TokenCounter:
    """Utility class for counting tokens"""
    
    def __init__(self, model: str = "text-embedding-3-small"):
        try:
            self.encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            # Fallback to a common encoding
            self.encoding = tiktoken.get_encoding("cl100k_base")
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        return len(self.encoding.encode(text))
    
    def truncate_text(self, text: str, max_tokens: int) -> str:
        """Truncate text to fit within token limit"""
        tokens = self.encoding.encode(text)
        if len(tokens) <= max_tokens:
            return text
        
        # Truncate and decode back to text
        truncated_tokens = tokens[:max_tokens]
        return self.encoding.decode(truncated_tokens)