import os
import asyncio
from dotenv import load_dotenv
load_dotenv()
from services.vector_engine.config import EmbeddingConfig, PineconeConfig
from services.vector_engine.vectorizer import DocumentVectorizer
from services.vector_engine.pipeline import complete_pipeline_with_free_vectorization

async def process_hackrx_policy_document():
    """Process the hackathon policy document with free models"""
    
    # Use the provided document URL
    document_url = "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D"
    
    # Configure for insurance documents
    embedding_config = EmbeddingConfig()
    embedding_config.model_name = "sentence-transformers/all-MiniLM-L6-v2"  # Fast and good
    embedding_config.dimensions = 384
    embedding_config.batch_size = 24
    
    pinecone_config = PineconeConfig(
        api_key=os.getenv("PINECONE_API_KEY"),
        environment=os.getenv("PINECONE_ENVIRONMENT", "gcp-starter"),
        index_name="hackrx-insurance-docs"
    )
    
    # Run the complete free pipeline
    results = await complete_pipeline_with_free_vectorization(
        document_url,
        embedding_config,
        pinecone_config
    )
    
    return results

# Test with hackathon questions
async def test_hackrx_queries():
    # Configure for insurance documents
    embedding_config = EmbeddingConfig()
    embedding_config.model_name = "sentence-transformers/all-MiniLM-L6-v2"  # Fast and good
    embedding_config.dimensions = 384
    embedding_config.batch_size = 24
    
    pinecone_config = PineconeConfig(
        api_key=os.getenv("PINECONE_API_KEY"),
        environment=os.getenv("PINECONE_ENVIRONMENT", "gcp-starter"),
        index_name="hackrx-insurance-docs"
    )
    
    vectorizer = DocumentVectorizer(embedding_config, pinecone_config)
    await vectorizer.initialize()
    
    # Hackathon sample questions
    questions = [
        "What is the grace period for premium payment under the National Parivar Mediclaim Plus Policy?",
        "What is the waiting period for pre-existing diseases (PED) to be covered?",
        "Does this policy cover maternity expenses, and what are the conditions?",
        "What is the waiting period for cataract surgery?",
        "Are the medical expenses for an organ donor covered under this policy?",
        "What is the No Claim Discount (NCD) offered in this policy?",
        "Is there a benefit for preventive health check-ups?",
        "How does the policy define a 'Hospital'?",
        "What is the extent of coverage for AYUSH treatments?",
        "Are there any sub-limits on room rent and ICU charges for Plan A?"
    ]
    
    results = {}
    for question in questions:
        search_results = await vectorizer.search_similar_chunks(question, top_k=3)
        results[question] = search_results
    
    return results

if __name__ == "__main__":
    # Run the policy document processing
    print("Processing policy document...")
    results = asyncio.run(process_hackrx_policy_document())
    print(f"Results: {results}")
    
    # Run the query testing
    print("Testing queries...")
    query_results = asyncio.run(test_hackrx_queries())
    print(f"Query results: {query_results}")
