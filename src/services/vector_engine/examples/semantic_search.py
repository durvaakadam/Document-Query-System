import os
import asyncio
from vector_engine.config import EmbeddingConfig, PineconeConfig
from vector_engine.vectorizer import DocumentVectorizer

async def run_search():
    query_list = [
        "What is the grace period for premium payment?",
        "What is the waiting period for pre-existing diseases?",
        "Does this policy cover maternity expenses?"
    ]

    embedding_config = EmbeddingConfig()
    pinecone_config = PineconeConfig(
        api_key=os.getenv("PINECONE_API_KEY"),
        index_name="hackrx-documents"
    )
    openai_api_key = os.getenv("OPENAI_API_KEY")

    vectorizer = DocumentVectorizer(embedding_config, pinecone_config, openai_api_key)
    await vectorizer.initialize()

    for query in query_list:
        print(f"\n🔍 Query: {query}")
        results = await vectorizer.search_similar_chunks(query, top_k=3)
        for i, result in enumerate(results, 1):
            print(f"{i}. Score: {result['similarity_score']:.4f}")
            print(f"   Section: {result['section_title']}")
            print(f"   Content: {result['content'][:200]}...")

if __name__ == "__main__":
    asyncio.run(run_search())
