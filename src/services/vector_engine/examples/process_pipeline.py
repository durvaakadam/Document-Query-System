import os
import asyncio
from vector_engine.config import EmbeddingConfig, PineconeConfig
from vector_engine.pipeline import complete_pipeline_with_vectorization

async def run_pipeline():
    document_url = "https://hackrx.blob.core.windows.net/assets/policy.pdf?..."

    embedding_config = EmbeddingConfig()
    pinecone_config = PineconeConfig(
        api_key=os.getenv("PINECONE_API_KEY"),
        index_name="hackrx-documents"
    )
    openai_api_key = os.getenv("OPENAI_API_KEY")

    result = await complete_pipeline_with_vectorization(
        document_url,
        embedding_config,
        pinecone_config,
        openai_api_key
    )

    print("\n✅ Pipeline Result:")
    for key, value in result.items():
        print(f"{key}: {value}")

if __name__ == "__main__":
    asyncio.run(run_pipeline())
