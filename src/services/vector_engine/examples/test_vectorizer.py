import os
import asyncio
from datetime import datetime
from vector_engine.config import EmbeddingConfig, PineconeConfig
from vector_engine.vectorizer import DocumentVectorizer
from vector_engine.models import VectorMetadata
from vector_engine.embedding_generator import EmbeddingGenerator

async def test_embedding_upsert():
    embedding_config = EmbeddingConfig()
    pinecone_config = PineconeConfig(
        api_key=os.getenv("PINECONE_API_KEY"),
        index_name="hackrx-documents"
    )
    openai_api_key = os.getenv("OPENAI_API_KEY")

    # Dummy extracted doc (mocked)
    class DummyChunk:
        def __init__(self, id, content):
            self.chunk_id = id
            self.chunk_type = "body"
            self.content = content
            self.page_number = 1
            self.section_title = "Sample Section"
            self.word_count = len(content.split())

    class DummyDocument:
        def __init__(self):
            self.document_id = "mock-doc-001"
            self.source_file = "mock.pdf"
            self.file_type = "pdf"
            self.chunks = [
                DummyChunk("chunk-001", "This is the first chunk of the document."),
                DummyChunk("chunk-002", "This is the second chunk related to insurance grace period."),
            ]

    vectorizer = DocumentVectorizer(embedding_config, pinecone_config, openai_api_key)
    await vectorizer.initialize(delete_existing_index=False)
    success = await vectorizer.vectorize_document(DummyDocument())
    
    print(f"\n✅ Vectorization test success: {success}")

if __name__ == "__main__":
    asyncio.run(test_embedding_upsert())
