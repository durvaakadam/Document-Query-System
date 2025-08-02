import asyncio
from ingestion_service import DocumentIngestionService

async def test_ingestion():
    test_urls = [
        "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D"
    ]

    async with DocumentIngestionService(
        cache_dir="./document_cache"
    ) as service:
        results = await service.ingest_multiple_documents(test_urls)
        for doc in results:
            print("File Name:", doc.file_name)
            print("Valid:", doc.is_valid)
            print("MIME Type:", doc.mime_type)
            print("Size (bytes):", doc.file_size)
            print("Errors:", doc.validation_errors)
            print("---")

if __name__ == "__main__":
    asyncio.run(test_ingestion())