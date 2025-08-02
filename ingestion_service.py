import asyncio
import aiohttp
import aiofiles
import hashlib
import mimetypes
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass
from urllib.parse import urlparse
from winmagic import magic
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DocumentInfo:
    """Document metadata and file information"""
    url: str
    file_path: str
    file_name: str
    file_size: int
    mime_type: str
    file_hash: str
    download_timestamp: datetime
    is_valid: bool
    validation_errors: List[str]

class DocumentValidationError(Exception):
    """Custom exception for document validation errors"""
    pass

class DocumentDownloadError(Exception):
    """Custom exception for document download errors"""
    pass

class DocumentIngestionService:
    """
    Service for downloading, validating and managing document ingestion
    """
    
    # Supported file types and their MIME types
    SUPPORTED_TYPES = {
        'application/pdf': ['.pdf'],
        'application/vnd.openxmlformats-officedocument.wordprocessingml.document': ['.docx'],
        'application/msword': ['.doc'],
        'message/rfc822': ['.eml'],
        'text/plain': ['.txt', '.eml'],
    }
    
    # Maximum file size (50MB)
    MAX_FILE_SIZE = 50 * 1024 * 1024
    
    # Request timeout (30 seconds)
    REQUEST_TIMEOUT = 30
    
    def __init__(self, cache_dir: Optional[str] = None):
        """
        Initialize the document ingestion service
        
        Args:
            cache_dir: Directory to cache downloaded documents
        """
        self.cache_dir = Path(cache_dir) if cache_dir else Path(tempfile.gettempdir()) / "doc_cache"
        self.cache_dir.mkdir(exist_ok=True, parents=True)
        
        # Session for HTTP requests
        self.session: Optional[aiohttp.ClientSession] = None
    
    async def __aenter__(self):
        """Async context manager entry"""
        connector = aiohttp.TCPConnector(limit=10, limit_per_host=5)
        timeout = aiohttp.ClientTimeout(total=self.REQUEST_TIMEOUT)
        self.session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            headers={
                'User-Agent': 'DocumentIngestionBot/1.0'
            }
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    def _generate_file_hash(self, content: bytes) -> str:
        """Generate SHA-256 hash of file content"""
        return hashlib.sha256(content).hexdigest()
    
    def _get_file_extension_from_url(self, url: str) -> str:
        """Extract file extension from URL"""
        parsed_url = urlparse(url)
        path = Path(parsed_url.path)
        return path.suffix.lower()
    
    def _generate_cache_filename(self, url: str, file_hash: str) -> str:
        """Generate a unique filename for caching"""
        url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
        extension = self._get_file_extension_from_url(url)
        return f"{url_hash}_{file_hash[:12]}{extension}"
    
    def _detect_mime_type(self, file_path: str, content: bytes) -> str:
        """
        Detect MIME type using multiple methods for accuracy
        """
        # Method 1: Use python-magic (most reliable)
        try:
            mime_type = magic.from_buffer(content, mime=True)
            if mime_type and mime_type != 'application/octet-stream':
                return mime_type
        except Exception as e:
            logger.warning(f"python-magic failed: {e}")
        
        # Method 2: Use mimetypes library with file extension
        mime_type, _ = mimetypes.guess_type(file_path)
        if mime_type:
            return mime_type
        
        # Method 3: Manual detection based on file signatures
        return self._detect_by_signature(content)
    
    def _detect_by_signature(self, content: bytes) -> str:
        """Detect file type by examining file signature/magic bytes"""
        if not content:
            return 'application/octet-stream'
        
        # PDF signature
        if content.startswith(b'%PDF'):
            return 'application/pdf'
        
        # DOCX/ZIP signature (DOCX files are ZIP archives)
        if content.startswith(b'PK\x03\x04'):
            # Check if it's a DOCX by looking for specific content
            if b'word/' in content[:1024] or b'[Content_Types].xml' in content[:1024]:
                return 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
            return 'application/zip'
        
        # DOC signature
        if content.startswith(b'\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1'):
            return 'application/msword'
        
        # Email signatures
        if (content.startswith(b'Return-Path:') or 
            content.startswith(b'Received:') or 
            content.startswith(b'From:') or
            b'MIME-Version:' in content[:500]):
            return 'message/rfc822'
        
        return 'application/octet-stream'
    
    def _validate_file_type(self, mime_type: str, file_extension: str) -> Tuple[bool, List[str]]:
        """
        Validate if the file type is supported
        
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        # Check if MIME type is supported
        if mime_type not in self.SUPPORTED_TYPES:
            errors.append(f"Unsupported MIME type: {mime_type}")
            return False, errors
        
        # Check if file extension matches MIME type
        expected_extensions = self.SUPPORTED_TYPES[mime_type]
        if file_extension not in expected_extensions:
            # This is a warning, not a critical error
            logger.warning(f"Extension {file_extension} doesn't match MIME type {mime_type}")
        
        return True, errors
    
    def _validate_file_size(self, size: int) -> Tuple[bool, List[str]]:
        """Validate file size"""
        errors = []
        
        if size == 0:
            errors.append("File is empty")
            return False, errors
        
        if size > self.MAX_FILE_SIZE:
            errors.append(f"File too large: {size} bytes (max: {self.MAX_FILE_SIZE})")
            return False, errors
        
        return True, errors
    
    def _validate_content(self, content: bytes, mime_type: str) -> Tuple[bool, List[str]]:
        """
        Validate file content based on type
        """
        errors = []
        
        if not content:
            errors.append("Empty file content")
            return False, errors
        
        # PDF-specific validation
        if mime_type == 'application/pdf':
            if not content.startswith(b'%PDF'):
                errors.append("Invalid PDF: Missing PDF signature")
                return False, errors
            
            if b'%%EOF' not in content:
                errors.append("Warning: PDF may be truncated (missing EOF marker)")
        
        # DOCX-specific validation
        elif mime_type == 'application/vnd.openxmlformats-officedocument.wordprocessingml.document':
            if not content.startswith(b'PK'):
                errors.append("Invalid DOCX: Not a valid ZIP archive")
                return False, errors
        
        return True, errors
    
    async def _download_with_retry(self, url: str, max_retries: int = 3) -> bytes:
        """
        Download file with retry logic
        """
        last_error = None
        
        for attempt in range(max_retries):
            try:
                logger.info(f"Downloading {url} (attempt {attempt + 1}/{max_retries})")
                
                async with self.session.get(url) as response:
                    # Check HTTP status
                    if response.status != 200:
                        raise DocumentDownloadError(f"HTTP {response.status}: {response.reason}")
                    
                    # Check content length
                    content_length = response.headers.get('content-length')
                    if content_length and int(content_length) > self.MAX_FILE_SIZE:
                        raise DocumentDownloadError(f"File too large: {content_length} bytes")
                    
                    # Download content
                    content = await response.read()
                    
                    # Validate downloaded size
                    if len(content) > self.MAX_FILE_SIZE:
                        raise DocumentDownloadError(f"Downloaded file too large: {len(content)} bytes")
                    
                    logger.info(f"Successfully downloaded {len(content)} bytes from {url}")
                    return content
                    
            except asyncio.TimeoutError:
                last_error = DocumentDownloadError(f"Timeout downloading {url}")
                logger.warning(f"Timeout on attempt {attempt + 1}, retrying...")
                
            except aiohttp.ClientError as e:
                last_error = DocumentDownloadError(f"Network error: {str(e)}")
                logger.warning(f"Network error on attempt {attempt + 1}: {e}")
                
            except Exception as e:
                last_error = DocumentDownloadError(f"Unexpected error: {str(e)}")
                logger.error(f"Unexpected error on attempt {attempt + 1}: {e}")
            
            # Wait before retry (exponential backoff)
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                logger.info(f"Waiting {wait_time} seconds before retry...")
                await asyncio.sleep(wait_time)
        
        # All retries failed
        raise last_error
    
    async def ingest_document(self, document_url: str, use_cache: bool = True) -> DocumentInfo:
        """
        Main method to ingest a document from URL
        
        Args:
            document_url: URL of the document to download
            use_cache: Whether to use cached version if available
            
        Returns:
            DocumentInfo object with all document metadata
        """
        if not self.session:
            raise RuntimeError("Service not initialized. Use async context manager.")
        
        logger.info(f"Starting document ingestion for: {document_url}")
        
        try:
            # Download document
            content = await self._download_with_retry(document_url)
            
            # Generate hash
            file_hash = self._generate_file_hash(content)
            
            # Check cache if enabled
            if use_cache:
                cached_path = self.cache_dir / self._generate_cache_filename(document_url, file_hash)
                if cached_path.exists():
                    logger.info(f"Using cached document: {cached_path}")
                    return await self._create_document_info_from_cache(document_url, str(cached_path), file_hash)
            
            # Detect file type
            file_extension = self._get_file_extension_from_url(document_url)
            mime_type = self._detect_mime_type(document_url, content)
            
            # Validate document
            validation_errors = []
            
            # Validate file type
            type_valid, type_errors = self._validate_file_type(mime_type, file_extension)
            validation_errors.extend(type_errors)
            
            # Validate file size
            size_valid, size_errors = self._validate_file_size(len(content))
            validation_errors.extend(size_errors)
            
            # Validate content
            content_valid, content_errors = self._validate_content(content, mime_type)
            validation_errors.extend(content_errors)
            
            is_valid = type_valid and size_valid and content_valid
            
            # Save to cache
            cache_filename = self._generate_cache_filename(document_url, file_hash)
            cache_path = self.cache_dir / cache_filename
            
            async with aiofiles.open(cache_path, 'wb') as f:
                await f.write(content)
            
            logger.info(f"Document cached at: {cache_path}")
            
            # Create document info
            doc_info = DocumentInfo(
                url=document_url,
                file_path=str(cache_path),
                file_name=cache_filename,
                file_size=len(content),
                mime_type=mime_type,
                file_hash=file_hash,
                download_timestamp=datetime.now(),
                is_valid=is_valid,
                validation_errors=validation_errors
            )
            
            if validation_errors:
                logger.warning(f"Document validation issues: {validation_errors}")
            
            logger.info(f"Document ingestion completed: {doc_info.file_name}")
            return doc_info
            
        except Exception as e:
            logger.error(f"Document ingestion failed for {document_url}: {str(e)}")
            raise DocumentDownloadError(f"Failed to ingest document: {str(e)}")
    
    async def _create_document_info_from_cache(self, url: str, file_path: str, file_hash: str) -> DocumentInfo:
        """Create DocumentInfo from cached file"""
        file_path_obj = Path(file_path)
        
        # Read a small portion to detect MIME type
        async with aiofiles.open(file_path, 'rb') as f:
            content_sample = await f.read(1024)
        
        mime_type = self._detect_mime_type(file_path, content_sample)
        
        return DocumentInfo(
            url=url,
            file_path=file_path,
            file_name=file_path_obj.name,
            file_size=file_path_obj.stat().st_size,
            mime_type=mime_type,
            file_hash=file_hash,
            download_timestamp=datetime.fromtimestamp(file_path_obj.stat().st_mtime),
            is_valid=True,  # Assume cached files are valid
            validation_errors=[]
        )
    
    async def ingest_multiple_documents(self, document_urls: List[str], max_concurrent: int = 5) -> List[DocumentInfo]:
        """
        Ingest multiple documents concurrently
        
        Args:
            document_urls: List of document URLs
            max_concurrent: Maximum number of concurrent downloads
            
        Returns:
            List of DocumentInfo objects
        """
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def bounded_ingest(url):
            async with semaphore:
                try:
                    return await self.ingest_document(url)
                except Exception as e:
                    logger.error(f"Failed to ingest {url}: {e}")
                    # Return a failed DocumentInfo object
                    return DocumentInfo(
                        url=url,
                        file_path="",
                        file_name="",
                        file_size=0,
                        mime_type="",
                        file_hash="",
                        download_timestamp=datetime.now(),
                        is_valid=False,
                        validation_errors=[str(e)]
                    )
        
        logger.info(f"Starting batch ingestion of {len(document_urls)} documents")
        
        tasks = [bounded_ingest(url) for url in document_urls]
        results = await asyncio.gather(*tasks, return_exceptions=False)
        
        successful = sum(1 for r in results if r.is_valid)
        logger.info(f"Batch ingestion completed: {successful}/{len(document_urls)} successful")
        
        return results
    
    def clear_cache(self, older_than_days: Optional[int] = None):
        """
        Clear document cache
        
        Args:
            older_than_days: Only clear files older than this many days
        """
        if older_than_days:
            cutoff_time = datetime.now().timestamp() - (older_than_days * 24 * 60 * 60)
            
            for file_path in self.cache_dir.glob("*"):
                if file_path.is_file() and file_path.stat().st_mtime < cutoff_time:
                    file_path.unlink()
                    logger.info(f"Removed old cached file: {file_path.name}")
        else:
            # Clear all cache
            for file_path in self.cache_dir.glob("*"):
                if file_path.is_file():
                    file_path.unlink()
            logger.info("Cleared all cached files")


# Usage example and testing
async def main():
    """Example usage of the DocumentIngestionService"""
    
    # Example document URLs (replace with actual URLs)
    test_urls = [
        "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D"
    ]
    
    # Initialize service with cache directory
    async with DocumentIngestionService(cache_dir="./document_cache") as service:
        
        # Ingest single document
        try:
            doc_info = await service.ingest_document(test_urls[0])
            print(f"Successfully ingested: {doc_info.file_name}")
            print(f"File size: {doc_info.file_size} bytes")
            print(f"MIME type: {doc_info.mime_type}")
            print(f"Valid: {doc_info.is_valid}")
            if doc_info.validation_errors:
                print(f"Validation errors: {doc_info.validation_errors}")
        
        except Exception as e:
            print(f"Failed to ingest document: {e}")
        
        # Ingest multiple documents (if you have more URLs)
        # results = await service.ingest_multiple_documents(test_urls)
        # print(f"Processed {len(results)} documents")

if __name__ == "__main__":
    asyncio.run(main())