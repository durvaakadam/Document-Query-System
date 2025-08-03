from dataclasses import dataclass
from typing import Optional

@dataclass
class VectorMetadata:
    """Metadata stored with each vector in Pinecone"""
    document_id: str
    chunk_id: str
    chunk_type: str
    content: str
    page_number: Optional[int] = None
    section_title: Optional[str] = None
    word_count: int = 0
    file_type: str = ""
    source_file: str = ""
    created_at: str = ""
    
    def to_dict(self) -> dict:
        """Convert metadata to dictionary for Pinecone storage"""
        return {
            "document_id": self.document_id,
            "chunk_id": self.chunk_id,
            "chunk_type": self.chunk_type,
            "content": self.content,
            "page_number": self.page_number,
            "section_title": self.section_title,
            "word_count": self.word_count,
            "file_type": self.file_type,
            "source_file": self.source_file,
            "created_at": self.created_at
        }
