from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any

@dataclass
class VectorMetadata:
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

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}
