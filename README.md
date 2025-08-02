# Document-Query-System
# Document Ingestion Module

This module is part of the LLM-Powered Query-Retrieval System and is responsible for securely downloading and validating enterprise documents (PDFs, DOCX, EML, etc.) for downstream processing.

---

## ✅ Features

- Asynchronous downloading using `aiohttp`
- File streaming to disk (memory efficient)
- MIME type detection (with optional `python-magic`)
- File size, type, and signature validation
- SHA-256 hashing and caching to avoid duplicates
- Support for concurrent downloads

---

## 🚀 Usage

1. **Install dependencies**

```bash
pip install -r requirements.txt