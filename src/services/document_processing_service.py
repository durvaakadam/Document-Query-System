import asyncio
import os
from pathlib import Path
from typing import List, Dict, Optional
import logging
from dataclasses import dataclass

# For PDF processing
try:
    import PyPDF2
    import fitz  # PyMuPDF
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

# For DOCX processing
try:
    import docx
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DocumentContent:
    """Extracted document content and metadata"""
    file_path: str
    text_content: str
    page_count: Optional[int] = None
    word_count: Optional[int] = None
    extraction_method: str = ""
    errors: List[str] = None

    def __post_init__(self):
        if self.errors is None:
            self.errors = []
        if self.word_count is None:
            self.word_count = len(self.text_content.split()) if self.text_content else 0

class DocumentProcessingService:
    """
    Service for processing documents and answering questions
    """

    def __init__(self, cache_dir: Optional[str] = None):
        """
        Initialize the document processing service
        """
        self.cache_dir = Path(cache_dir) if cache_dir else Path("./document_cache")

    async def extract_text_from_pdf(self, file_path: str) -> DocumentContent:
        content = DocumentContent(file_path=file_path, text_content="", extraction_method="pdf")

        if not PDF_AVAILABLE:
            content.errors.append("PDF processing libraries not available. Install PyPDF2 and PyMuPDF.")
            return content

        try:
            doc = fitz.open(file_path)
            text_parts = [doc[page_num].get_text() for page_num in range(len(doc))]
            content.text_content = "\n".join(text_parts)
            content.page_count = len(doc)
            content.extraction_method = "PyMuPDF"
            doc.close()

        except Exception as e:
            logger.warning(f"PyMuPDF failed, trying PyPDF2: {e}")
            try:
                with open(file_path, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    text_parts = [page.extract_text() for page in pdf_reader.pages]
                    content.text_content = "\n".join(text_parts)
                    content.page_count = len(pdf_reader.pages)
                    content.extraction_method = "PyPDF2"
            except Exception as e2:
                content.errors.append(f"PDF extraction failed: PyMuPDF: {e}, PyPDF2: {e2}")
        return content

    async def extract_text_from_docx(self, file_path: str) -> DocumentContent:
        content = DocumentContent(file_path=file_path, text_content="", extraction_method="docx")

        if not DOCX_AVAILABLE:
            content.errors.append("DOCX processing library not available. Install python-docx.")
            return content

        try:
            doc = docx.Document(file_path)
            text_parts = [para.text for para in doc.paragraphs if para.text.strip()]
            content.text_content = "\n".join(text_parts)
            content.extraction_method = "python-docx"
        except Exception as e:
            content.errors.append(f"DOCX extraction failed: {e}")
        return content

    async def extract_text_from_txt(self, file_path: str) -> DocumentContent:
        content = DocumentContent(file_path=file_path, text_content="", extraction_method="plain_text")
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content.text_content = file.read()
        except UnicodeDecodeError:
            try:
                with open(file_path, 'r', encoding='latin-1') as file:
                    content.text_content = file.read()
            except Exception as e:
                content.errors.append(f"Text file reading failed: {e}")
        except Exception as e:
            content.errors.append(f"Text file reading failed: {e}")
        return content

    async def extract_text_from_file(self, file_path: str) -> DocumentContent:
        file_path_obj = Path(file_path)
        if not file_path_obj.exists():
            return DocumentContent(file_path=file_path, text_content="", errors=[f"File not found: {file_path}"])
        extension = file_path_obj.suffix.lower()

        if extension == '.pdf':
            return await self.extract_text_from_pdf(file_path)
        elif extension in ['.docx']:
            return await self.extract_text_from_docx(file_path)
        elif extension in ['.txt', '.eml']:
            return await self.extract_text_from_txt(file_path)
        else:
            return DocumentContent(file_path=file_path, text_content="", errors=[f"Unsupported file type: {extension}"])

    async def process_questions(self, document_contents: List[DocumentContent], questions: List[str]) -> List[str]:
        answers = []
        combined_text = "\n\n".join([doc.text_content for doc in document_contents if doc.text_content and not doc.errors])

        if not combined_text.strip():
            return ["No valid document content found to process questions."] * len(questions)

        for question in questions:
            try:
                answer = await self._generate_answer(combined_text, question)
                answers.append(answer)
            except Exception as e:
                logger.error(f"Error processing question '{question}': {e}")
                answers.append(f"Error processing question: {str(e)}")
        return answers

    async def _generate_answer(self, document_text: str, question: str) -> str:
        question_lower = question.lower()
        text_lower = document_text.lower()

        question_words = [word.strip('.,!?;:"()[]') for word in question_lower.split()
                          if len(word) > 3 and word not in [
                              'what', 'when', 'where', 'who', 'how', 'why', 'which', 'that', 'this', 'the', 'and', 'for', 'with'
                          ]]

        if not question_words:
            return "Unable to extract meaningful keywords from the question."

        sentences = document_text.split('.')
        relevant_sentences = []

        for sentence in sentences:
            sentence_lower = sentence.lower()
            relevance_score = sum(1 for word in question_words if word in sentence_lower)
            if relevance_score > 0:
                relevant_sentences.append((sentence.strip(), relevance_score))

        if not relevant_sentences:
            return f"No relevant information found in the documents for the question: '{question}'"

        relevant_sentences.sort(key=lambda x: x[1], reverse=True)
        top_sentences = [sent[0] for sent in relevant_sentences[:3] if sent[0]]

        if top_sentences:
            return "Based on the documents: " + " ".join(top_sentences)
        else:
            return f"Found some relevant content but unable to extract a clear answer for: '{question}'"

    async def process_documents_and_questions(self, document_identifiers: List[str], questions: List[str]) -> Dict:
        logger.info(f"Processing {len(document_identifiers)} documents and {len(questions)} questions")
        document_contents = []

        for doc_id in document_identifiers:
            file_path = doc_id if os.path.isabs(doc_id) else self.cache_dir / doc_id
            if not Path(file_path).exists():
                matching_files = list(self.cache_dir.glob(f"*{doc_id}*"))
                file_path = matching_files[0] if matching_files else None
                if not file_path:
                    logger.warning(f"Document not found: {doc_id}")
                    continue

            content = await self.extract_text_from_file(str(file_path))
            document_contents.append(content)

            if content.errors:
                logger.warning(f"Errors extracting from {doc_id}: {content.errors}")

        answers = await self.process_questions(document_contents, questions)

        return {
            "answers": answers,
            "metadata": {
                "documents_processed": len([doc for doc in document_contents if not doc.errors]),
                "total_documents": len(document_identifiers),
                "total_words_processed": sum(doc.word_count or 0 for doc in document_contents),
                "extraction_errors": [doc.errors for doc in document_contents if doc.errors]
            }
        }

# Example usage
async def main():
    service = DocumentProcessingService(cache_dir="src/services/document_cache")
    
    documents = ["doc1.pdf"]  # ✅ Updated document filename
    questions = [
        "What is the coverage amount?",
        "Who is eligible for this policy?"
    ]
    result = await service.process_documents_and_questions(documents, questions)
    print("Answers:")
    for i, answer in enumerate(result["answers"]):
        print(f"{i+1}. {answer}")
    print("\nMetadata:", result["metadata"])

if __name__ == "__main__":
    asyncio.run(main())
