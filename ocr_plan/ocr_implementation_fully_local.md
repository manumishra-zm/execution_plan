# OCR Implementation: Fully Local (Zero API Cost)

## Overview

This approach uses **PaddleOCR** (primary for tables/invoices) and **Surya OCR** (fallback for general documents) with local parsers for digital files. No external API calls, 100% free.

```mermaid
flowchart TD
    A[File Upload] --> B{File Type?}

    B -->|.txt .md .csv| C[Direct Read]
    B -->|.docx| D[python-docx]
    B -->|.xlsx .xls| E[openpyxl]
    B -->|.pdf| F{Has Text?}
    B -->|Images| G{Content Analysis}

    F -->|Yes| H[PyMuPDF Extract]
    F -->|No| G

    G -->|Tables/Invoices| I[PaddleOCR]
    G -->|General Docs| J[Surya OCR]

    I --> K{Confidence >= 55%?}
    J --> K

    K -->|Yes| L[Accept Result]
    K -->|No| M[Try Other Engine]

    M --> L

    C --> N[Structured Output]
    D --> N
    E --> N
    H --> N
    L --> N

    N --> O[Inject to Chat]

    style C fill:#2D5016,color:#fff
    style D fill:#2D5016,color:#fff
    style E fill:#2D5016,color:#fff
    style H fill:#2D5016,color:#fff
    style I fill:#1E3A5F,color:#fff
    style J fill:#1E3A5F,color:#fff
    style M fill:#1E3A5F,color:#fff
```

**Legend:**
- Dark Green = Free (local parsers)
- Dark Blue = Free (local OCR - PaddleOCR/Surya)

---

## Cost

| Item | Cost |
|------|------|
| All processing | **$0.00** |
| Monthly total | **$0.00** |

---

## OCR Engine Selection

| Content Type | Primary Engine | Fallback | Reason |
|-------------|----------------|----------|--------|
| Tables | PaddleOCR | Surya | PP-Structure excels at tables |
| Invoices | PaddleOCR | Surya | Better structured data extraction |
| Forms | PaddleOCR | Surya | Good checkbox/field recognition |
| General Documents | Surya | PaddleOCR | Better layout detection |
| Handwritten | Surya | PaddleOCR | Superior handwriting support |

---

## Limitations

- Cannot interpret graphs/charts (just extracts text/numbers)
- No visual understanding (unlike Gemini)
- Slower than API-based solutions on first load (model loading)

---

## Dependencies

### requirements.txt additions

```text
# ============================================
# FULLY LOCAL OCR DEPENDENCIES
# ============================================

# OCR Engines (FREE)
paddlepaddle>=2.6.0           # PaddlePaddle framework
paddleocr>=2.7.0              # PaddleOCR + PP-Structure
surya-ocr>=0.6.0              # Surya OCR (layout + recognition)

# Document Parsers (FREE)
python-docx>=1.1.0            # .docx parsing
openpyxl>=3.1.0               # .xlsx parsing
PyMuPDF>=1.24.0               # PDF text extraction
pandas>=2.2.0                 # Data handling

# File Handling
Pillow>=10.0.0                # Image processing
python-multipart>=0.0.9       # FastAPI file upload
aiofiles>=24.1.0              # Async file I/O
python-magic>=0.4.27          # MIME detection
numpy>=1.26.0                 # Array operations
```

### Dockerfile additions

```dockerfile
# Add to existing Dockerfile
RUN apt-get update && apt-get install -y --no-install-recommends \
    libmagic1 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
```

---

## Project Structure

```
app/
├── services/
│   └── file_processing/
│       ├── __init__.py
│       ├── router.py              # Routes file to handler
│       ├── base_handler.py        # Abstract base class
│       ├── text_handler.py        # .txt, .md, .csv
│       ├── docx_handler.py        # .docx
│       ├── excel_handler.py       # .xlsx, .xls
│       ├── pdf_handler.py         # .pdf
│       ├── image_handler.py       # Images
│       ├── ocr/
│       │   ├── __init__.py
│       │   ├── paddle_ocr.py      # PaddleOCR wrapper
│       │   ├── surya_ocr.py       # Surya OCR wrapper
│       │   └── ocr_router.py      # Routes to best engine
│       ├── classifiers/
│       │   └── table_detector.py  # Detect tables in images
│       └── models.py              # Pydantic schemas
├── api/
│   └── v1/
│       └── endpoints/
│           └── upload.py          # Upload endpoint
```

---

## Implementation

### Step 1: Pydantic Models

**File: `app/services/file_processing/models.py`**

```python
"""Pydantic models for file processing."""

from enum import StrEnum
from pydantic import BaseModel, Field


class ExtractionMethod(StrEnum):
    """Method used to extract content."""
    DIRECT_READ = "direct_read"
    PYTHON_DOCX = "python_docx"
    OPENPYXL = "openpyxl"
    PYMUPDF = "pymupdf"
    PADDLE_OCR = "paddle_ocr"
    SURYA_OCR = "surya_ocr"


class ImageType(StrEnum):
    """Type of image content."""
    DOCUMENT = "document"
    TABLE = "table"
    INVOICE = "invoice"
    FORM = "form"
    UNKNOWN = "unknown"


class ExtractionMetadata(BaseModel):
    """Metadata about the extraction process."""
    file_type: str
    file_size: int
    pages: int | None = None
    extraction_method: ExtractionMethod
    confidence: float = Field(ge=0.0, le=1.0, default=1.0)
    processing_time_ms: int
    ocr_engines_used: list[str] = []


class TableData(BaseModel):
    """Extracted table data."""
    headers: list[str]
    rows: list[list[str]]
    confidence: float


class ExtractedContent(BaseModel):
    """Extracted content from a file."""
    text: str
    structured_data: dict | None = None
    tables: list[TableData] | None = None
    markdown: str
    metadata: ExtractionMetadata


class UploadResponse(BaseModel):
    """Response from upload endpoint."""
    file_id: str
    filename: str
    extraction: ExtractedContent
```

---

### Step 2: PaddleOCR Wrapper

**File: `app/services/file_processing/ocr/paddle_ocr.py`**

```python
"""PaddleOCR wrapper for table extraction."""

import time
from pathlib import Path

from paddleocr import PaddleOCR, PPStructure
from PIL import Image
import numpy as np

from app.services.file_processing.models import (
    ExtractedContent,
    ExtractionMetadata,
    ExtractionMethod,
    TableData,
)


class PaddleOCREngine:
    """Wrapper for PaddleOCR with table structure recognition."""

    _instance = None
    _ocr = None
    _table_engine = None

    def __new__(cls):
        """Singleton pattern for model caching."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def _load_models(self):
        """Lazy load OCR models."""
        if self._ocr is None:
            # Standard OCR
            self._ocr = PaddleOCR(
                use_angle_cls=True,
                lang="en",
                show_log=False,
            )

            # Table structure recognition (PP-Structure)
            self._table_engine = PPStructure(
                show_log=False,
                layout=True,
                table=True,
            )

    async def extract_text(self, image_path: Path) -> tuple[str, float]:
        """Extract text from image using PaddleOCR."""
        self._load_models()

        img = np.array(Image.open(image_path))
        result = self._ocr.ocr(img, cls=True)

        text_lines = []
        confidences = []

        for line in result[0] or []:
            text = line[1][0]
            confidence = line[1][1]
            text_lines.append(text)
            confidences.append(confidence)

        text = "\n".join(text_lines)
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

        return text, avg_confidence

    async def extract_tables(self, image_path: Path) -> tuple[str, list[TableData], float]:
        """Extract tables from image using PP-Structure."""
        self._load_models()

        img = np.array(Image.open(image_path))
        result = self._table_engine(img)

        tables = []
        all_text = []
        confidences = []

        for item in result:
            if item["type"] == "table":
                table_data = self._parse_table(item)
                if table_data:
                    tables.append(table_data)
                    all_text.append(self._table_to_text(table_data))
                    confidences.append(table_data.confidence)

            elif item["type"] == "text":
                if "res" in item:
                    for line in item["res"]:
                        text = line["text"]
                        conf = line.get("confidence", 0.9)
                        all_text.append(text)
                        confidences.append(conf)

        text = "\n\n".join(all_text)
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

        return text, tables, avg_confidence

    def _parse_table(self, table_item: dict) -> TableData | None:
        """Parse PP-Structure table result into TableData."""
        try:
            html = table_item.get("res", {}).get("html", "")
            if not html:
                return None

            import re

            rows_match = re.findall(r"<tr>(.*?)</tr>", html, re.DOTALL)
            if not rows_match:
                return None

            parsed_rows = []
            for row_html in rows_match:
                cells = re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", row_html, re.DOTALL)
                cells = [re.sub(r"<[^>]+>", "", cell).strip() for cell in cells]
                parsed_rows.append(cells)

            if not parsed_rows:
                return None

            headers = parsed_rows[0] if parsed_rows else []
            data_rows = parsed_rows[1:] if len(parsed_rows) > 1 else []

            return TableData(
                headers=headers,
                rows=data_rows,
                confidence=table_item.get("score", 0.9),
            )
        except Exception:
            return None

    def _table_to_text(self, table: TableData) -> str:
        """Convert TableData to plain text."""
        lines = []
        lines.append(" | ".join(table.headers))
        lines.append("-" * 40)
        for row in table.rows:
            lines.append(" | ".join(row))
        return "\n".join(lines)

    async def extract_from_image(self, image_path: Path) -> ExtractedContent:
        """Full extraction with table structure."""
        start_time = time.perf_counter()

        text, tables, confidence = await self.extract_tables(image_path)
        markdown = self._build_markdown(text, tables)

        processing_time = int((time.perf_counter() - start_time) * 1000)

        return ExtractedContent(
            text=text,
            tables=tables if tables else None,
            structured_data={
                "table_count": len(tables),
                "has_tables": len(tables) > 0,
            },
            markdown=markdown,
            metadata=ExtractionMetadata(
                file_type=image_path.suffix.lower(),
                file_size=image_path.stat().st_size,
                extraction_method=ExtractionMethod.PADDLE_OCR,
                confidence=confidence,
                processing_time_ms=processing_time,
                ocr_engines_used=["paddle_ocr"],
            ),
        )

    def _build_markdown(self, text: str, tables: list[TableData]) -> str:
        """Build markdown with tables."""
        parts = []

        if text:
            parts.append(text)

        for i, table in enumerate(tables):
            parts.append(f"\n### Table {i + 1}")
            parts.append(self._table_to_markdown(table))

        return "\n\n".join(parts)

    def _table_to_markdown(self, table: TableData) -> str:
        """Convert TableData to markdown table."""
        lines = []
        lines.append("| " + " | ".join(table.headers) + " |")
        lines.append("| " + " | ".join(["---"] * len(table.headers)) + " |")

        for row in table.rows:
            padded_row = row + [""] * (len(table.headers) - len(row))
            lines.append("| " + " | ".join(padded_row) + " |")

        return "\n".join(lines)


# Singleton instance
paddle_ocr = PaddleOCREngine()
```

---

### Step 3: Surya OCR Wrapper

**File: `app/services/file_processing/ocr/surya_ocr.py`**

```python
"""Surya OCR wrapper for document text extraction."""

import time
from pathlib import Path

from PIL import Image
from surya.ocr import run_ocr
from surya.model.detection.model import load_model as load_det_model
from surya.model.detection.processor import load_processor as load_det_processor
from surya.model.recognition.model import load_model as load_rec_model
from surya.model.recognition.processor import load_processor as load_rec_processor

from app.services.file_processing.models import (
    ExtractedContent,
    ExtractionMetadata,
    ExtractionMethod,
)


class SuryaOCR:
    """Wrapper for Surya OCR engine."""

    _instance = None
    _det_model = None
    _det_processor = None
    _rec_model = None
    _rec_processor = None

    def __new__(cls):
        """Singleton pattern for model caching."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def _load_models(self):
        """Lazy load OCR models."""
        if self._det_model is None:
            self._det_processor = load_det_processor()
            self._det_model = load_det_model()
            self._rec_processor = load_rec_processor()
            self._rec_model = load_rec_model()

    async def extract_from_image(
        self,
        image_path: Path,
        languages: list[str] | None = None
    ) -> ExtractedContent:
        """Extract text from an image using Surya OCR."""
        start_time = time.perf_counter()

        self._load_models()

        image = Image.open(image_path)

        if languages is None:
            languages = ["en"]

        results = run_ocr(
            [image],
            [languages],
            self._det_model,
            self._det_processor,
            self._rec_model,
            self._rec_processor,
        )

        text_lines = []
        confidence_scores = []

        for page_result in results:
            for line in page_result.text_lines:
                text_lines.append(line.text)
                confidence_scores.append(line.confidence)

        text = "\n".join(text_lines)
        avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.0

        processing_time = int((time.perf_counter() - start_time) * 1000)

        return ExtractedContent(
            text=text,
            structured_data={"line_count": len(text_lines)},
            markdown=f"```\n{text}\n```",
            metadata=ExtractionMetadata(
                file_type=image_path.suffix.lower(),
                file_size=image_path.stat().st_size,
                extraction_method=ExtractionMethod.SURYA_OCR,
                confidence=avg_confidence,
                processing_time_ms=processing_time,
                ocr_engines_used=["surya_ocr"],
            ),
        )

    async def extract_from_images(
        self,
        images: list[Image.Image],
        languages: list[str] | None = None,
    ) -> tuple[str, float]:
        """Extract text from multiple images (e.g., PDF pages)."""
        self._load_models()

        if languages is None:
            languages = ["en"]

        results = run_ocr(
            images,
            [languages] * len(images),
            self._det_model,
            self._det_processor,
            self._rec_model,
            self._rec_processor,
        )

        all_text = []
        all_confidence = []

        for page_idx, page_result in enumerate(results):
            page_text = []
            for line in page_result.text_lines:
                page_text.append(line.text)
                all_confidence.append(line.confidence)

            all_text.append(f"--- Page {page_idx + 1} ---\n" + "\n".join(page_text))

        text = "\n\n".join(all_text)
        avg_confidence = sum(all_confidence) / len(all_confidence) if all_confidence else 0.0

        return text, avg_confidence


# Singleton instance
surya_ocr = SuryaOCR()
```

---

### Step 4: Table Detector

**File: `app/services/file_processing/classifiers/table_detector.py`**

```python
"""Detector for tables in images."""

from pathlib import Path

import numpy as np
from PIL import Image


class TableDetector:
    """Detect if an image contains tables."""

    def has_tables(self, image_path: Path) -> bool:
        """Check if image likely contains tables."""
        img = Image.open(image_path).convert("L")  # Grayscale
        img_array = np.array(img)

        # Detect horizontal and vertical lines
        h_lines = self._detect_horizontal_lines(img_array)
        v_lines = self._detect_vertical_lines(img_array)

        # Tables typically have multiple intersecting lines
        return h_lines >= 3 and v_lines >= 2

    def _detect_horizontal_lines(self, img: np.ndarray) -> int:
        """Count horizontal lines in image."""
        h_diff = np.abs(np.diff(img.astype(np.int16), axis=0))
        h_edges = np.sum(h_diff > 100, axis=1)
        line_rows = np.sum(h_edges > img.shape[1] * 0.3)
        return line_rows

    def _detect_vertical_lines(self, img: np.ndarray) -> int:
        """Count vertical lines in image."""
        v_diff = np.abs(np.diff(img.astype(np.int16), axis=1))
        v_edges = np.sum(v_diff > 100, axis=0)
        line_cols = np.sum(v_edges > img.shape[0] * 0.3)
        return line_cols


# Singleton instance
table_detector = TableDetector()
```

---

### Step 5: OCR Router (FULLY_LOCAL)

**File: `app/services/file_processing/ocr/ocr_router.py`**

```python
"""Routes images to the best local OCR engine."""

import time
from pathlib import Path

from app.services.file_processing.models import ExtractedContent, ImageType
from app.services.file_processing.ocr.paddle_ocr import paddle_ocr
from app.services.file_processing.ocr.surya_ocr import surya_ocr
from app.services.file_processing.classifiers.table_detector import table_detector


class OCRRouter:
    """Routes images to optimal local OCR engine."""

    CONFIDENCE_THRESHOLD = 0.55

    async def process(self, image_path: Path) -> ExtractedContent:
        """Process image with optimal local OCR selection."""
        start_time = time.perf_counter()

        # Check for tables
        has_tables = table_detector.has_tables(image_path)

        if has_tables:
            # PaddleOCR is best for tables
            result = await paddle_ocr.extract_from_image(image_path)
            primary_engine = "paddle_ocr"
            fallback_engine = surya_ocr
        else:
            # Surya is best for general documents
            result = await surya_ocr.extract_from_image(image_path)
            primary_engine = "surya_ocr"
            fallback_engine = paddle_ocr

        # Check confidence for fallback
        if result.metadata.confidence < self.CONFIDENCE_THRESHOLD:
            # Try the other engine
            alt_result = await fallback_engine.extract_from_image(image_path)

            if alt_result.metadata.confidence > result.metadata.confidence:
                result = alt_result
                result.metadata.ocr_engines_used.append(primary_engine)

        # Update total processing time
        result.metadata.processing_time_ms = int((time.perf_counter() - start_time) * 1000)

        return result

    async def process_pdf_pages(
        self,
        images: list,
    ) -> tuple[str, list, float]:
        """Process multiple PDF pages with optimal routing."""
        import tempfile
        from PIL import Image as PILImage

        all_text = []
        all_tables = []
        all_confidence = []

        for page_idx, image in enumerate(images):
            # Save temporarily for processing
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
                image.save(f.name)
                temp_path = Path(f.name)

            try:
                result = await self.process(temp_path)
                all_text.append(f"--- Page {page_idx + 1} ---\n{result.text}")

                if result.tables:
                    all_tables.extend(result.tables)

                all_confidence.append(result.metadata.confidence)
            finally:
                temp_path.unlink()

        text = "\n\n".join(all_text)
        avg_confidence = sum(all_confidence) / len(all_confidence) if all_confidence else 0.0

        return text, all_tables, avg_confidence


# Singleton instance
ocr_router = OCRRouter()
```

---

### Step 6: Image Handler (Updated)

**File: `app/services/file_processing/image_handler.py`**

```python
"""Handler for image files with multi-OCR routing."""

import time
from pathlib import Path

from PIL import Image

from app.services.file_processing.base_handler import BaseFileHandler
from app.services.file_processing.models import ExtractedContent
from app.services.file_processing.ocr.ocr_router import ocr_router


class ImageHandler(BaseFileHandler):
    """Handler for image files with smart OCR routing."""

    SUPPORTED_EXTENSIONS = [".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff", ".webp"]
    MAX_DIMENSION = 4096

    async def extract(self, file_path: Path) -> ExtractedContent:
        """Extract content from image using optimal OCR."""
        start_time = time.perf_counter()

        # Resize if needed
        self._resize_if_needed(file_path)

        # Use OCR router for optimal selection
        result = await ocr_router.process(file_path)

        # Update total processing time
        total_time = int((time.perf_counter() - start_time) * 1000)
        result.metadata.processing_time_ms = total_time

        return result

    def _resize_if_needed(self, file_path: Path) -> None:
        """Resize image if it exceeds maximum dimensions."""
        img = Image.open(file_path)

        if img.width > self.MAX_DIMENSION or img.height > self.MAX_DIMENSION:
            ratio = min(self.MAX_DIMENSION / img.width, self.MAX_DIMENSION / img.height)
            new_size = (int(img.width * ratio), int(img.height * ratio))

            img = img.resize(new_size, Image.Resampling.LANCZOS)
            img.save(file_path)
```

---

### Step 7: PDF Handler (Updated)

**File: `app/services/file_processing/pdf_handler.py`**

```python
"""Handler for PDF files with multi-OCR support."""

import time
from pathlib import Path

import fitz  # PyMuPDF
from PIL import Image

from app.services.file_processing.base_handler import BaseFileHandler
from app.services.file_processing.models import (
    ExtractedContent,
    ExtractionMetadata,
    ExtractionMethod,
)
from app.services.file_processing.ocr.ocr_router import ocr_router


class PDFHandler(BaseFileHandler):
    """Handler for PDF documents with smart OCR selection."""

    SUPPORTED_EXTENSIONS = [".pdf"]
    MIN_TEXT_LENGTH = 50

    async def extract(self, file_path: Path) -> ExtractedContent:
        """Extract content from PDF file."""
        start_time = time.perf_counter()

        doc = fitz.open(str(file_path))
        page_count = len(doc)

        # Try native text extraction first
        native_text = self._extract_native_text(doc)

        if self._has_sufficient_text(native_text, page_count):
            doc.close()
            processing_time = int((time.perf_counter() - start_time) * 1000)

            return ExtractedContent(
                text=native_text,
                structured_data={"pages": page_count},
                markdown=self._text_to_markdown(native_text),
                metadata=ExtractionMetadata(
                    file_type=".pdf",
                    file_size=file_path.stat().st_size,
                    pages=page_count,
                    extraction_method=ExtractionMethod.PYMUPDF,
                    confidence=1.0,
                    processing_time_ms=processing_time,
                    ocr_engines_used=["pymupdf"],
                ),
            )

        # Scanned PDF - use multi-OCR router
        images = self._pdf_to_images(doc)
        doc.close()

        text, tables, confidence = await ocr_router.process_pdf_pages(images)

        processing_time = int((time.perf_counter() - start_time) * 1000)

        # Build markdown with tables
        markdown = self._build_markdown(text, tables)

        return ExtractedContent(
            text=text,
            tables=tables if tables else None,
            structured_data={
                "pages": page_count,
                "ocr_used": True,
                "table_count": len(tables) if tables else 0,
            },
            markdown=markdown,
            metadata=ExtractionMetadata(
                file_type=".pdf",
                file_size=file_path.stat().st_size,
                pages=page_count,
                extraction_method=ExtractionMethod.PADDLE_OCR,
                confidence=confidence,
                processing_time_ms=processing_time,
            ),
        )

    def _extract_native_text(self, doc: fitz.Document) -> str:
        """Extract text from PDF using PyMuPDF."""
        pages_text = []
        for page_num, page in enumerate(doc):
            text = page.get_text().strip()
            if text:
                pages_text.append(f"--- Page {page_num + 1} ---\n{text}")
        return "\n\n".join(pages_text)

    def _has_sufficient_text(self, text: str, page_count: int) -> bool:
        """Check if extracted text is sufficient."""
        if not text:
            return False
        return len(text) >= (self.MIN_TEXT_LENGTH * page_count)

    def _pdf_to_images(self, doc: fitz.Document, dpi: int = 200) -> list[Image.Image]:
        """Convert PDF pages to PIL Images."""
        images = []
        zoom = dpi / 72
        matrix = fitz.Matrix(zoom, zoom)

        for page in doc:
            pix = page.get_pixmap(matrix=matrix)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            images.append(img)

        return images

    def _build_markdown(self, text: str, tables: list) -> str:
        """Build markdown with extracted tables."""
        parts = [self._text_to_markdown(text)]

        if tables:
            parts.append("\n## Extracted Tables\n")
            for i, table in enumerate(tables):
                parts.append(f"### Table {i + 1}")
                parts.append(self._table_to_markdown(table))

        return "\n\n".join(parts)

    def _table_to_markdown(self, table) -> str:
        """Convert table to markdown."""
        lines = []
        lines.append("| " + " | ".join(table.headers) + " |")
        lines.append("| " + " | ".join(["---"] * len(table.headers)) + " |")
        for row in table.rows:
            padded = row + [""] * (len(table.headers) - len(row))
            lines.append("| " + " | ".join(padded) + " |")
        return "\n".join(lines)
```

---

## Testing

```bash
# Test text file
curl -X POST "http://localhost:8000/api/v1/upload" \
  -F "file=@test.txt"

# Test PDF
curl -X POST "http://localhost:8000/api/v1/upload" \
  -F "file=@invoice.pdf"

# Test Excel
curl -X POST "http://localhost:8000/api/v1/upload" \
  -F "file=@data.xlsx"

# Test image with tables
curl -X POST "http://localhost:8000/api/v1/upload" \
  -F "file=@table.png"
```

---

## Summary

| Component | OCR Engine | Cost |
|-----------|-----------|------|
| Text files | Direct read | **$0.00** |
| DOCX | python-docx | **$0.00** |
| Excel | openpyxl | **$0.00** |
| PDF (native) | PyMuPDF | **$0.00** |
| PDF (scanned - tables) | PaddleOCR | **$0.00** |
| PDF (scanned - general) | Surya OCR | **$0.00** |
| Images (tables) | PaddleOCR | **$0.00** |
| Images (documents) | Surya OCR | **$0.00** |
| **Total API cost** | | **$0.00** |
