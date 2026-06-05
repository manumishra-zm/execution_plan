# LIBRARY_GEMINI - Python Libraries + Gemini Fallback

## Overview

**LIBRARY_GEMINI** is a cost-efficient OCR approach that uses **Python libraries for text-based files** and only falls back to **Gemini Vision for images and scanned PDFs**.

### Key Characteristics

| Feature | Value |
|---------|-------|
| **Primary Method** | Python libraries (FREE) |
| **Fallback** | Gemini Vision API |
| **Cost** | Very Low (~$0 for 80% of docs) |
| **Speed** | Fast (libraries) / Medium (Gemini) |
| **Best For** | Text-heavy documents, cost optimization |

---

## Architecture

### High-Level Flow

```mermaid
flowchart TD
    A[File Input] --> B{File Extension?}

    B -->|.txt .md .tex| C[Direct Text Read]
    B -->|.csv| D[CSV Parser]
    B -->|.docx| E[python-docx]
    B -->|.xlsx .xls| F[pandas + openpyxl]
    B -->|.pdf| G{PDF Type Detection}
    B -->|Images| H[Gemini Vision]

    G --> I[Extract Text with PyMuPDF]
    I --> J{Avg chars/page >= 50?}

    J -->|Yes: Native PDF| K[Return PyMuPDF Text]
    J -->|No: Scanned PDF| L[Gemini Vision Fallback]

    C --> M[ExtractedContent]
    D --> M
    E --> M
    F --> M
    K --> M
    L --> M
    H --> M

    style C fill:#2D5016,color:#fff
    style D fill:#2D5016,color:#fff
    style E fill:#2D5016,color:#fff
    style F fill:#2D5016,color:#fff
    style K fill:#2D5016,color:#fff
    style H fill:#8B0000,color:#fff
    style L fill:#8B0000,color:#fff
```

---

## File Type Handling

### Processing Matrix

```mermaid
flowchart LR
    subgraph Free["FREE - Python Libraries"]
        TXT[".txt .md .tex"]
        CSV[".csv"]
        DOCX[".docx"]
        XLSX[".xlsx .xls"]
        PDF_N["PDF (Native)"]
    end

    subgraph Paid["PAID - Gemini API"]
        IMG["Images"]
        PDF_S["PDF (Scanned)"]
    end

    TXT --> R1["Direct Read"]
    CSV --> R2["csv.reader"]
    DOCX --> R3["python-docx"]
    XLSX --> R4["pandas"]
    PDF_N --> R5["PyMuPDF"]

    IMG --> R6["Gemini Vision"]
    PDF_S --> R6

    style Free fill:#2D5016,color:#fff
    style Paid fill:#8B0000,color:#fff
```

### Detailed File Handling

| Extension | Library | Method | Cost |
|-----------|---------|--------|------|
| `.txt`, `.md`, `.tex` | Built-in | `open().read()` | FREE |
| `.csv` | Built-in | `csv.reader()` | FREE |
| `.docx` | python-docx | `Document()` | FREE |
| `.xlsx`, `.xls` | pandas | `pd.read_excel()` | FREE |
| `.pdf` (native) | PyMuPDF | `fitz.open()` | FREE |
| `.pdf` (scanned) | Gemini | `gemini_multi_model` | ~$0.0001/page |
| Images | Gemini | `gemini_multi_model` | ~$0.0001/image |

---

## Scanned PDF Detection

### Detection Algorithm

```mermaid
flowchart TD
    A[PDF File] --> B[Open with PyMuPDF]
    B --> C[Extract All Text]
    C --> D[Count Total Characters]
    D --> E[Count Total Pages]

    E --> F["Calculate: avg_chars = total_chars / pages"]

    F --> G{avg_chars >= 50?}

    G -->|Yes| H["Native PDF<br/>(Has Text Layer)"]
    G -->|No| I["Scanned PDF<br/>(Image-based)"]

    H --> J[Use PyMuPDF Text]
    I --> K[Trigger Gemini Fallback]

    style H fill:#2D5016,color:#fff
    style I fill:#8B0000,color:#fff
```

### Detection Logic

```python
MIN_PDF_TEXT_PER_PAGE = 50  # characters

def is_scanned_pdf(pdf_path: Path) -> bool:
    doc = fitz.open(str(pdf_path))
    total_chars = 0

    for page in doc:
        text = page.get_text()
        total_chars += len(text.strip())

    doc.close()

    avg_chars_per_page = total_chars / len(doc)

    # If less than 50 chars per page on average = scanned
    return avg_chars_per_page < MIN_PDF_TEXT_PER_PAGE
```

### Example Scenarios

| PDF Content | Total Chars | Pages | Avg/Page | Classification |
|-------------|-------------|-------|----------|----------------|
| Full text document | 5000 | 10 | 500 | **Native** |
| Scanned invoice | 0 | 1 | 0 | **Scanned** |
| Mixed (some OCR'd) | 200 | 5 | 40 | **Scanned** |
| Minimal text | 100 | 1 | 100 | **Native** |

---

## Text File Extraction

### Direct Read Flow

```mermaid
sequenceDiagram
    participant Client
    participant Processor as LibraryGeminiProcessor
    participant FileSystem as File System

    Client->>Processor: process(text_file.txt)
    Processor->>Processor: Check extension

    Processor->>FileSystem: Try UTF-8 encoding
    alt Success
        FileSystem-->>Processor: Text content
    else UnicodeDecodeError
        Processor->>FileSystem: Try latin-1 encoding
        FileSystem-->>Processor: Text content
    end

    Processor->>Processor: Build ExtractedContent
    Processor-->>Client: ExtractedContent
```

### Encoding Fallback

```python
ENCODING_FALLBACK = ["utf-8", "latin-1", "cp1252"]

def extract_text_file(path: Path) -> str:
    for encoding in ENCODING_FALLBACK:
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Cannot decode file with any encoding")
```

---

## CSV Extraction

### CSV Processing Flow

```mermaid
flowchart TD
    A[CSV File] --> B[Detect Delimiter]
    B --> C{Delimiter?}

    C -->|Comma| D["delimiter=','"]
    C -->|Tab| E["delimiter='\\t'"]
    C -->|Semicolon| F["delimiter=';'"]

    D --> G[Parse with csv.reader]
    E --> G
    F --> G

    G --> H[Extract Headers]
    H --> I[Extract Rows]
    I --> J[Build TableData]
    J --> K[Convert to Markdown]
    K --> L[ExtractedContent]

    style G fill:#2D5016,color:#fff
```

---

## DOCX Extraction

### Word Document Processing

```mermaid
flowchart TD
    A[DOCX File] --> B[Open with python-docx]
    B --> C[Extract Paragraphs]
    C --> D[Extract Tables]

    D --> E{Has Tables?}
    E -->|Yes| F[Parse Table Headers/Rows]
    E -->|No| G[Text Only]

    F --> H[Build TableData List]
    G --> I[Join Paragraphs]
    H --> I

    I --> J[Generate Markdown]
    J --> K[ExtractedContent]

    style B fill:#2D5016,color:#fff
```

---

## Excel Extraction

### Spreadsheet Processing

```mermaid
flowchart TD
    A[Excel File] --> B{Extension?}

    B -->|.xlsx| C[openpyxl engine]
    B -->|.xls| D[xlrd engine]

    C --> E[pd.read_excel]
    D --> E

    E --> F[Get Sheet Names]
    F --> G[Read All Sheets]

    G --> H[Loop: Each Sheet]
    H --> I[DataFrame to Text]
    I --> J[DataFrame to Table]

    J --> K{More Sheets?}
    K -->|Yes| H
    K -->|No| L[Combine All Sheets]

    L --> M[ExtractedContent]

    style E fill:#2D5016,color:#fff
```

---

## PDF Native Extraction

### PyMuPDF Processing

```mermaid
flowchart TD
    A[PDF File] --> B[Open with PyMuPDF]
    B --> C[Check Encryption]

    C --> D{Encrypted?}
    D -->|Yes| E[Raise Error]
    D -->|No| F[Loop: Each Page]

    F --> G[Extract Text Blocks]
    G --> H[Get Bounding Boxes]
    H --> I[Build Page Layout]

    I --> J{More Pages?}
    J -->|Yes| F
    J -->|No| K[Combine Pages]

    K --> L[Build Document Layout]
    L --> M[ExtractedContent]

    style B fill:#2D5016,color:#fff
    style E fill:#8B0000,color:#fff
```

---

## Gemini Fallback Flow

### When Gemini is Used

```mermaid
flowchart TD
    A[File Input] --> B{Needs Gemini?}

    B -->|Image File| C[Yes - Direct Gemini]
    B -->|Scanned PDF| D[Yes - Gemini Fallback]
    B -->|Native PDF| E[No - PyMuPDF]
    B -->|Text Files| F[No - Libraries]

    C --> G[GeminiMultiModel]
    D --> G

    G --> H[4-Model Fallback Chain]
    H --> I[ExtractedContent]

    E --> J[Library Extraction]
    F --> J
    J --> I

    style C fill:#8B0000,color:#fff
    style D fill:#8B0000,color:#fff
    style E fill:#2D5016,color:#fff
    style F fill:#2D5016,color:#fff
```

### Gemini Fallback Trigger

```mermaid
sequenceDiagram
    participant Processor as LibraryGeminiProcessor
    participant PyMuPDF
    participant Gemini as GeminiMultiModel

    Processor->>PyMuPDF: Extract text from PDF
    PyMuPDF-->>Processor: Text (maybe empty)

    Processor->>Processor: Calculate avg chars/page

    alt avg >= 50 (Native PDF)
        Processor->>Processor: Use PyMuPDF text
        Processor-->>Processor: ExtractedContent
    else avg < 50 (Scanned PDF)
        Processor->>Processor: Raise ValueError("Scanned PDF")
        Note over Processor: Caller catches and uses Gemini
        Processor->>Gemini: process_pdf(path)
        Gemini-->>Processor: ExtractedContent
    end
```

---

## Configuration

### Supported Extensions

```python
SUPPORTED_EXTENSIONS = {
    # Text files (direct read)
    ".txt", ".md", ".tex",

    # CSV
    ".csv",

    # Office documents
    ".docx",
    ".xlsx", ".xls",

    # PDF (native + scanned detection)
    ".pdf",

    # Images (always Gemini)
    ".png", ".jpg", ".jpeg", ".gif",
    ".bmp", ".tiff", ".webp"
}
```

### Detection Threshold

```python
MIN_PDF_TEXT_PER_PAGE = 50  # characters

# If average chars per page < 50, treat as scanned
```

---

## Output Structure

### Native Extraction (Libraries)

```json
{
  "text": "Document content...",
  "structured_data": {
    "library_gemini": true,
    "approach": "library_gemini",
    "extraction_type": "native",
    "library_used": "python-docx"
  },
  "metadata": {
    "extraction_method": "python_docx",
    "confidence": 1.0,
    "tokens_used": 0
  }
}
```

### Gemini Fallback

```json
{
  "text": "OCR extracted text...",
  "structured_data": {
    "library_gemini": true,
    "approach": "library_gemini",
    "extraction_type": "gemini_fallback",
    "reason": "Scanned PDF detected (avg 0 chars/page)"
  },
  "metadata": {
    "extraction_method": "gemini_vision",
    "confidence": 0.89,
    "tokens_used": 1500
  }
}
```

---

## Cost Analysis

### Cost Breakdown

```mermaid
pie title Cost Distribution (100 uploads)
    "Native Files (FREE)" : 60
    "Native PDFs (FREE)" : 20
    "Scanned PDFs (Gemini)" : 10
    "Images (Gemini)" : 10
```

### Per-Type Cost

| File Type | Processing | Cost |
|-----------|------------|------|
| .docx, .xlsx, .csv, .txt | Libraries | **$0.00** |
| Native PDF | PyMuPDF | **$0.00** |
| Scanned PDF | Gemini | ~$0.0001/page |
| Images | Gemini | ~$0.0001/image |

### Monthly Projection

| Daily Volume | % Native | % Gemini | Monthly Cost |
|--------------|----------|----------|--------------|
| 100 docs | 80% | 20% | ~$0.06 |
| 500 docs | 80% | 20% | ~$0.30 |
| 1000 docs | 80% | 20% | ~$0.60 |

---

## Comparison with Other Approaches

```mermaid
flowchart LR
    subgraph Approaches["OCR Approaches"]
        A["FULLY_LOCAL<br/>100% Free<br/>Local OCR only"]
        B["LIBRARY_GEMINI<br/>~80% Free<br/>Libraries + Gemini"]
        C["GEMINI_COMPLETE<br/>100% Gemini<br/>Best quality"]
    end

    A -->|"Lower accuracy<br/>for scanned docs"| X[Trade-offs]
    B -->|"Best cost/quality<br/>balance"| X
    C -->|"Highest cost<br/>best accuracy"| X

    style B fill:#2D5016,color:#fff
```

---

## When to Use LIBRARY_GEMINI

### Best Use Cases

| Scenario | Why Use |
|----------|---------|
| Text-heavy documents | FREE library extraction |
| Office files (.docx, .xlsx) | Perfect native parsing |
| Native PDFs | PyMuPDF extracts perfectly |
| Cost-sensitive workflows | Minimal API usage |
| High volume processing | 80%+ documents are FREE |

### When NOT to Use

| Scenario | Better Alternative |
|----------|-------------------|
| All images/scans | GEMINI_COMPLETE |
| Complex layouts | GEMINI_COMPLETE |
| Mission-critical | ULTIMATE_CASCADE |

---

## File Location

**Implementation:** `app/services/file_processing/ocr/library_gemini.py`

**Singleton Instance:** `library_gemini`

```python
from app.services.file_processing.ocr import library_gemini

# Process any supported file
result = await library_gemini.process(Path("document.docx"))
result = await library_gemini.process(Path("report.xlsx"))
result = await library_gemini.process(Path("invoice.pdf"))

# Check if Gemini fallback is available
if library_gemini.is_available():
    # Can handle scanned PDFs and images
    pass
```

---

## Error Handling

### Error Flow

```mermaid
flowchart TD
    A[Process File] --> B{Extension Supported?}

    B -->|No| C[ValueError: Unsupported]
    B -->|Yes| D{File Type}

    D -->|Text/Office| E[Try Library]
    D -->|PDF| F[Try PyMuPDF]
    D -->|Image| G[Use Gemini]

    E --> H{Success?}
    F --> I{Native or Scanned?}
    G --> J{Gemini Available?}

    H -->|No| K[Return Error]
    I -->|Native| L[Return Result]
    I -->|Scanned| M[Gemini Fallback]

    J -->|No| N[RuntimeError]
    J -->|Yes| O[Process with Gemini]

    M --> J
    O --> L

    style C fill:#8B0000,color:#fff
    style K fill:#8B0000,color:#fff
    style N fill:#8B0000,color:#fff
```
