# GEMINI_COMPLETE - 100% Gemini Vision OCR Approach

## Overview

**GEMINI_COMPLETE** is an OCR approach that uses **only Gemini Vision API** for all text extraction. No local OCR engines are used.

### Key Characteristics

| Feature | Value |
|---------|-------|
| **Local OCR** | None |
| **API Usage** | 100% Gemini |
| **Cost** | ~$0.0001/page |
| **Speed** | Medium (API latency) |
| **Accuracy** | Highest (especially for complex docs) |
| **Best For** | Graphs, charts, complex layouts |

---

## Architecture

### High-Level Flow

```mermaid
flowchart TD
    A[File Input] --> B{File Type?}

    B -->|Image| C[Direct Gemini Vision]
    B -->|PDF| D[Convert to Images]

    D --> E[Page-by-Page Processing]
    E --> F[Gemini Vision per Page]

    C --> G[Model Fallback Chain]
    F --> G

    G --> H{Model 1: gemini-2.0-flash-exp}
    H -->|Success| I[Return Result]
    H -->|Fail| J{Model 2: gemini-2.0-flash}

    J -->|Success| I
    J -->|Fail| K{Model 3: gemini-1.5-pro}

    K -->|Success| I
    K -->|Fail| L{Model 4: gemini-1.5-flash}

    L -->|Success| I
    L -->|Fail| M[All Models Failed]

    I --> N[ExtractedContent]

    style C fill:#8B0000,color:#fff
    style F fill:#8B0000,color:#fff
    style G fill:#B35900,color:#fff
```

---

## Model Fallback Chain

### 4-Model Cascade

```mermaid
flowchart LR
    subgraph Chain["Gemini Model Fallback Chain"]
        M1["gemini-2.0-flash-exp<br/>(Experimental)"]
        M2["gemini-2.0-flash<br/>(Default)"]
        M3["gemini-1.5-pro<br/>(High Quality)"]
        M4["gemini-1.5-flash<br/>(Fast Fallback)"]

        M1 -->|Fail| M2
        M2 -->|Fail| M3
        M3 -->|Fail| M4
    end

    style M1 fill:#2D5016,color:#fff
    style M2 fill:#1E3A5F,color:#fff
    style M3 fill:#B35900,color:#fff
    style M4 fill:#8B0000,color:#fff
```

### Model Specifications

| Model | Timeout | Max Tokens | Use Case |
|-------|---------|------------|----------|
| `gemini-2.0-flash-exp` | 60s | 8192 | Experimental, most capable |
| `gemini-2.0-flash` | 60s | 8192 | Production default |
| `gemini-1.5-pro` | 90s | 8192 | Complex documents |
| `gemini-1.5-flash` | 45s | 4096 | Fast fallback |

---

## Image Processing Flow

### Direct Image Extraction

```mermaid
sequenceDiagram
    participant Client
    participant Processor as GeminiCompleteProcessor
    participant MultiModel as GeminiMultiModel
    participant API as Gemini API

    Client->>Processor: process_image(path)
    Processor->>Processor: Detect image type
    Processor->>Processor: Preprocess if needed

    alt Image > 4096px
        Processor->>Processor: Resize to max 4096px
    end

    Processor->>MultiModel: extract_from_image(path, type)

    loop Model Fallback
        MultiModel->>API: Call with current model
        alt Success
            API-->>MultiModel: Extraction result
            MultiModel-->>Processor: Result + model_used
        else Rate Limited
            MultiModel->>MultiModel: Wait (exponential backoff)
            MultiModel->>API: Retry
        else Model Failed
            MultiModel->>MultiModel: Try next model
        end
    end

    Processor->>Processor: Build ExtractedContent
    Processor-->>Client: ExtractedContent
```

### Image Type Detection

```mermaid
flowchart TD
    A[Image Path] --> B{Filename Contains?}

    B -->|invoice, bill, receipt| C[ImageType.INVOICE]
    B -->|chart, graph, plot| D[ImageType.GRAPH]
    B -->|table, spreadsheet| E[ImageType.TABLE]
    B -->|form, application| F[ImageType.FORM]
    B -->|None of above| G[ImageType.DOCUMENT]

    C --> H[Build Invoice Prompt]
    D --> I[Build Graph Prompt]
    E --> J[Build Table Prompt]
    F --> K[Build Form Prompt]
    G --> L[Build Document Prompt]

    style C fill:#B35900,color:#fff
    style D fill:#8B0000,color:#fff
    style E fill:#1E3A5F,color:#fff
```

---

## PDF Processing Flow

### Page-by-Page Extraction

```mermaid
flowchart TD
    A[PDF File] --> B[Open with PyMuPDF]
    B --> C[Get Page Count]

    C --> D{max_pdf_pages > 0?}
    D -->|Yes| E[Limit to max_pdf_pages]
    D -->|No| F[Process ALL pages]

    E --> G[Loop: Each Page]
    F --> G

    G --> H[Convert Page to Image]
    H --> I[Set DPI: 200]
    I --> J[Create PIL Image]
    J --> K[Save as Temp PNG]

    K --> L[Gemini Vision Extract]
    L --> M{Success?}

    M -->|Yes| N[Collect Page Result]
    M -->|No| O[Retry up to 2 times]
    O -->|Still Fail| P[Mark as Error]

    N --> Q{More Pages?}
    P --> Q
    Q -->|Yes| G
    Q -->|No| R[Combine All Pages]

    R --> S[Calculate Avg Confidence]
    S --> T[Build ExtractedContent]

    style L fill:#8B0000,color:#fff
    style H fill:#1E3A5F,color:#fff
```

### PDF Page Processing Detail

```mermaid
sequenceDiagram
    participant Handler as PDFHandler
    participant Processor as GeminiCompleteProcessor
    participant Gemini as GeminiMultiModel
    participant TempFile as Temp Storage

    Handler->>Processor: process_pdf(path)
    Processor->>Processor: Open PDF (PyMuPDF)

    loop Each Page (0 to N)
        Processor->>Processor: Get page pixmap (200 DPI)
        Processor->>TempFile: Save as temp PNG

        loop Retry (max 2)
            Processor->>Gemini: extract_with_fallback(temp_path)
            alt Success
                Gemini-->>Processor: (text, confidence, model, tokens)
            else Fail
                Processor->>Processor: Wait 1s, retry
            end
        end

        Processor->>TempFile: Delete temp file
        Processor->>Processor: Append to results
    end

    Processor->>Processor: Combine page texts
    Processor->>Processor: Average confidence
    Processor-->>Handler: ExtractedContent
```

---

## Configuration

### GeminiCompleteConfig

```python
@dataclass
class GeminiCompleteConfig:
    # PDF processing
    pdf_dpi: int = 200              # Resolution for PDF→Image
    max_pdf_pages: int = 0          # 0 = no limit
    pdf_timeout_per_page: int = 120 # Seconds per page

    # Image processing
    max_image_dimension: int = 4096 # Max width/height
    image_quality: int = 95         # JPEG quality

    # Retry settings
    max_retries_per_page: int = 2
    retry_delay_seconds: float = 1.0
```

---

## Rate Limiting

### Exponential Backoff

```mermaid
flowchart TD
    A[API Call] --> B{Success?}

    B -->|Yes| C[Reset Delay to 2s]
    B -->|Rate Limited| D[Current Delay]

    D --> E[Wait: delay + jitter]
    E --> F[Multiply Delay × 2]
    F --> G{Delay > 60s?}

    G -->|Yes| H[Cap at 60s]
    G -->|No| I[Keep New Delay]

    H --> J[Retry API Call]
    I --> J
    J --> B

    C --> K[Return Result]
```

### Rate Limit Configuration

```python
@dataclass
class RateLimitConfig:
    initial_delay: float = 2.0      # Start at 2 seconds
    max_delay: float = 60.0         # Cap at 60 seconds
    multiplier: float = 2.0         # Double each failure
    max_retries: int = 5            # Max 5 retries
```

---

## Output Structure

### ExtractedContent

```json
{
  "text": "Combined text from all pages...",
  "structured_data": {
    "gemini_complete": true,
    "approach": "gemini_complete",
    "pages": 5,
    "pages_processed": 5,
    "models_used": ["gemini-2.0-flash"],
    "total_tokens": 4500,
    "processing_time_ms": 12340
  },
  "metadata": {
    "file_type": ".pdf",
    "extraction_method": "gemini_vision",
    "confidence": 0.92,
    "processing_time_ms": 12340,
    "tokens_used": 4500,
    "ocr_engines_used": ["gemini_2.0-flash"]
  }
}
```

---

## When to Use GEMINI_COMPLETE

### Best Use Cases

| Document Type | Why Use |
|---------------|---------|
| Graphs/Charts | Visual understanding required |
| Complex Layouts | Multi-column, mixed content |
| Low Quality Scans | Gemini handles noise better |
| Handwritten + Printed | Mixed content interpretation |
| Non-standard Formats | Gemini adapts to any format |

### When NOT to Use

| Scenario | Better Alternative |
|----------|-------------------|
| High volume processing | LIBRARY_GEMINI (cost) |
| Simple text documents | FULLY_LOCAL (free) |
| Offline requirement | FULLY_LOCAL |
| Budget constraints | LIBRARY_GEMINI |

---

## Cost Analysis

### Per-Document Cost

| Document Type | Pages | Estimated Cost |
|---------------|-------|----------------|
| Single Image | 1 | ~$0.0001 |
| 5-page PDF | 5 | ~$0.0005 |
| 20-page PDF | 20 | ~$0.002 |
| 100-page PDF | 100 | ~$0.01 |

### Monthly Cost Projection

| Daily Volume | Monthly Cost |
|--------------|--------------|
| 50 docs | ~$0.15 |
| 200 docs | ~$0.60 |
| 1000 docs | ~$3.00 |

---

## File Location

**Implementation:** `app/services/file_processing/ocr/gemini_complete.py`

**Singleton Instance:** `gemini_complete`

```python
from app.services.file_processing.ocr import gemini_complete

# Process image
result = await gemini_complete.process(Path("invoice.jpg"))

# Process PDF
result = await gemini_complete.process(Path("document.pdf"))

# Check availability
if gemini_complete.is_available():
    # API key is configured
    pass
```

---

## Error Handling

### Common Errors and Handling

| Error | Cause | Handling |
|-------|-------|----------|
| `ValueError` | Unsupported file type | Return clear message |
| `RuntimeError` | All models failed | After 4 model attempts |
| `TimeoutError` | API timeout | Retry with next model |
| `ResourceExhausted` | Rate limited | Exponential backoff |

### Error Flow

```mermaid
flowchart TD
    A[API Error] --> B{Error Type?}

    B -->|Rate Limited| C[Exponential Backoff]
    B -->|Timeout| D[Try Next Model]
    B -->|Invalid Response| E[Try Next Model]
    B -->|Auth Error| F[Fail Immediately]

    C --> G[Wait and Retry]
    D --> H{More Models?}
    E --> H

    H -->|Yes| I[Use Next Model]
    H -->|No| J[RuntimeError: All Failed]

    G --> K[Retry Same Model]
    I --> L[Continue Processing]

    style F fill:#8B0000,color:#fff
    style J fill:#8B0000,color:#fff
```
