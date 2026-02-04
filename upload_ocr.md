# ERPSense — File Upload & OCR Implementation Roadmap

## Table of Contents

1. [Current State](#1-current-state)
2. [Goal](#2-goal)
3. [Approach Comparison: 3 Options](#3-approach-comparison)
4. [Recommended Architecture (Hybrid Pipeline)](#4-recommended-architecture)
5. [File-Type Handling Strategy](#5-file-type-handling-strategy)
6. [Gemini API — Pricing & Cost Breakdown](#6-gemini-api--pricing--cost-breakdown)
7. [Cost Estimation Scenarios](#7-cost-estimation-scenarios)
8. [Python Libraries Required](#8-python-libraries-required)
9. [Implementation Plan (Step-by-Step)](#9-implementation-plan)
10. [API Design](#10-api-design)
11. [Frontend Changes](#11-frontend-changes)
12. [Risk & Considerations](#12-risks--considerations)
13. [Sources](#13-sources)

---

## 1. Current State

- **Backend:** FastAPI (Python), using `gemini-3-flash-preview` via LangChain
- **Frontend:** Next.js 14, React 18, TailwindCSS
- **File Upload:** NOT implemented. The paperclip icon in `chat-input.tsx` is a UI placeholder — no upload endpoint, no file processing logic exists
- **OCR:** None
- **Storage:** No file storage (MinIO/S3 planned but not set up)
- **Gemini Integration:** Already active for chat (text-only via `langchain-google-genai`)

---

## 2. Goal

Accept **any file type** uploaded by the user — `.pdf`, `.docx`, `.xlsx`, `.csv`, `.txt`, `.md`, `.tex`, `.png`, `.jpg`, `.jpeg`, `.gif`, `.bmp`, `.tiff`, `.webp`, etc. — extract content in a **properly structured format**, and feed it into the chat context so Gemini can reason over it.

---

## 3. Approach Comparison

### Option A: Gemini-Only (Send files directly to Gemini API)

| Pros | Cons |
|------|------|
| Simplest implementation — just upload file to Gemini | Costs money per request (token-based) |
| Gemini natively handles PDF (up to 1000 pages), images | Cannot process `.docx`, `.xlsx`, `.csv` natively |
| Understands layout, tables, charts, diagrams | 50MB file size limit for inline; must use Files API for larger |
| Structured output via `response_schema` (Pydantic) | Rate limits on free tier (1,500 RPD for Flash) |
| Zero infrastructure — no Tesseract, no system deps | You pay for every extraction, even re-uploads |

**Verdict:** Great for PDFs and images. Cannot handle Office formats natively. Need a parser layer for `.docx`/`.xlsx`.

### Option B: Full Local OCR Pipeline (Tesseract + Parsers)

| Pros | Cons |
|------|------|
| Free — no API costs | Tesseract OCR is ~1000x slower than native text extraction |
| Full control, works offline | Complex setup (Tesseract system dependency, language packs) |
| Handles all file types with right parsers | Poor on complex layouts, tables, handwritten text |
| python-docx, openpyxl, PyMuPDF are fast & reliable | No semantic understanding — just raw text |

**Verdict:** Good for digital documents. Struggles with scanned/image-heavy content. No intelligence in extraction.

### Option C: Hybrid Pipeline (Recommended)

| Pros | Cons |
|------|------|
| Best accuracy across ALL file types | Slightly more complex to build |
| Free extraction for digital files (docx, xlsx, csv, txt) | Still costs for scanned PDFs / images via Gemini |
| Gemini only called when OCR is actually needed | Need to detect if PDF is scanned vs native |
| Structured output from Gemini for complex documents | — |
| Fastest possible — parsers for digital, API for visual | — |

**Verdict: THIS IS THE RECOMMENDED APPROACH.** Use free local parsers for digital documents, Gemini Vision for scanned/image content.

---

## 4. Recommended Architecture (Hybrid Pipeline)

```
User uploads file
       │
       ▼
┌─────────────────────┐
│  File Type Router    │
│  (by MIME / ext)     │
└─────┬───────────────┘
      │
      ├── .txt / .md / .csv / .tex ──► Direct text read (free, instant)
      │
      ├── .docx ──────────────────────► python-docx parser (free, instant)
      │
      ├── .xlsx / .xls ──────────────► openpyxl / pandas (free, instant)
      │
      ├── .pdf ──────────────────────► PyMuPDF text extraction
      │                                  │
      │                          Has text layer?
      │                           ├── YES ► Use extracted text (free)
      │                           └── NO ──► Gemini Vision API (paid)
      │
      ├── .png/.jpg/.jpeg/.gif/.bmp/.tiff/.webp
      │                          ──────────► Gemini Vision API (paid)
      │
      └── Unknown ───────────────────► Try text read, fallback to Gemini
              │
              ▼
┌─────────────────────┐
│  Structured Output   │
│  (Markdown / JSON)   │
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  Inject into Chat    │
│  Context for Gemini  │
└─────────────────────┘
```

---

## 5. File-Type Handling Strategy

| File Type | Method | Library / API | Cost | Speed |
|-----------|--------|---------------|------|-------|
| `.txt`, `.md`, `.csv` | Direct read | Python built-in / `csv` | Free | Instant |
| `.tex` | Direct read (plain text) | Python built-in | Free | Instant |
| `.docx` | Parse document structure | `python-docx` | Free | ~50ms |
| `.xlsx`, `.xls` | Parse spreadsheet | `openpyxl` / `pandas` | Free | ~100ms |
| `.pdf` (native/digital) | Extract embedded text | `PyMuPDF` (fitz) | Free | ~120ms |
| `.pdf` (scanned/image) | Vision OCR | Gemini API | ~$0.001-0.01/page | ~2-5s/page |
| `.png`, `.jpg`, `.jpeg` | Vision OCR | Gemini API | ~$0.0001/image | ~1-3s |
| `.gif`, `.bmp`, `.tiff`, `.webp` | Convert → Vision OCR | Pillow + Gemini API | ~$0.0001/image | ~1-3s |

### How to Detect if a PDF is Scanned vs Native

```python
import fitz  # PyMuPDF

def is_scanned_pdf(file_path: str) -> bool:
    doc = fitz.open(file_path)
    for page in doc:
        text = page.get_text().strip()
        if len(text) > 50:  # Has meaningful text
            return False
    return True  # No text found — it's scanned
```

---

## 6. Gemini API — Pricing & Cost Breakdown

### Models Relevant to OCR (January 2026)

| Model | Input (per 1M tokens) | Output (per 1M tokens) | Best For |
|-------|----------------------|------------------------|----------|
| **Gemini 2.0 Flash** | $0.10 | $0.40 | Budget OCR, high volume |
| **Gemini 2.5 Flash-Lite** | $0.10 | $0.40 | Cheapest option |
| **Gemini 2.5 Flash** | $0.30 | $2.50 | Good balance of quality/cost |
| **Gemini 3 Flash Preview** | $0.50 | $3.00 | Best Flash-tier quality |
| **Gemini 2.5 Pro** | $1.25 | $10.00 | Complex document understanding |
| **Gemini 3 Pro Preview** | $2.00 | $12.00 | Best-in-class document AI |

### Key Pricing Facts

- **Images:** Each image ≈ 258 tokens input regardless of size (~$0.000026 per image on Gemini 2.0 Flash)
- **PDF pages:** Native text in PDFs is extracted for free (not charged as tokens) since Gemini 3
- **Batch API:** 50% discount on all models (process files async, results within 24h)
- **Free Tier:** Up to 1,500 requests/day on Gemini 2.0 Flash (generous for dev/small scale)
- **Context Caching:** Re-use uploaded files without re-paying input tokens

### Token Estimation per File Type

| Document Type | Approx. Tokens (Input) | Cost (Gemini 2.0 Flash) |
|--------------|----------------------|------------------------|
| 1-page image OCR | ~258 tokens + prompt (~100) | ~$0.000036 |
| 10-page scanned PDF | ~2,580 + prompt (~200) | ~$0.00028 |
| 50-page scanned PDF | ~12,900 + prompt (~200) | ~$0.0013 |
| Complex invoice image | ~258 + prompt (~300) | ~$0.000056 |

These costs are **extremely low**. Even at 1,000 documents/day, you'd spend ~$0.04-$1.30/day depending on complexity.

---

## 7. Cost Estimation Scenarios

### Scenario A: Small Business (50 uploads/day)

| Item | Volume | Model | Monthly Cost |
|------|--------|-------|-------------|
| DOCX/XLSX/CSV/TXT files | 30/day | Local parsers | **$0.00** |
| Image OCR (invoices, receipts) | 15/day | Gemini 2.0 Flash | **~$0.02/month** |
| Scanned PDF (5 pages avg) | 5/day | Gemini 2.0 Flash | **~$0.02/month** |
| **TOTAL** | | | **~$0.04/month** |

### Scenario B: Medium Business (500 uploads/day)

| Item | Volume | Model | Monthly Cost |
|------|--------|-------|-------------|
| DOCX/XLSX/CSV/TXT files | 300/day | Local parsers | **$0.00** |
| Image OCR | 150/day | Gemini 2.5 Flash | **~$0.50/month** |
| Scanned PDF (10 pages avg) | 50/day | Gemini 2.5 Flash | **~$1.50/month** |
| **TOTAL** | | | **~$2.00/month** |

### Scenario C: Heavy Usage (2,000 uploads/day)

| Item | Volume | Model | Monthly Cost |
|------|--------|-------|-------------|
| DOCX/XLSX/CSV/TXT files | 1,200/day | Local parsers | **$0.00** |
| Image OCR | 500/day | Gemini 2.5 Flash | **~$3.30/month** |
| Scanned PDF (10 pages avg) | 300/day | Gemini 2.5 Flash | **~$10/month** |
| **TOTAL** | | | **~$13.30/month** |

**Bottom line:** Gemini-based OCR is absurdly cheap. The hybrid approach makes it even cheaper by avoiding API calls for digital files.

---

## 8. Python Libraries Required

### New Dependencies (add to `requirements.txt`)

```
# File parsing — digital documents (FREE, local)
python-docx>=1.1.0          # .docx parsing
openpyxl>=3.1.0              # .xlsx parsing
PyMuPDF>=1.24.0              # PDF text extraction + page rendering
pandas>=2.2.0                # CSV/Excel reading + data structuring
Pillow>=10.0.0               # Image format conversion (BMP, TIFF, WebP → PNG/JPEG)

# Gemini multimodal (already partially installed)
google-generativeai>=0.8.0   # Direct Gemini API (for file upload + vision)

# File handling
python-multipart>=0.0.9      # FastAPI multipart file upload support
aiofiles>=24.1.0             # Async file I/O
python-magic>=0.4.27         # MIME type detection (reliable file type identification)
```

### Already Installed (no changes needed)

```
langchain-google-genai       # Already in requirements.txt
fastapi                      # Already in requirements.txt
pydantic                     # Already in requirements.txt
```

### System Dependencies

- **None required** for the hybrid approach (no Tesseract needed!)
- Gemini handles all OCR, local libs handle digital files
- If you want a Tesseract fallback (optional, for offline mode): `apt-get install tesseract-ocr`

---

## 9. Implementation Plan (Step-by-Step)

### Phase 1: Backend — File Processing Service

**Step 1.1: Create file processing module**
```
app/
  services/
    file_processing/
      __init__.py
      router.py          # Routes file to correct handler
      text_handler.py     # .txt, .md, .csv, .tex
      docx_handler.py     # .docx
      excel_handler.py    # .xlsx, .xls
      pdf_handler.py      # .pdf (native + scanned detection)
      image_handler.py    # .png, .jpg, etc.
      gemini_ocr.py       # Gemini Vision API integration
      models.py           # Pydantic schemas for extracted content
```

**Step 1.2: File type router logic**
- Detect MIME type using `python-magic`
- Route to appropriate handler
- Each handler returns a standardized `ExtractedContent` schema:
  ```python
  class ExtractedContent(BaseModel):
      text: str                    # Extracted plain text
      structured_data: dict | None # Tables, key-value pairs if applicable
      metadata: dict               # Page count, file type, extraction method
      markdown: str                # Formatted markdown for chat display
  ```

**Step 1.3: Gemini Vision OCR service**
- Use `google-generativeai` SDK directly (not LangChain) for file uploads
- Upload file via `client.files.upload()` for files > 20MB
- For smaller files, send inline with the prompt
- Use structured output (`response_schema`) for consistent JSON
- Model: `gemini-2.0-flash` (cheapest, sufficient for OCR)

### Phase 2: Backend — Upload Endpoint

**Step 2.1: Create upload API endpoint**
```
POST /api/v1/upload
  - Accept: multipart/form-data
  - Fields: file (binary), session_id (optional)
  - Validate: file size (max 50MB), file type (allowlist)
  - Process: extract content via file processing service
  - Return: { file_id, extracted_content, metadata }
```

**Step 2.2: File storage (optional, for future reference)**
- Store uploaded files temporarily (local disk or MinIO if available)
- Store extraction results in Redis (tied to session, same TTL as conversations)
- Clean up files after TTL expires

### Phase 3: Backend — Chat Integration

**Step 3.1: Extend chat to accept file context**
- Modify `ChatRequest` schema to include optional `file_ids: list[str]`
- When file IDs are provided, load extracted content from Redis
- Prepend extracted content to the conversation context
- Gemini then reasons over both the file content and the user's question

**Step 3.2: Update supervisor agent**
- Add file context injection into the system prompt
- Format: include file content as a clearly delimited block in the conversation

### Phase 4: Frontend — Upload UI

**Step 4.1: Wire up the paperclip button**
- Add `<input type="file" multiple accept="*/*">` behind the paperclip icon
- Show file preview (name, size, type icon) before sending
- Upload via `POST /api/v1/upload` with progress tracking
- Display extraction result in chat as a message or attachment

**Step 4.2: Drag & drop support**
- Add drop zone to the chat input area
- Same flow as paperclip upload

**Step 4.3: File preview in messages**
- Show uploaded files as cards in message bubbles
- Display extraction status (processing → done)
- Option to expand/collapse extracted content

### Phase 5: Polish & Edge Cases

- Handle password-protected PDFs (error message)
- Handle corrupted files gracefully
- Add file size validation (frontend + backend)
- Rate limiting on upload endpoint
- Logging & monitoring for OCR costs

---

## 10. API Design

### Upload Endpoint

```
POST /api/v1/upload
Content-Type: multipart/form-data

Request:
  - file: binary (required)
  - session_id: string (optional)

Response (200):
{
  "file_id": "uuid",
  "filename": "invoice.pdf",
  "file_type": "application/pdf",
  "file_size": 245000,
  "extraction": {
    "text": "Invoice #1234\nDate: 2026-01-15\n...",
    "structured_data": {
      "invoice_number": "1234",
      "date": "2026-01-15",
      "items": [...]
    },
    "markdown": "## Invoice #1234\n\n| Item | Qty | Price |\n|...",
    "method": "pymupdf_native",  // or "gemini_vision"
    "pages": 3,
    "tokens_used": 0  // 0 for local, >0 for Gemini
  }
}

Response (400): { "error": "File type not supported" }
Response (413): { "error": "File too large (max 50MB)" }
```

### Extended Chat Endpoint

```
POST /api/v1/chat
{
  "message": "What is the total amount on this invoice?",
  "session_id": "...",
  "file_ids": ["uuid-1", "uuid-2"]  // NEW FIELD
}
```

---

## 11. Frontend Changes

### Files to Modify

| File | Change |
|------|--------|
| `src/components/chat/chat-input.tsx` | Wire paperclip button to file input, add upload logic |
| `src/lib/api/upload.ts` | New file — upload API client |
| `src/types/chat.ts` | Add `FileAttachment` type, extend `SendMessageRequest` |
| `src/lib/hooks/use-chat.ts` | Handle file uploads in send flow |
| `src/components/chat/message-bubble.tsx` | Render file attachments in messages |
| `src/stores/chat-store.ts` | Add pending files state |

### New Components to Create

| Component | Purpose |
|-----------|---------|
| `file-preview.tsx` | Shows file card (icon, name, size) before/after upload |
| `file-drop-zone.tsx` | Drag & drop overlay for chat area |
| `upload-progress.tsx` | Progress bar during upload/extraction |

---

## 12. Risks & Considerations

### Technical Risks

| Risk | Mitigation |
|------|------------|
| Large files (>50MB) timeout | Implement chunked upload or reject with clear error |
| Scanned PDF detection fails | Default to Gemini if PyMuPDF returns < 50 chars/page |
| Gemini rate limits | Use `gemini-2.0-flash` free tier (1,500 RPD), queue excess |
| Password-protected files | Detect and return user-friendly error |
| Malicious file uploads | Validate MIME type, scan with ClamAV (optional), size limits |

### Cost Risks

| Scenario | Risk | Mitigation |
|----------|------|------------|
| User uploads 100-page PDF | High token count | Warn user, set page limit (e.g., 50 pages) |
| Abuse/spam uploads | Unnecessary API spend | Rate limit per user (e.g., 20 uploads/hour) |
| Large images (4K+) | Higher token count | Resize to max 2048px before sending to Gemini |

### Security

- **Never execute uploaded files** — only parse/read them
- **Validate file types server-side** using `python-magic` (not just extension)
- **Sanitize extracted content** before injecting into prompts (prevent prompt injection)
- **Temporary storage only** — delete files after extraction (or after TTL)

---

## 13. Sources

### Gemini API & Pricing
- [Gemini API Pricing — Official](https://ai.google.dev/gemini-api/docs/pricing)
- [Gemini API Pricing 2026 Complete Guide — AI Free API](https://www.aifreeapi.com/en/posts/gemini-api-pricing-2026)
- [Gemini 3 Pro & API Costs 2026 — GLBGPT](https://www.glbgpt.com/hub/gemini-3-pro-costs-gemini-3-api-costs-latest-insights-for-2025/)
- [Gemini API Cost 2026 — Apidog](https://apidog.com/blog/gemini-3-0-api-cost/)
- [Gemini 3 for Developers — Google Blog](https://blog.google/technology/developers/gemini-3-developers/)

### Document Processing & OCR
- [Gemini Document Understanding — Official](https://ai.google.dev/gemini-api/docs/document-processing)
- [Gemini Files API — Official](https://ai.google.dev/gemini-api/docs/files)
- [Gemini File Input Methods — Official](https://ai.google.dev/gemini-api/docs/file-input-methods)
- [Gemini 2.0 Flash OCR Workflow — Apidog](https://apidog.com/blog/gemini-2-0-flash-ocr/)
- [Structured Data from PDFs with Gemini — Phil Schmid](https://www.philschmid.de/gemini-pdf-to-data)
- [Gemini for Document OCR — Rogue Marketing](https://the-rogue-marketing.github.io/why-google-gemini-2.5-pro-api-provides-best-and-cost-effective-solution-for-ocr-and-document-intelligence/)
- [Evaluating Gemini for Invoice OCR — DEV Community](https://dev.to/mayankcse/evaluating-google-gemini-for-document-ocr-using-hugging-face-invoice-dataset-567i)

### OCR Libraries & Comparisons
- [Best PDF OCR Software 2026 — Unstract](https://unstract.com/blog/best-pdf-ocr-software/)
- [AI OCR Models Comparison — IntuitionLabs](https://intuitionlabs.ai/articles/ai-ocr-models-pdf-structured-text-comparison)
- [Document Extraction: LLMs vs OCRs 2026 — Vellum](https://www.vellum.ai/blog/document-data-extraction-llms-vs-ocrs)
- [Best Python PDF-to-Text Libraries 2026 — Unstract](https://unstract.com/blog/evaluating-python-pdf-to-text-libraries/)
- [Tesseract OCR Guide 2026 — Unstract](https://unstract.com/blog/guide-to-optical-character-recognition-with-tesseract-ocr/)
- [PyMuPDF OCR Docs](https://pymupdf.readthedocs.io/en/latest/recipes-ocr.html)
- [Top 3 OCR Tools 2026 — Koncile](https://www.koncile.ai/en/ressources/top-3-best-ocr-tools-for-extracting-text-from-images-in-2025)

### Open Source References
- [gemini-ocr — GitHub](https://github.com/skitsanos/gemini-ocr)
- [gemini-file-api — GitHub](https://github.com/abhaydixit07/gemini-file-api)
- [Google Cloud Document Processing Notebook](https://github.com/GoogleCloudPlatform/generative-ai/blob/main/gemini/use-cases/document-processing/document_processing.ipynb)
- [SpeedyOCR — Gemini Competition](https://ai.google.dev/competition/projects/speedyocr)
