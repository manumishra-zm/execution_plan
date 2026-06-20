# ERPSense Auto-Compression — Runtime, Timings & Decisions

> Companion to `plan.md`. This documents **how the implemented feature actually runs**, **how long it takes** (measured), how **compression** and **preview** work, the **library analysis**, and the **ERPNext built-in-optimizer** decision.
>
> **Status:** implemented on branch `feat/auto-compression`, **OFF by default** (`COMPRESSION_ENABLED=false`) — a runtime no-op until an env flips it. **83 backend tests + 8 frontend chip tests green (~96% coverage on the compression package, orchestrator 100%)**; ruff/mypy/eslint/type-check clean. Covers images, scanned PDFs, **and** heavy native DOCX/XLSX/PDF (embedded-image recompress, behind a flag). The upload chip now **reflects the compressed size in the UI** (§14) and the whole flow is traceable through **structured logs** (§15).

---

## 1. TL;DR

- On upload, the file is **compressed, then OCR'd — sequentially** within the same request (compression runs in a worker thread so it never blocks *other* requests). It is deliberately **not** run concurrently with OCR because **PyMuPDF (used by both) is not multithread-safe** (see §2.3). The added time is small vs OCR (see §3).
- **OCR reads the original** file; **storage saves a smaller, compressed copy** under the **same filename + extension**.
- Compression targets **images** (≤200 KB, ≤1200 px) and **scanned PDFs** (≤300 KB/page). **Heavy native PDF/DOCX/XLSX** (big embedded images) are also compressible behind `compression_native_rebuild_enabled` — only their embedded images are recompressed, text/data untouched. Pure-text files and files <500 KB are **stored as-is**.
- **Preview/download is unchanged** — you see your files exactly as before, just smaller. If compression fails or wouldn't shrink the file, the **original is stored** (you can always open what you uploaded).
- **No new libraries** were added — uses **Pillow, PyMuPDF, reportlab** plus stdlib **zipfile** and the already-present **python-docx / openpyxl** (for native-file validation). **ERPNext's built-in image optimizer is intentionally not used** (see §8). How each library compresses internally: see §12.
- **Upload limits aligned to spec §6.1:** **10 MB / file** on both surfaces (was 30 MB on Genie); Genie session = **3 files × 10 MB = 30 MB total** (`MAX_FILE_SIZE`/`MAX_TOTAL_SIZE`/`MAX_FILES_PER_SESSION` in `upload.py`, mirrored in FE `constants.ts`). ⚠️ Consequence: files **10–30 MB that previously uploaded are now rejected** at upload (spec-compliant).

---

## 2. The pipeline — how it runs

### 2.1 Genie Chat (OCR surface)

```
  upload  →  validate (auth · rate-limit 10/min · ≤10MB/file · ≤3 files & ≤30MB/session · type · PDF ≤10 pages)
                    │
            read bytes into memory  +  write a temp file
                    │
   COMPRESSION (copy of in-memory bytes) → save smaller file to GCS (same name)
                    │   (runs in a worker thread; never touches the OCR temp file)
                    ▼
   OCR PIPELINE (reads the untouched temp file) → text / tables / JSON
                    │
                    ▼
            return response
```

- Compression runs **before** OCR, sequentially. The two never overlap, which is required because **PyMuPDF is not thread-safe** (§2.3).
- Compression reads a **copy of the in-memory bytes** — it never touches the OCR temp file, so it cannot affect extraction.
- The CPU-bound compression work runs inside `asyncio.to_thread`, so it never blocks the server's event loop / **other** requests.
- The compressed copy replaces the bytes at the single existing storage write, so **filename, extension and the preview route are unchanged**.

### 2.2 ERPHome (attachment surface — no OCR)

```
  attach file → validate (≤10MB · type) → COMPRESS (copy) → POST to ERPNext → Frappe File
```

There is **no OCR** here, so there's nothing to overlap with — compression is the only added step (small; see §3).

### 2.3 "Parallel" — what we tried, and why it's sequential

The architecture diagram showed OCR ‖ compression as concurrent "independent workers." We initially implemented that with `asyncio.gather`, but a blocker review caught a real hazard and we reverted to sequential:

| Approach | Status | Why |
|---|---|---|
| **Concurrent in-request** (OCR ‖ compression via `asyncio.gather`) | ❌ **reverted** | For a scanned PDF, OCR and compression **both use PyMuPDF (fitz)**. The installed PyMuPDF (1.27.x) is **not multithread-safe** — it calls `reinit_singlethreaded()` at import and shares one global MuPDF context. Running fitz on two threads at once risks **intermittent heap corruption / segfault** (passes CI, fails rarely in prod). Not worth it. |
| **Sequential in-request** (compress → then OCR) | ✅ **shipped** | Zero concurrent-fitz risk. Compression still runs in `asyncio.to_thread` (doesn't block other requests). Added latency is small (§3). |
| **Background worker/queue** (request returns instantly) | ⏸️ not built | Net-new infrastructure (Redis/RQ worker for this path); would also need fitz serialized. Future work. |

**Net:** the safe, shipped design is **sequential**. As defense-in-depth, a **process-wide fitz lock** is also implemented — `app/services/file_processing/_fitz_lock.py` (`FITZ_LOCK = threading.RLock()` + a `run_fitz()` helper) wraps **every** fitz call site in **both** compression and OCR, so even if a future code path runs them off the event loop they can never execute MuPDF concurrently. If true in-request parallelism is ever wanted, that lock is the foundation it would build on (or move fitz to a separate process).

---

## 3. Timing & performance

### 3.1 Compression cost (measured on this machine)

| Input | Size | Compress time |
|---|---|---|
| Image 2400×1600 | 4 MB → 235 KB | **~206 ms** |
| Image 4000×3000 | 13 MB → 222 KB | **~380 ms** |
| Scanned PDF, 3 pages | 2.2 MB → 492 KB | **~401 ms** |
| Scanned PDF, 10 pages | 2.2 MB → 493 KB | **~1.16 s** |

### 3.2 OCR cost (from the active OCR flow doc)

- **Native / text** files (`.txt .csv .docx .xlsx`, text-layer PDF) → pure Python libraries: **~0.1–1 s**, $0, no AI.
- **Images + scanned PDFs** → Gemini Vision (Vertex, 4-model fallback chain `2.5-flash → 2.0-flash → 2.5-pro → 2.0-flash-lite`): typically **~2–10 s** for one image; **~10–60 s+** for a multi-page scan (per-page model timeouts are 60–180 s).

### 3.3 What you actually wait (wall-clock), per case

Because the flow is **sequential**, wall-clock = compression **+** OCR. Compression is a small fraction of OCR, so the added latency is modest:

| Upload | Compression (added) | OCR | Total wall-clock |
|---|---|---|---|
| Image (Genie) | +~0.2–0.4 s | Gemini ~2–10 s | OCR **+ ~0.2–0.4 s** |
| Scanned PDF (Genie) | +~0.4–1.2 s | Gemini ~10–60 s+ | OCR **+ ~0.4–1.2 s** (small %) |
| Native PDF / DOCX / XLSX / CSV (Genie) | **0** (skipped) | libraries ~0.1–1 s | ~unchanged |
| **ERPHome attachment** (no OCR) | +~0.2–1.2 s | — | **~0.2–1.2 s** (compression is the only work) |

**Takeaway:** on the Genie OCR surface compression adds only a small fraction on top of OCR (sub-second for images, ~1 s for a 10-page scan, vs OCR's seconds-to-minutes). On ERPHome you pay the compression time directly, in exchange for a much smaller stored file. (A concurrent design would hide the Genie compression time entirely, but is unsafe with the current PyMuPDF — see §2.3.)

---

## 4. Compression criteria & per-format behavior

### 4.1 When it compresses — the decision gate (when & at what size)

A file is compressed **only if every check below passes**; otherwise it is stored **as-is** (original bytes, untouched). Evaluated in order:

| # | Check | Pass condition | Else → |
|---|---|---|---|
| 1 | **Master + surface switch** | `COMPRESSION_ENABLED=true` **AND** the surface flag (`_GENIE_ENABLED` / `_ERPHOME_ENABLED`) on | `disabled` → as-is |
| 2 | **Minimum size** | `size ≥ 500 KB` (`COMPRESSION_MIN_BYTES`) | `below_min` → as-is |
| 3 | **Eligible type** | **image** or **scanned PDF** (default); **native PDF / DOCX / XLSX** only if `NATIVE_REBUILD_ENABLED` | `not_eligible` → as-is |
| 4 | **Not animated** | single-frame image | `animated` → as-is |
| 5 | **Under security caps** | image ≤ 64 MP · PDF ≤ 50 pages · OOXML within zip-bomb caps · page raster ≤ pixel cap | exceeds → **fallback to original** |
| 6 | **Actually smaller** | compressed bytes re-open **and** are `< original` | not smaller / error → **fallback to original** |

**The trigger, in one line:** an **eligible image / scanned PDF (or flagged heavy native) that is ≥ 500 KB**. There is **no upper-size gate _inside_ compression** — a 9 MB image is *more* likely to be compressed, not exempt. (The 10 MB/file limit is a separate **upload-acceptance** cap, not a compression skip — see the surface table below.)

#### Target sizes — what it compresses down to

| Type | Compresses when | Down to |
|---|---|---|
| **Floor (all types)** | ≥ **500 KB** | below this → never compressed |
| **Image** (jpg/jpeg/png/webp/gif/bmp/tiff) | ≥ 500 KB | longest edge **≤ 1200 px**, re-encoded to **≤ 200 KB** (JPEG/WebP quality ladder **80 → 65 → 50 → 35**; PNG `optimize` → palette-quantize). **Same extension kept.** |
| **Scanned PDF** (image-based) | ≥ 500 KB | each page rasterized at **150 DPI**, JPEG'd to **≤ 300 KB/page**, rebuilt PDF |
| **Native PDF** (flag) | ≥ 500 KB | embedded images over **200 DPI → downsampled to 150 DPI** (JPEG q80); text/vectors untouched |
| **DOCX / XLSX** (flag) | ≥ 500 KB | only embedded raster images recompressed (JPEG q80 / PNG optimize); all XML/text/data byte-identical |

All targets are env-tunable (§9).

#### Genie (chat) vs ERPHome (attachment) — same criteria, different wrapper

The **size/type criteria above are identical on both surfaces** (same `compress_upload`, same thresholds). Only the surrounding rules differ:

| | **Genie (chat · OCR)** | **ERPHome (ERPNext attach)** |
|---|---|---|
| Compression criteria | ≥ 500 KB + eligible type | **identical** |
| Surface enable flag | `COMPRESSION_GENIE_ENABLED` | `COMPRESSION_ERPHOME_ENABLED` |
| Per-file upload cap | **10 MB** (`MAX_FILE_SIZE`) | **10 MB** (`_MAX_ATTACHMENT_SIZE_BYTES`) |
| Session-total / file-count cap | 30 MB / 3 files | **none** — per-file only |
| OCR | yes (compress runs first, sequentially) | **no OCR** |
| Stored to | **GCS** (`sessions/{id}/originals/`) | **ERPNext Frappe File** (never GCS) |
| Compress runs | before OCR | before the adapter POST to ERPNext |
| Opt-outs | — | company **logo/branding** sends `compress=False` |

> **Over 10 MB** is **rejected at the upload endpoint** on both surfaces — it never reaches compression. Everything **500 KB – 10 MB** of an eligible type **is** compressed. Being "already ~10 MB" makes a file *more* likely to compress, not exempt.

#### Worked examples (with compression ON)

| Upload | Surface | Outcome |
|---|---|---|
| 4 MB phone photo (jpg) | either | → 1200 px, ≈ 200 KB — **compressed** |
| 8 MB scanned PDF, 6 pages | either | ≈ 300 KB/page ≈ 1.8 MB — **compressed** |
| 300 KB receipt image | either | under the 500 KB floor — **stored as-is** |
| 2 MB Word/Excel with photos | either | **only** with `NATIVE_REBUILD_ENABLED` — else as-is |
| 1 MB CSV / text-layer PDF | either | not an eligible type — **stored as-is** |
| 70 MP image | either | over the 64 MP cap — **fallback to original** |
| Company logo | ERPHome | `compress=False` — **never compressed** |
| 12 MB file | either | **rejected at upload** (10 MB cap) — never reaches compression |

### 4.2 Per-format compression

| Type | What happens | Target | Library |
|---|---|---|---|
| **Image** (`.jpg .jpeg .png .webp .gif .bmp .tiff`) | EXIF-orient → if longest edge >1200 px, downscale (keep aspect) → re-encode stepping quality down until ≤target. PNG stays PNG (optimize → palette-quantize if opaque); JPEG stays JPEG; **extension never changes** | ≤200 KB, ≤1200 px | Pillow |
| **Scanned PDF** (image-based) | Render each page at 150 DPI → JPEG-compress each page to ≤target → rebuild a fresh `.pdf` (same page geometry) → validate it opens | ≤300 KB/page | PyMuPDF + Pillow |
| **HEAVY native PDF / DOCX / XLSX** (embedded images) | **As-is by default.** When `compression_native_rebuild_enabled=true`: recompress **only the embedded raster images** (text/data/XML/vectors/structure preserved exactly) → re-open to validate → fall back if it isn't smaller. *This is the answer to "what if a native doc is too heavy."* | image-driven | PyMuPDF `rewrite_images` (PDF); `zipfile`+Pillow on `media/*` (DOCX/XLSX) |
| **Pure-text native** (text-only PDF / CSV / TXT) | **Stored as-is** — text is already efficient; can't be losslessly shrunk while staying the same openable format | — | — |
| **Animated images** (multi-frame GIF/WEBP/TIFF) | **Stored as-is** (out of scope) | — | — |
| **Files < 500 KB** | **Stored as-is** (not worth compressing) | — | — |

**Always-safe behavior (fallback):**
- If compression **errors** (corrupt/unknown/encrypted file) → store the **original**.
- If the "compressed" file is **not actually smaller** → store the **original**.
- Compression **never raises** into the request — it can never block your upload or OCR.

**Security caps (defense-in-depth):** image-bomb guard (rejects images over ~64 MP by declared dimensions, thread-safe) — applied to standalone images **and** images embedded in DOCX/XLSX; **scanned-PDF page cap** (≤50) + **per-page raster bound** (DPI × page-area ≤ the pixel cap, so a huge/zoomed page can't allocate an enormous bitmap); **OOXML zip-bomb caps** (member-count ≤ 5 000 and total inflate ≤ 400 MB, checked from the ZIP directory before reading); decode-time validation (a spoofed/garbage file fails to decode → fallback). All caps raise an internal `CompressionError` that the orchestrator converts to a safe fallback-to-original. No secrets/PII or file content in logs.

---

## 5. Storage — where files go (one store per surface)

| Surface | Durable store | Path / sink |
|---|---|---|
| **Genie OCR** | **GCS only** | `sessions/{session_id}/originals/{filename}` (the compressed copy) + OCR outputs `.json/.md/.txt` |
| **ERPHome attachment** | **ERPNext (Frappe File) only** | `/api/method/upload_file` → File doctype |

No file is written to both. The temp OCR file is deleted after processing.

---

## 6. Preview / download — "will I still see my files?"

**Yes — identical to before, just smaller.**

- The stored file keeps the **same filename and extension** (`invoice.jpg` stays `invoice.jpg`, `scan.pdf` stays `scan.pdf`).
- Preview/download is **unchanged**: the frontend builds the preview URL from the filename and infers the type from the extension (both untouched), so it renders normally. Genie serves from `/api/v1/upload/sessions/{id}/originals/{filename}`; ERPHome serves through its `/erp/file` proxy.
- **Quality:** lossy but visually fine for documents/receipts/scans (the point is smaller-but-readable). If a file can't be safely compressed, you get the **original** back — so you can always open what you uploaded.
- It's **quiet during processing** — no "compressing…" spinner — but once processing completes the **upload chip's size number updates from the original to the compressed size** (e.g. `1.9 MB` → `177.6 KB`). Just the number swaps; no extra badge. See §14.

---

## 7. Libraries — already optimal, nothing added

The feature uses only libraries **already pinned** in `erpsense-backend/requirements.txt`:

| Library | Role | Status |
|---|---|---|
| **Pillow** (`>=10.3.0`, bumped for CVEs) | image decode/resize/encode | already present |
| **PyMuPDF (fitz)** (`>=1.23.0`) | PDF page-count, rasterize, rebuild, `rewrite_images` | already present |
| **reportlab** (`>=4.1`) | available for PDF assembly | already present |
| **zipfile** (stdlib) | unzip/repack DOCX/XLSX (OOXML) for native rebuild | stdlib |
| **python-docx / openpyxl** (`>=1.0` / `>=3.1`) | re-open DOCX/XLSX to validate after native rebuild | already present |

No new runtime dependency was added. Not installed / not used: `pikepdf`, `img2pdf`, `opencv`, `wand`, `pdf2image`.

**Optional future libs (only if a need appears):**
- **pikepdf** — would enable native-PDF stream/object rebuild (the optional, flag-OFF path; spec says skip native PDFs).
- **pillow-heif** — only if you want to support iPhone **HEIC** uploads (not a supported type today).

---

## 8. ERPNext / Frappe built-in image optimizer — checked, not used (on purpose)

The deployed Frappe (**version-15/16**) ships a built-in optimizer: `optimize_image()`, an `optimize` param on `upload_file`, and a System-Settings toggle **"Automatically optimize images on upload."**

We **deliberately don't rely on it**, because it's a strict subset of our pipeline:

| | Our pipeline | Frappe built-in optimize |
|---|---|---|
| Images | ✅ (our thresholds) | ✅ (Frappe's thresholds) |
| **Scanned PDFs** | ✅ | ❌ |
| **Genie surface (→ GCS)** | ✅ | ❌ (never touches ERPNext) |
| Saves upload bandwidth to ERPNext | ✅ (we send already-small bytes) | ❌ (optimizes after full upload) |
| Consistent across both surfaces | ✅ | ❌ |

→ **Keep our pipeline as the single source of truth.**

**One interaction to be aware of:** if a tenant turns **ON** Frappe's "Automatically optimize images on upload" System Setting, an ERPHome image we already compressed would be optimized **again** by Frappe (minor extra quality loss, **not a break**). We considered forcing it off by sending `optimize=0` on the upload, but **did not** — Frappe parses that flag differently across versions (a string `"0"` can read as *truthy* in some code paths and would *trigger* optimization, the opposite of intended), and the framework source isn't available to verify. The safe guidance is operational: **leave that System Setting OFF** (it is off by default) since we handle compression.

---

## 9. Config / kill switch / enabling in dev

All knobs live in `app/config.py` (`Settings`), env-overridable; documented in `.env.example`:

| Setting | Default | Meaning |
|---|---|---|
| `COMPRESSION_ENABLED` | `false` | master kill-switch (no-op until on) |
| `COMPRESSION_GENIE_ENABLED` / `COMPRESSION_ERPHOME_ENABLED` | `true` | per-surface (only when master on) |
| `COMPRESSION_MIN_BYTES` | `512000` | files smaller than this are exempt |
| `COMPRESSION_IMAGE_TARGET_BYTES` / `_MAX_DIMENSION` | `204800` / `1200` | image targets |
| `COMPRESSION_SCANNED_PDF_TARGET_BYTES_PER_PAGE` / `_DPI` | `307200` / `150` | scanned-PDF targets |
| `COMPRESSION_NATIVE_REBUILD_ENABLED` | `false` | **implemented** — recompresses embedded images in heavy DOCX/XLSX/native-PDF (text/data lossless, validate+fallback) |
| `COMPRESSION_MAX_IMAGE_PIXELS` / `_MAX_PDF_PAGES` | `64000000` / `50` | security caps (image pixels; PDF page count + per-page raster bound reuses the pixel cap) |
| `COMPRESSION_MAX_OOXML_ENTRIES` / `_MAX_OOXML_UNCOMPRESSED_BYTES` | `5000` / `419430400` (400 MB) | OOXML (DOCX/XLSX) zip-bomb caps — member count + total inflate |

To turn on in dev: set `COMPRESSION_ENABLED=true` (GCS is already selected when a bucket is configured; no new IAM needed — `objectCreator`+`objectViewer` cover it). Staged rollout: dev → soak → staging/prod.

---

## 10. Implementation status vs plan

| Area | Status |
|---|---|
| Config flags + validators + `.env.example` | ✅ done |
| Shared `is_native_pdf` classifier (OCR sites delegate, behavior-preserved) | ✅ done |
| Compression engine (image + scanned-PDF + orchestrator + metrics) | ✅ done |
| Genie integration (sequential compress→OCR, fallback) | ✅ done — reverted from `gather` after the PyMuPDF concurrency finding (§2.3) |
| ERPHome integration (compress before POST; logo `compress=False`) | ✅ done |
| Preview/download | ✅ verified transparent (no change needed) |
| Logging & metrics (`compression.decision/.fallback`, `gcs.upload`) | ✅ done — now with human sizes + a `summary=` before→after marker (§15.2) |
| **Structured upload logging** (`upload.processing_start/complete`, `ocr.extraction_summary`, `ocr.output_saved`, `upload.response_summary`) | ✅ done — replaced ~75 banner lines; fixed a reserved-key (`filename`) crash + AST guard test (§15) |
| **UI compressed-size reflection** (chip shows stored size; `UploadResponse` carries `original_size`/`compressed_size`) | ✅ done (§14) |
| **File-type icons** (all accepted types distinct + type-colour-on-complete fix) | ✅ done (§14.3) |
| Process-wide fitz lock (`_fitz_lock.py` / `run_fitz`, OCR + compression) | ✅ done (§2.3) |
| Security guards (image-bomb, page cap, never-raise, decode-validate) | ✅ done |
| Tests (83 backend + 8 frontend chip, ~96% coverage) + ruff/mypy/eslint/type-check gates | ✅ green |
| **Native rebuild** (heavy DOCX/XLSX/native-PDF — embedded-image recompress) | ✅ **implemented** behind `compression_native_rebuild_enabled` (default OFF); text/data lossless, validate+fallback, 6 tests |
| **Background worker** (request returns before processing) | ⏸️ not built — net-new infra, not required |
| Dev/prod Terraform/IAM/runbook | 🟡 ops task — flipping `COMPRESSION_ENABLED` only; no code/IAM change needed |
| Genie compress→persist byte-path test | ✅ done — extracted `_compress_and_persist_original()` helper + `test_upload_compression_wiring.py` asserts the **compressed** copy is stored (not the raw upload), via a real `LocalStorageClient` (`upload.py` itself stays coverage-omitted) |

---

## 11. Files changed / added

**New (backend):**
- `app/services/file_processing/_pdf_classifier.py`
- `app/services/file_processing/_fitz_lock.py` (process-wide fitz `RLock` + `run_fitz`)
- `app/services/file_processing/compression/` (`__init__`, `models`, `config`, `metrics`, `image`, `pdf`, `native`, `orchestrator`)
- `tests/unit/file_processing/test_compression.py`, `test_pdf_classifier.py`, `test_compression_metrics.py`, `test_native_compression.py`, `test_fitz_lock.py`, `test_upload_compression_wiring.py`, `test_upload_logging.py`
- `tests/unit/core/test_compression_config.py`
- `tests/unit/adapters/test_attachment_compression.py`

**New (frontend):**
- `tests/unit/components/chat/file-attachment.test.tsx` (compressed-size display + file-type icons)

**Modified (backend):**
- `app/config.py` (compression Settings + validators)
- `app/api/v1/endpoints/upload.py` (compress→persist sequentially before OCR; `gcs.upload` telemetry; **structured logging** §15; `_compress_and_persist_original()` now **returns** the `CompressionResult`; response carries the sizes)
- `app/services/file_processing/models.py` (`UploadResponse` += `original_size` / `compressed_size`)
- `app/core/logging.py` (`StandardFormatter` appends `extra={}` fields in dev)
- `app/services/file_processing/compression/metrics.py` (human-readable sizes + `summary=` marker)
- `app/adapters/erpnext/_attachments.py` (compress before POST; `compress` kwarg; lazy import)
- `app/services/erpnext_provisioning/_steps_company.py` (logo `compress=False`)
- `app/services/file_processing/handlers/pdf_handler.py`, `ocr/library_gemini.py`, `ocr/gemini_complete.py` (delegate classifier; fitz calls via `run_fitz`)
- `app/services/file_processing/compression/orchestrator.py` + `models.py` (route native PDF / DOCX / XLSX)
- `requirements.txt` (Pillow `>=10.3.0`), `.env.example`

**Modified (frontend):**
- `src/types/chat.ts` (`UploadResponse` + `FileAttachment` += compression sizes)
- `src/lib/hooks/use-chat.ts` (set `compressedSize` on upload completion when smaller)
- `src/components/chat/file-attachment.tsx` (size number swaps to compressed; full file-type icon map + type-colour-on-complete)
- `src/lib/constants.ts` (OCR upload limits aligned to spec §6.1)

---

## 12. How each library compresses internally (per format)

Everything we do reduces to **two engines**: a **lossy image transform** (JPEG/WebP) and a **lossless entropy coder** (DEFLATE/zlib). Text, XML and vectors are never re-encoded — only image *pixels* are.

### 12.1 Pillow — images
- **JPEG (lossy):** convert to YCbCr → split into **8×8 blocks** → **DCT** (Discrete Cosine Transform) turns each block into frequency coefficients → **quantization** discards high-frequency detail the eye barely sees (the `quality` knob = how aggressive) → **Huffman entropy coding** packs it. `optimize=True` builds optimal Huffman tables; `progressive=True` reorders scans. Lower quality → coarser quantization → smaller + slightly softer.
- **PNG (lossless):** per-row **predictive filtering** (Sub/Up/Average/Paeth — stores pixel *differences*, which compress better) → then **DEFLATE** (zlib). `optimize` tries filter combinations; `quantize()` reduces to a ≤256-colour palette (a lossy colour reduction → much smaller).
- **WebP:** VP8 predictive transform + arithmetic coding (lossy or lossless); `method=6` = slowest/best ratio.
- **resize (LANCZOS):** a windowed-sinc resampling filter — fewer pixels before encoding (the single biggest lever).

### 12.2 PyMuPDF (fitz) — PDF
- **`get_pixmap(matrix=DPI)`** renders a page (text + vector + images) to a raw RGB bitmap; we JPEG-compress that bitmap via Pillow → the **scanned-PDF** rebuild.
- **`rewrite_images(dpi_threshold, dpi_target, quality)`** walks the PDF's embedded image XObjects, decodes each, **downsamples** over-resolution ones and **re-encodes JPEG**, replacing the stream — leaving text/vector operators untouched → the **native-PDF** rebuild.
- **`save(garbage=4, deflate=True, clean=True)`** drops unused/duplicate objects and **FlateDecode**-compresses content streams (zlib/DEFLATE — the PDF format's native stream compression).

### 12.3 zipfile + Pillow — DOCX / XLSX (OOXML / "XML format")
OOXML files are **ZIP packages of XML parts + a `media/` image folder**, and the ZIP already uses **DEFLATE** (zlib: **LZ77** replaces repeated byte runs with back-references, **Huffman** gives frequent symbols shorter codes). Re-compressing the XML itself gains ~nothing, so:
1. **Unzip** with `zipfile`.
2. Keep every **XML part byte-identical** (`document.xml`, styles, `_rels`, `[Content_Types].xml`, sheets — untouched).
3. **Recompress only the raster images** in `word/media/*` / `xl/media/*` with **Pillow** (same image format kept).
4. **Repack** the same ZIP (same entry names/order/compress-type) → same `.docx/.xlsx`.
5. **Re-open** with python-docx / openpyxl to validate.

→ The "XML formatting" / structure / data is preserved **exactly**; all the size win comes from the embedded images.

---

## 13. Per-page rendering — no information loss, no context switching

Each unit is compressed **independently and in place**, so what renders is identical in layout — only image pixels are lighter.

| Format | How each page/part is handled | What's preserved |
|---|---|---|
| **Image** | The single image is re-encoded once (same format/extension) | Dimensions (unless >1200 px), orientation, colours; renders the same |
| **Scanned PDF** | Each **page** is rasterized → JPEG → re-assembled **in original order** with the **same page geometry** (`page.rect`); rebuilt page-count is validated to match | Page order, page size, page count — **page N stays page N** |
| **Native PDF** | Only embedded **image streams** are recompressed; **text, fonts and vectors per page are untouched**; page order preserved | Text layer (still selectable/searchable), vectors, layout, page order |
| **DOCX / XLSX** | Only images in `media/*` are recompressed; **all XML/text/sheets/styles/relationships are byte-identical** | Text, formulas, tables, styles, formatting, structure |

**No context switching (OCR is never confused):**
- **OCR reads the ORIGINAL** file (the on-disk temp file); the compressed copy goes only to storage and is **never fed to OCR**.
- The OCR **prompt is chosen per image / per scanned-page from the original** (`_detect_image_type` → INVOICE / TABLE / GRAPH / FORM / DOCUMENT), so the prompt for page *N* always matches page *N*'s original content.
- The native-vs-scanned **classifier** that decides the prompt *route* (library vs Gemini) is shared but **byte-identical** to the old logic — so routing never changes.
- No OCR module imports the compression code; the two paths are fully independent. **Result: zero per-page mismatch and zero context bleed between original and compressed.**

---

## 14. UI surfacing — the upload chip now reflects the compressed size

Earlier the feature was silent in the UI (§6). It now surfaces the real stored size, end-to-end.

### 14.1 Response contract — the backend returns the sizes

`UploadResponse` (Genie `POST /api/v1/upload`) gained two optional fields so the client can show the result of compression:

| Field | Meaning |
|---|---|
| `original_size` | uploaded byte size (before compression) |
| `compressed_size` | stored byte size (after compression; **== `original_size`** when nothing shrank) |

- Populated from the `CompressionResult` that `_compress_and_persist_original()` now **returns** (it previously returned `None`). The endpoint threads `result.original_size` / `result.stored_size` into the response.
- Both `null` when the upload had **no session** (compression skipped, e.g. legacy mode).

### 14.2 What the chip shows (logic)

| State | Size shown on the chip |
|---|---|
| Uploading / processing | **original** size (compressed size not known yet) |
| Completed **and** `compressed_size < original_size` | **compressed** size — the number simply swaps |
| Completed, not shrunk / < 500 KB | original size (unchanged) |

- **Deliberately minimal:** only the single size number updates — **no** strikethrough, arrow, or "−%" badge (product decision: keep the card clean).
- FE wiring: `use-chat.ts` sets `FileAttachment.compressedSize` on completion **only when the stored size is actually smaller**; `file-attachment.tsx` renders `compressedSize ?? fileSize`.

### 14.3 File-type icons — every accepted type is now distinct

The chip icon + colour derive from the file extension. Two issues were fixed: `docx/txt/md/tex` previously fell to a generic grey icon, **and** a wiring bug meant `getFileTypeVisuals`' type colours were dead code — every *completed* file showed a green icon regardless of type.

| Type | Icon | Colour (when completed) |
|---|---|---|
| pdf | `FileText` | red |
| png/jpg/jpeg/gif/bmp/webp/tiff/tif | `Image` | blue |
| xlsx/xls/csv | `FileSpreadsheet` | green |
| **docx/doc** | `FileType` | indigo |
| **md/tex** | `FileCode` | purple |
| **txt** | `FileText` | slate |

During upload / processing / error the icon uses the **status** colour (spinner / error); on completion it uses the **file-type** colour.

---

## 15. Structured logging & traceability

The upload path's logging was migrated from ~75 f-string "banner" lines (`====`, `----`, one field per `logger.info`) to a handful of **structured events** — event name as the message, context in `extra={}` (the repo logging idiom). Each event is one readable line in the dev terminal and a fully queryable record in GCP Cloud Logging.

### 15.1 Events emitted per Genie upload (in order)

| Event | When | Key fields |
|---|---|---|
| `upload.processing_start` | file received | `file_name`, `file_bytes`, `file_type`, `session_id`, limits |
| `compression.decision` / `compression.fallback` | compress step | §15.2 / `reason` |
| `gcs.upload` | stored to GCS | `path`, `bytes`, `bytes_human`, `status` |
| `upload.processing_complete` | OCR done | `file_id`, `document_type`, `confidence` |
| `ocr.extraction_summary` | extraction details | `extraction_method`, `processing_time_ms`, `text_chars`, `tables` |
| `ocr.output_saved` ×3 | each artifact | `kind` (json/markdown/text), `path`, `bytes` |
| `upload.response_summary` | final | all output paths + timings |
| `ocr.gemini_cost` | only if paid OCR ran | `model`, tokens, `cost_usd` |

### 15.2 The compression markers (before → after)

`compression.decision` (a file actually shrank) carries raw bytes **and** human-readable sizes, plus a single at-a-glance marker:

```
compression.decision | strategy=image surface=genie
  original_bytes=1992294 stored_bytes=177601 saved_bytes=1814693 saved_pct=91.1
  original_human=1.90 MB stored_human=173.4 KB saved_human=1.73 MB
  summary=1.90 MB -> 173.4 KB (-91.1%)  duration_ms=…
```

`compression.fallback` (skipped) carries `reason` + `original_human`. **All benign reasons** (`below_min, disabled, not_eligible, size_regression, animated, passthrough, no_media, unsupported_image`) log at **INFO**; only a genuine `error` logs at **WARNING** (with an `error=` field) — so a dashboard alert on `severity=WARNING AND message=compression.fallback` fires only on real failures.

### 15.3 What to search

```bash
# terminal (dev StandardFormatter) or any log sink
grep "compression.decision"   # → summary=1.90 MB -> 173.4 KB (-91.1%) + strategy=…
grep "compression.fallback"   # → reason=below_min / size_regression / …
grep "gcs.upload"             # → path + bytes_human + status
grep -E "upload\.|ocr\."      # → the whole upload lifecycle
```

GCP Cloud Logging: `jsonPayload.message="compression.decision"`; trace one upload with `jsonPayload.session_id="…"`; real failures only with `severity="WARNING" AND jsonPayload.message="compression.fallback"`. Every `extra={}` field is its own indexed `jsonPayload.*` key, and `request_id` / `tenant_id` / `user_id` are auto-injected for correlation.

### 15.4 Safety fixes shipped with the refactor

- **Crash fix:** `extra={"filename": …}` collided with a reserved `LogRecord` attribute and raised `KeyError` on **every** upload → renamed to `file_name` at all 4 call sites; a new **AST guard test** (`test_upload_logging.py`) statically forbids any reserved key in a logger `extra`, so the bug class can't recur.
- **PII:** the document-text preview moved from INFO → **DEBUG** (extracted invoice text can carry names / GSTINs); the full text is still persisted to the `.txt` artifact.
- **Dev parity:** `StandardFormatter` now appends the redacted `extra={}` fields to each line, so local terminal logs are as informative as the GCP JSON ones.

---

## 16. Complete configuration, limits & behavior reference

One-stop lookup for **every** setting, limit, trigger, what stays unchanged, and where files land. All values verified against the code (`app/config.py`, `compression/config.py`, `upload.py`, FE `constants.ts`).

### 16.1 Switches — master + per-surface (what turns it on)

| Env var | Default | What it is | Triggers when |
|---|---|---|---|
| `COMPRESSION_ENABLED` | **`false`** | Master kill-switch | Must be `true` or the whole feature is a runtime no-op |
| `COMPRESSION_GENIE_ENABLED` | `true` | Genie / chat (OCR) surface | Active only when master on |
| `COMPRESSION_ERPHOME_ENABLED` | `true` | ERPHome attachment surface | Active only when master on |

### 16.2 Compression targets — what it shrinks down to

| Env var | Default | Controls | Applies to |
|---|---|---|---|
| `COMPRESSION_MIN_BYTES` | `512000` (**500 KB**) | Floor — below this is never compressed | all types |
| `COMPRESSION_IMAGE_TARGET_BYTES` | `204800` (**200 KB**) | Image size target | images |
| `COMPRESSION_IMAGE_MAX_DIMENSION` | **`1200`** px | Resize longest edge above this | images |
| `COMPRESSION_IMAGE_JPEG_QUALITY` | **`80`** | JPEG start quality → ladder `80 → 65 → 50 → 35` | images / scanned pages |
| `COMPRESSION_SCANNED_PDF_TARGET_BYTES_PER_PAGE` | `307200` (**300 KB/pg**) | Per-page size target | scanned PDF |
| `COMPRESSION_SCANNED_PDF_DPI` | **`150`** | Raster DPI for rebuild (OCR rasters at 200, separately) | scanned PDF |
| `COMPRESSION_NATIVE_REBUILD_ENABLED` | **`false`** | Recompress embedded images in heavy DOCX/XLSX/native-PDF | native files |

### 16.3 Upload limits — acceptance gate (reject *before* compression)

| Limit | Value | Where enforced | Over → |
|---|---|---|---|
| Per-file size | **10 MB** | `upload.py` `MAX_FILE_SIZE` + FE `OCR_MAX_FILE_SIZE` | rejected (413 / toast) |
| Session total | **30 MB** | `MAX_TOTAL_SIZE` / `OCR_MAX_TOTAL_SIZE` | rejected |
| Files per session | **3** | `MAX_FILES_PER_SESSION` / `OCR_MAX_FILES` | rejected |
| PDF pages | **10** | `MAX_PAGES_PER_PDF` (fitz) | rejected |
| Rate limit | **10 / min** | `create_rate_limit_dependency(10)` | 429 |

> ERPHome attach enforces **10 MB/file only** — no session-total or file-count cap.

### 16.4 Security caps — defense-in-depth (→ fallback to original)

| Env var | Default | Guards against |
|---|---|---|
| `COMPRESSION_MAX_IMAGE_PIXELS` | `64000000` (**64 MP**) | image/pixel bomb (standalone **and** images embedded in DOCX/XLSX) |
| `COMPRESSION_MAX_PDF_PAGES` | **`50`** | huge PDFs + per-page raster bound (DPI × page-area ≤ pixel cap) |
| `COMPRESSION_MAX_OOXML_ENTRIES` | **`5000`** | zip-bomb (member count, read from ZIP directory) |
| `COMPRESSION_MAX_OOXML_UNCOMPRESSED_BYTES` | `419430400` (**400 MB**) | zip-bomb (total inflate) |

### 16.5 Validator clamps — the settings themselves can't be set unsafely

Pydantic validators in `app/config.py` clamp each knob so a typo/misconfig can never disable a guard or invert behavior:

| Setting | Clamp | Why |
|---|---|---|
| `image_jpeg_quality` | **10 – 95** | keep JPEG in a sane range |
| `image_max_dimension` | floored to **≥ 256** | never resize to a useless size |
| `scanned_pdf_dpi` | **72 – 300** | readable but bounded raster |
| `max_image_pixels` / `max_pdf_pages` / `max_ooxml_entries` | floored to **≥ 1** | a `0`/negative cap must not silently disable a guard |
| `max_ooxml_uncompressed_bytes` | floored to **≥ 1 MB** | keep the zip-bomb cap meaningful |
| `min_bytes` | floored to **≥ 0** | no negative floor |

### 16.6 The trigger — WHEN it actually compresses (all must pass, in order)

| # | Check | Pass | Else → reason logged |
|---|---|---|---|
| 1 | master + surface on | both `true` | `disabled` → as-is |
| 2 | size | **≥ 500 KB** | `below_min` → as-is |
| 3 | eligible type | image / scanned PDF (native only if flag) | `not_eligible` → as-is |
| 4 | not animated | single-frame | `animated` → as-is |
| 5 | under security caps | §16.4 | exceeds → **fallback to original** |
| 6 | actually smaller | re-opens **and** `< original` | `size_regression` / `error` → **fallback to original** |

**One-liner:** an eligible **image or scanned PDF ≥ 500 KB** (or a flagged heavy native file).

### 16.7 What STAYS THE SAME / stored as-is (never compressed)

| Case | Why (reason) |
|---|---|
| File **< 500 KB** | not worth it (`below_min`) |
| **CSV / TXT / MD / TEX / pure-text PDF** | text already efficient (`not_eligible`) |
| **gif / bmp / webp / tiff** images | extensions we don't re-encode (`unsupported_image`) |
| **Animated** GIF/WebP | out of scope (`animated`) |
| **DOCX / XLSX / native PDF** | as-is unless `NATIVE_REBUILD_ENABLED=true` |
| Compressed copy **not smaller** | keep original (`size_regression`) |
| Any **error / corrupt / encrypted** | keep original (`error`, WARNING) |
| **Filename + extension** | **always unchanged** (`invoice.jpg` stays `invoice.jpg`) |
| **Company logo / onboarding import files** | `compress=False` / separate endpoints |

### 16.8 WHERE it saves (one store per surface)

| Surface | Store | Path |
|---|---|---|
| **Genie OCR** | **GCS only** | `gs://<bucket>/sessions/{session_id}/originals/{filename}` + OCR `.json/.md/.txt` |
| **ERPHome attach** | **ERPNext Frappe File only** | `/api/method/upload_file` (never GCS) |
| Temp OCR file | local OS temp | `/var/folders/.../tmpXXXX` → **deleted** after OCR (`finally`) |
| No `session_id` | — | original **not persisted** (compression skipped) |

### 16.9 The result (response + UI + log)

| Output | Value |
|---|---|
| `UploadResponse` | `original_size`, `compressed_size` (both `null` if no session) |
| UI chip | size number swaps to the compressed size once done (e.g. `1.9 MB → 177.6 KB`) |
| Log on shrink | `compression.decision … summary=1.90 MB -> 173.4 KB (-91.1%)` |
| Log on skip | `compression.fallback … reason=below_min` (INFO; only `error` = WARNING) |
| Storage write | `gcs.upload … bytes_human=173.4 KB status=ok` |

**To enable in dev:** set `COMPRESSION_ENABLED=true` (GCS auto-selected when a bucket is configured; no new IAM). Staged rollout: dev → soak → staging/prod.
