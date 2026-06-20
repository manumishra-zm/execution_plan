# ERPSense — Auto-Compression: Implementation Plan

> Generated 2026-06-14. Single source of truth for implementing silent auto-compression across **Genie Chat (OCR)** and **ERPHome (attachments)** without breaking existing functionality. Designed for **one autonomous pass** (no weekly phases). Architecture diagram: `auto_compresion/auto_compression_pipeline.excalidraw`. OCR ground truth: `auto_compresion/ocr_active_flow_complete.md`.

## Executive Summary

Auto-compression for the Genie chat OCR upload surface and the ERPHome attachment surface can be delivered in one pass without touching OCR behavior, because both surfaces already expose a single, clean in-memory-bytes seam (`content` at `upload.py:864` feeding `_save_original_file` at `upload.py:1024`; `file_content` feeding `upload_attachment`/`upload_file_standalone` in `_attachments.py`). The work is a new, coverage-counted compression module that operates on a *copy* of those bytes, plus structured logging/metrics and frontend transparency — not a re-architecture. Storage stays strictly per-surface (Genie to GCS via the single `upload_bytes` write; ERPHome to ERPNext only), the size limits are already 30MB on both Genie FE and BE (the "10MB" is unrelated surfaces), and every compression path falls back to the original bytes on failure so OCR and uploads can never be blocked.

### Goals
- Compress images (target <=200KB, <=1200px) and scanned PDFs (target <=300KB/page) on a copy of the in-memory bytes, before persisting, on both the Genie OCR surface and the in-scope ERPHome attach callers.
- Genie: persist the compressed bytes through the existing single `upload_bytes` write in `_save_original_file` (`upload.py:258`); keep filename + extension unchanged so the suffix-derived preview content-type and the cancel/purge matcher keep working.
- OCR keeps reading the untouched original `temp_path` (`upload.py:1044`); storage saves the compressed copy.
- ERPHome: compress the in-memory `file_content` copy at the adapter seam (covers all 5 in-scope callers in one place) and pass it as the multipart tuple with matching content_type; persist to ERPNext only.
- Full observability: structured event logs (`compression.decision`, `compression.fallback`, `gcs.upload`) mirroring `diagnostic_metrics.py`, plus an optional Prometheus `_init_compression_metrics()` family; replace the f-string/banner logs at the persist seam with compliant `extra={}` events.
- A shared `is_native_pdf` classifier delegated from both existing call sites (preserving each site's total-vs-average semantics and raise-vs-branch control flow) so the compression "native vs scanned" verdict stays in lock-step with OCR.
- Tests (unit + api) for compress, fallback-on-failure, persist-compressed, preview-still-works, ERPHome adapter hook, and cross-tenant BOLA; new logic placed in non-omitted modules so it counts toward the 75% backend / 85%-new gates.
- Works in DEV with GCS configured (factory selects GCS when a bucket is set).

### Non-goals
- No background worker / queue. Execution is synchronous inline within the request (`asyncio.gather` + `asyncio.to_thread` for "parallel"); true async infra is net-new and out of scope.
- No second storage sink per surface. Genie does NOT also write to ERPNext; ERPHome does NOT write to GCS. The import/branding/FileStorageService GCS sinks are untouched.
- No compression of native PDF / DOCX / XLSX / CSV. Container/PDF rebuild for native files is optional, flag-guarded, and off by default.
- No MIN-500KB-exempt behavior change beyond a threshold (files under the floor are not compressed; this is net-new, exempt).
- No switch to signed URLs (the helper is dead and would 403 without `signBlob`); preview stays a byte proxy.
- No lowering of any 30MB limit to 10MB; no change to the ERPHome 10MB cap (`_shared.py:111`) absent explicit product intent.
- No live-GCS unit test harness (conftest pins Local); real-GCS confidence is via dev smoke.
- No FE compression; FE changes are additive transparency only.

### Key decisions

| Decision | Choice | Why |
|---|---|---|
| Storage per surface | Genie -> GCS only (single `upload_bytes` at `upload.py:258`); ERPHome -> ERPNext only (adapter multipart) | The two sinks are independent; adding a second write conflates features and the dup-filename suffixer (`upload.py:249-254`) would create a `_1` duplicate. |
| Compress on a copy | Always compress a copy of in-memory `content`/`file_content`; never `temp_path` | `ImageHandler._resize_if_needed` overwrites `temp_path` in place before OCR (`image_handler.py:162`); OCR + PDF page-count validation read `temp_path`. Mutating it corrupts OCR. |
| Scope | Images + scanned PDFs now; native PDF/DOCX/XLSX/CSV pass through; container/PDF rebuild flag-guarded, off | Native files are already conf-1.0 / $0 and small; rebuild adds risk for little gain. `is_native_pdf` classifier gates the scanned-PDF branch. |
| 30MB vs 10MB | Genie stays 30MB/32MB/5/10 on BE (`upload.py:81-84`) and FE (`constants.ts:1893-1896`) — already aligned; ERPHome stays 10MB (`_shared.py:111`) | The only "10MB" on the Genie path doesn't exist; the FE 10MB strings are unrelated surfaces (BRS, FileStorage). No reconciliation needed beyond confirming agreement. |
| Fallback | On compression error OR compressed >= original: store the ORIGINAL bytes, no retry, log at WARNING with `fallback_reason`; never raise | `_save_original_file` is non-blocking (returns None on failure); a compression failure must never block OCR or the upload. |
| Execution model | Synchronous inline; "parallel" = `asyncio.gather` + `asyncio.to_thread` inside the request | No worker exists; CPU-bound Pillow/PyMuPDF work runs in a thread to avoid blocking the event loop. |
| Dev-env GCS | Rely on `get_storage_client()` selecting GCS when a bucket is configured; ADC auth; no new IAM (objectCreator+objectViewer cover upload/download/exists) | Decision 10: must work in DEV with GCS, not only local. The suffixer keeps writes to fresh paths so objectCreator (create-only) suffices. |
| Default-off kill switch | Compression behind a `Settings` flag (config.py), defaulting OFF for the optional native rebuild and a master on/off for the feature; thresholds (KB/px) in `Settings`, not `os.getenv` | Safe rollout + instant disable; honors the "no os.getenv outside config.py" (R03) rule. |

### Build order
1. New `app/services/file_processing/compression.py` (non-omitted): pure compress functions for image and scanned-PDF on a BytesIO copy, returning `(bytes, strategy, decision)` with same-extension/content-type guarantees; `Settings` flags + thresholds in `config.py`.
2. Shared `is_native_pdf` helper (`handlers/_pdf_classifier.py`); delegate from `PDFHandler._has_sufficient_text` and `library_gemini._extract_pdf_native`, preserving each site's semantics and control flow.
3. `compression_metrics.py` (mirror `diagnostic_metrics.py`) for `compression.decision`/`compression.fallback`/`gcs.upload`; optional `MetricsRegistry._init_compression_metrics()` wired into `__post_init__` + `reset_metrics()`.
4. Genie wiring: compute compressed bytes after validation (post `upload.py:1019`), pass into `_save_original_file` at the call site (`1024-1025`); replace the f-string/banner persist logs with structured `upload.original_saved`; OCR untouched at `1044`.
5. ERPHome wiring: adapter-level hook in `upload_attachment`/`upload_file_standalone` with `compress: bool = True`; pass `compress=False` from the provisioning logo push (`_steps_company.py:335`); fall back to original within each method's existing success envelope.
6. Backend tests: unit (compress, fallback-on-OSError, ratio) beside `test_storage_client.py`/`test_brs_gcs_upload.py`; api (Genie persist-compressed + preview via `test_upload_cancel.py` pattern; ERPHome adapter via respx; cross-tenant == 404).
7. Frontend (additive): optional `original_size`/`stored_size`/`was_compressed` on `UploadResponse`/`ExtractionMetadata`; surface in chat `file-preview-modal.tsx` header with i18n; tests in `tests/unit/lib/api/upload.test.ts`.
8. Per-pass gates: backend `ruff` + `mypy app/` + `pytest --cov-fail-under=75`; frontend `format`/`lint`/`type-check`/`test:coverage`; repeated `/code-review` and `/security-review`.

## Table of Contents

- [1. Architecture & Corrected Principles](#1-architecture-corrected-principles)
- [2. Tech Stack, Dependencies & Versions](#2-tech-stack-dependencies-versions)
- [3. Configuration, Feature Flags & Limit Reconciliation](#3-configuration-feature-flags-limit-reconciliation)
- [4. GCS Connection, Storage Layer & Dev-Env Setup](#4-gcs-connection-storage-layer-devenv-setup)
- [5. Compression Engine: Module Structure & Interfaces](#5-compression-engine-module-structure-interfaces)
- [6. Core Compressors: Image + Scanned PDF](#6-core-compressors-image-scanned-pdf)
- [7. Optional Container Rebuild Engine (DOCX / XLSX / native-PDF) [flagged, default OFF]](#7-optional-container-rebuild-engine-docx-xlsx-nativepdf-flagged-default-off)
- [8. Genie OCR-Flow Integration (upload.py)](#8-genie-ocrflow-integration-uploadpy)
- [9. ERPHome Attachment + OCR-Attach Entity Integration](#9-erphome-attachment-ocrattach-entity-integration)
- [10. Preview / Download & On-Demand Unpack](#10-preview-download-ondemand-unpack)
- [11. Observability: Logging & Metrics (incl. every GCS op)](#11-observability-logging-metrics-incl-every-gcs-op)
- [12. Security Review & Hardening (repeated gates)](#12-security-review-hardening-repeated-gates)
- [13. Edge Cases & Robustness Catalog](#13-edge-cases-robustness-catalog)
- [14. Testing Strategy & Test Files (unit/integration/e2e/perf)](#14-testing-strategy-test-files-unitintegratione2eperf)
- [15. Dev-Environment Rollout, Backward-Compat & Kill Switch](#15-devenvironment-rollout-backwardcompat-kill-switch)
- [16. Autonomous One-Pass Execution & Verification Gates](#16-autonomous-onepass-execution-verification-gates)

---

All facts confirmed. There's a `classifiers/` subpackage in `file_processing/` — a natural home for the shared classifier. I now have a complete, verified picture and will write the plan section.

## 1. Architecture & Corrected Principles

This feature adds **automatic, server-side file compression** to ERPSense's two independent upload surfaces, plus the supporting GCS connection hardening, structured observability, preview correctness, and the ERPNext+OCR attach-entity wiring — all in one pass. The design is governed by a single overriding rule learned from the seams: **the bytes we run OCR on and the bytes we persist are decoupled.** OCR always reads the untouched original on the temp path; storage always receives a compressed copy (or the original on fallback). Everything else follows from keeping those two byte-streams separate and per-surface.

### 1.1 The two surfaces (never conflate them)

There are exactly two upload surfaces, with **per-surface storage sinks** (corrected decision 5):

| | Surface A — Genie Chat OCR | Surface B — ERPHome attachment |
|---|---|---|
| Route | `POST /api/v1/upload` | `POST /api/v1/erp/{doctype}/{name}/attachment`, `POST /api/v1/erp/files/upload` |
| Handler | `upload.py:upload_file()` (650-1363) | `doc_social.py:upload_attachment` / `file_proxy.py:upload_erp_file` |
| Adapter sink | `storage_client.get_storage_client()` → **GCS only** (Local in dev w/o bucket) | `_attachments.py:AttachmentsMixin.upload_attachment` / `upload_file_standalone` → **ERPNext Frappe File only** |
| OCR? | **Yes** — `file_router.process(temp_path)` synchronous inline | **No** OCR ever |
| Hard size cap | `MAX_FILE_SIZE=30MB` (`upload.py:81`) | `_MAX_ATTACHMENT_SIZE_BYTES=10MB` (`_shared.py:111`) |
| Compression copy source | in-memory `content` (`upload.py:864`) | in-memory `file_content` arg |

Decision 5 is load-bearing: **do not add a GCS write to Surface B and do not add an ERPNext write to Surface A.** The `attachments.py`/`FileStorageService` (GCS+DB), `import_files/upload.py` (onboarding master-data), and `files.py`/`BrandingService` (logos) sinks are explicitly **out of scope** and must not gain compression hooks.

### 1.2 Non-negotiable invariants

These hold on **both** surfaces and are enforced by tests + code review every pass:

1. **OCR reads the original; storage saves the compressed copy.** On Surface A, the OCR call at `upload.py:1044-1047` reads `temp_path` (written from raw `content` at 948-949) and must remain untouched. Compression operates on a **copy of in-memory `content`/`file_content`**, never on `temp_path` (corrected decisions 1-2). This is critical because `ImageHandler._resize_if_needed` (`image_handler.py:162`) **overwrites `temp_path` in place** for images >4096px *before* OCR — compressing the temp file would corrupt OCR input and double-process.
2. **Same filename + same extension** after compression (corrected decision 4). The Genie preview content-type is suffix-derived (`serve_original_file` → `MIME_TYPE_MAP` at `upload.py:1908-1924,1986-1987`), the FE preview routing keys off extension (`chat-utils.ts:buildPreviewUrl` + `EXT_TO_MIME`), and the ERPNext multipart tuple's 3rd element carries `content_type`. A JPEG stays `.jpg`/`image/jpeg`; a PDF stays `.pdf`/`application/pdf`. Never rename, never change extension.
3. **Single write per surface.** On Surface A, overwrite the **one** `upload_bytes` call inside `_save_original_file` (`upload.py:258`) — do **not** add a second write (corrected decision 3). The dup-filename suffixer (`upload.py:249-254`) would otherwise rename the compressed copy to `{stem}_1{suffix}` and break the cancel-purge matcher. On Surface B, compress the bytes just before the existing `files = {...}` multipart build (`_attachments.py:108`/`:196`).
4. **Fallback never raises, never blocks OCR** (corrected decision 9). If compression errors OR `len(compressed) >= len(original)`, persist the **original** bytes, log a WARNING with `fallback_reason`, and continue. `_save_original_file` is already non-blocking (returns `None` on failure, OCR proceeds); the compression layer must preserve that contract. No retry.
5. **Scope = images + scanned PDFs only** (corrected decision 7). Compress images (target `<=200KB`, `<=1200px` longest edge) and scanned PDFs (target `<=300KB/page`). Native PDF/DOCX/XLSX/CSV/TXT/MD pass through **as-is**. Native-vs-scanned must agree with what OCR considers native/scanned, so the decision reuses the OCR classifier (§1.4).
6. **MIN 500KB exempt; MAX is 30MB (Genie) / 10MB (ERPHome)** (corrected decision 8). Files `<500KB` skip compression entirely (net-new threshold). The "10MB" in ground truth was the *advisory* `RECOMMENDED_LIMITS`/ERPHome cap — the Genie hard cap is genuinely 30MB and the FE already matches it (`constants.ts:1893`), so frontend reconciliation is a no-op except adding optional transparency fields.
7. **Works in dev with GCS configured** (corrected decision 10). `get_storage_client()` (`storage_client.py:400-418`) selects `GCSStorageClient` when `gcs_bucket_name or gcs_upload_bucket` is set; the compression output flows through the identical `upload_bytes(path, data, content_type)` contract, so GCS-in-dev needs no new code path — but a dev smoke must **assert the GCS client was actually selected** (Local degrades silently with only a WARNING).

### 1.3 Component map — files to CREATE

All compression logic lives in **new, non-omitted modules** so it counts toward the 75% coverage gate (`upload.py`, `library_gemini.py`, `pdf_handler.py`, `image_handler.py` are all in `pyproject.toml [tool.coverage.run].omit` at lines 127,140,141,145 — code placed there is invisible to coverage).

| New file (full path) | Responsibility | Key symbols |
|---|---|---|
| `app/services/file_processing/compression/__init__.py` | Package re-exports | `compress_for_storage`, `CompressionResult`, `CompressionDecision` |
| `app/services/file_processing/compression/compressor.py` | Pure orchestrator: classify → route → compress copy → fallback. **No I/O on temp_path, no storage calls.** | see signature below |
| `app/services/file_processing/compression/image_compressor.py` | Pillow resize-to-`<=1200px` + re-encode to target `<=200KB`; RGBA/P→RGB conversion mirroring `image_handler.py:159-161`; preserve format/extension | `compress_image(data, suffix, *, max_px, target_bytes)` |
| `app/services/file_processing/compression/pdf_compressor.py` | Scanned-PDF rebuild via **PyMuPDF (fitz) raster at 200 DPI + Pillow JPEG + reportlab** (pikepdf/img2pdf NOT available); target `<=300KB/page`; guarded behind off-by-default flag | `compress_scanned_pdf(data, *, target_bytes_per_page)` |
| `app/services/file_processing/compression/metrics.py` | Log-based + registry telemetry (mirrors `diagnostic_metrics.py` + `metrics.log_form_generated`) | `record_compression_decision`, `record_compression_fallback`, `record_gcs_upload` |
| `app/services/file_processing/classifiers/pdf_classifier.py` | Shared `is_native_pdf(text, page_count, *, mode)` delegate (§1.4) | `is_native_pdf`, `avg_chars_per_page` |
| `tests/unit/file_processing/test_compression.py` | Compressor unit tests (decision matrix, fallback, copy-not-temp, ext-preserved) | — |
| `tests/unit/file_processing/test_pdf_classifier.py` | Both classifier semantics (total vs avg) | — |
| `tests/api/test_upload_compression.py` | Genie surface: persist-compressed, OCR-reads-original, fallback, preview round-trip | — |
| `tests/api/v1/test_erp_attachment_compression.py` | ERPHome surface via respx; provisioning-logo opt-out; BOLA `==404` | — |
| FE: `tests/unit/lib/api/upload.test.ts` (extend) + `tests/unit/lib/utils/` | 30MB agreement + optional `was_compressed` field | — |

Compressor entrypoint signature (synchronous-pure core, called via `asyncio.to_thread`):

```python
# compression/compressor.py
@dataclass(frozen=True)
class CompressionResult:
    data: bytes              # compressed OR original (on fallback)
    original_size: int
    stored_size: int
    was_compressed: bool
    decision: str            # "image" | "scanned_pdf" | "skip_native" | "skip_small" | "skip_unsupported"
    fallback_reason: str | None
    strategy: str            # "pillow" | "pymupdf_reportlab" | "none"

def compress_for_storage(
    content: bytes, *, filename: str, content_type: str | None,
    settings: CompressionSettings,        # thresholds from app/config.py
) -> CompressionResult: ...
```

`compress_for_storage` **never raises** — it catches internally and returns a `CompressionResult` with `was_compressed=False` + `fallback_reason`. The async wrapper used at call sites:

```python
# compression/__init__.py
async def compress_async(content, *, filename, content_type, settings) -> CompressionResult:
    return await asyncio.to_thread(compress_for_storage, content,
        filename=filename, content_type=content_type, settings=settings)
```

### 1.4 Shared native-vs-scanned classifier (no OCR behavior change)

The two existing classifiers have **different semantics and control flow** and must both keep their behavior:
- `PDFHandler._has_sufficient_text` (`pdf_handler.py:308`): `len(text) >= MIN_TEXT_LENGTH * page_count` (**total** chars), **branches** to OCR.
- `LibraryGeminiProcessor._extract_pdf_native` (`library_gemini.py:613-617`): `avg = total/page_count < MIN_PDF_TEXT_PER_PAGE` (**per-page average**), **raises** `FileProcessingException`.

New `pdf_classifier.is_native_pdf(text, page_count, *, mode)` parameterizes on `mode` (`"total"` vs `"average"`) and threshold so each call site keeps its exact verdict. Wire as **thin delegates**: replace the body of `_has_sufficient_text` with `return is_native_pdf(text, page_count, mode="total", threshold=self.MIN_TEXT_LENGTH)` (keep the method + its branch), and replace the inline `avg < MIN` check in `_extract_pdf_native` with the helper (keep the `raise`). Do **not** change `MIN_TEXT_LENGTH`/`MIN_PDF_TEXT_PER_PAGE` values, the 200 DPI, or the raise-vs-branch flow. The compressor's PDF-vs-image-vs-native decision calls the **same** helper so "compress only scanned PDFs" stays in lock-step with OCR's verdict.

### 1.5 Component map — files to MODIFY

| File:symbol | Change | Invariant honored |
|---|---|---|
| `upload.py:_save_original_file` (225-264) | Add optional `compressed: CompressionResult \| None` param; persist `result.data` at the **single** `upload_bytes` (258); replace the f-string log (259) with `logger.info("upload.original_saved", extra={...sizes,strategy,decision,storage_backend})` | 2,3 |
| `upload.py:upload_file` call site (1024-1025) | After page-validation (ends 1019), before persist: `result = await compress_async(content, filename=original_filename, content_type=file.content_type, settings=...)`; pass `result` into `_save_original_file`. **OCR at 1044 still reads `temp_path` (raw).** Replace banner logs 1027-1041 with structured event | 1,4 |
| `_attachments.py:upload_attachment` (77-168) | Add `compress: bool = True`; before multipart (108) compress a copy of `file_content`; fall back on failure; extend the success log (`extra`) with sizes | 1,4,9 |
| `_attachments.py:upload_file_standalone` (170-233) | Same hook before multipart (196); same `compress` kwarg | 1,4,9 |
| `_steps_company.py` logo push (335) | Pass `compress=False` (logo already optimized by `BrandingService`) | scope |
| `pdf_handler.py:_has_sufficient_text` (302-308) | Delegate to shared classifier (mode="total") | OCR-unchanged |
| `library_gemini.py:_extract_pdf_native` (613-617) | Delegate to shared classifier (mode="average"), keep `raise` | OCR-unchanged |
| `app/config.py` Settings | Add `compression_*` knobs (NOT `os.getenv`): `compression_enabled=True`, `compression_min_bytes=500*1024`, `compression_image_target_bytes=200*1024`, `compression_image_max_px=1200`, `compression_scanned_pdf_target_bytes_per_page=300*1024`, `compression_pdf_rebuild_enabled=False` (off-by-default container rebuild, decision 7) | config-only knobs |
| `app/core/metrics.py` | Add `MetricsRegistry._init_compression_metrics()` (Counters `compression_decisions_total{strategy,decision}`, `compression_fallback_total{reason}`, `gcs_upload_total{backend,status}`; Histograms `compression_bytes_saved`, `gcs_upload_latency_seconds`); call from `__post_init__` (490) **and** `reset_metrics` (980) | telemetry |
| `pyproject.toml [tool.coverage.run].omit` | Ensure new `compression/` + `classifiers/pdf_classifier.py` are **NOT** added to omit | coverage |
| FE `src/types/chat.ts` (`ExtractionMetadata`/`UploadResponse` 442-464) | Add optional `original_size?`, `stored_size?`, `was_compressed?` (additive only) | transparency |
| FE `components/chat/file-preview-modal.tsx` (164-167) | Show "compressed from X" when present | transparency |

### 1.6 How existing behavior stays unbroken

- **OCR path is byte-identical**: `temp_path` is never touched by compression; classifier refactor is a pure delegate with preserved semantics/control-flow → no OCR routing change.
- **Cancel/purge keeps working**: same filename+extension means `_purge_upload_artifacts` (`upload.py:1366-1510`) still matches `{stem}.{ext}`/`{stem}_{n}.{ext}`.
- **Preview keeps working**: suffix-preserved → `MIME_TYPE_MAP` resolves; FE blob-fetch path (`useAuthedFileUrl`) unchanged.
- **GCS least-privilege respected**: still one create-only `upload_bytes` to a fresh path (objectCreator), no overwrite/delete/signBlob added; byte-proxy preview retained (no dead signed-URL path activated).
- **Fallback contract**: compression failure → original persisted, OCR proceeds — same observable result as today, plus a WARNING.
- **Telemetry is additive**: log-based events (primary) + optional registry counters at `GET /api/v1/metrics`; storage_client's `%`-style DEBUG logs untouched (structured logs emitted at the caller seam, keeping storage transport-only).

### 1.7 Diagram

See the excalidraw diagram `compression-architecture` (two-surface flow: Surface A `content`→`compress_async`→`_save_original_file`→GCS, with `temp_path`→OCR drawn as a parallel untouched branch; Surface B `file_content`→compress-copy→multipart→ERPNext; shared `pdf_classifier` node feeding both OCR classifiers and the compressor; fallback edges from the compressor back to "store original"; config knobs + metrics emitters annotated at each seam).

### 1.x Checklist

- [ ] Create `app/services/file_processing/compression/__init__.py` with `compress_async` + re-exports
- [ ] Create `compression/compressor.py` with `compress_for_storage` (never-raises, returns `CompressionResult`)
- [ ] Implement decision matrix in compressor: skip if `<500KB`, skip native PDF/DOCX/XLSX/CSV/TXT/MD, route images, route scanned PDF (via shared classifier)
- [ ] Create `compression/image_compressor.py`: RGBA/P→RGB, resize longest edge `<=1200px`, re-encode to `<=200KB`, preserve format+extension
- [ ] Create `compression/pdf_compressor.py`: PyMuPDF 200 DPI raster + Pillow JPEG + reportlab rebuild, `<=300KB/page`, guarded by `compression_pdf_rebuild_enabled` (default False)
- [ ] Create `compression/metrics.py` mirroring `diagnostic_metrics.py` (`record_compression_decision/_fallback`, `record_gcs_upload`)
- [ ] Create `app/services/file_processing/classifiers/pdf_classifier.py` with `is_native_pdf(text, page_count, *, mode, threshold)` + `avg_chars_per_page`
- [ ] Wire `PDFHandler._has_sufficient_text` (`pdf_handler.py:308`) to delegate (mode="total"), keep branch behavior
- [ ] Wire `LibraryGeminiProcessor._extract_pdf_native` (`library_gemini.py:613-617`) to delegate (mode="average"), keep `raise FileProcessingException`
- [ ] Add `compression_*` Settings to `app/config.py` (enabled, min_bytes=500KB, image_target=200KB, image_max_px=1200, scanned_pdf_per_page=300KB, pdf_rebuild_enabled=False)
- [ ] Modify `upload.py:_save_original_file` (225-264): accept `CompressionResult`, persist `result.data` at single `upload_bytes` (258), replace f-string log with `logger.info("upload.original_saved", extra=...)`
- [ ] Modify `upload.py:upload_file` call site (1024-1025): compress copy of `content` after page-validation, before persist; pass result in; leave OCR (1044) reading `temp_path`; replace banner logs (1027-1041) with structured event
- [ ] Verify `upload.py` OCR call at 1044-1047 still passes `temp_path` (raw original) untouched
- [ ] Modify `_attachments.py:upload_attachment` (77-168): add `compress: bool = True`, compress copy of `file_content` before multipart (108), fallback on failure, extend success log
- [ ] Modify `_attachments.py:upload_file_standalone` (170-233): same hook before multipart (196) + `compress` kwarg
- [ ] Modify `_steps_company.py` logo push (335): pass `compress=False`
- [ ] Add `MetricsRegistry._init_compression_metrics()` in `metrics.py`; register in `__post_init__` (490) AND `reset_metrics` (980)
- [ ] Confirm new `compression/` + `classifiers/pdf_classifier.py` are NOT in `pyproject.toml [tool.coverage.run].omit`
- [ ] Add optional `original_size`/`stored_size`/`was_compressed` to FE `ExtractionMetadata`/`UploadResponse` (`chat.ts:442-464`)
- [ ] Surface "compressed from X" in FE `components/chat/file-preview-modal.tsx` (164-167) when present
- [ ] Confirm FE `OCR_MAX_FILE_SIZE` (`constants.ts:1893`) stays 30MB (no lowering); leave 500KB-min as BE-only
- [ ] Unit test `tests/unit/file_processing/test_compression.py`: every decision branch, fallback-on-error, compressed>=original→original, extension preserved, copy-not-temp
- [ ] Unit test `tests/unit/file_processing/test_pdf_classifier.py`: total vs average semantics on mixed-density PDFs
- [ ] API test `tests/api/test_upload_compression.py`: persist-compressed via `get_storage_client().download_bytes`, OCR-reads-original, preview round-trip (suffix-derived content-type), fallback path
- [ ] API test `tests/api/v1/test_erp_attachment_compression.py` (respx): compressed bytes in multipart to `/api/method/upload_file`, `compress=False` opt-out for logo, cross-tenant `==404`
- [ ] Extend FE `tests/unit/lib/api/upload.test.ts` for 30MB agreement + optional compression metadata
- [ ] Verify per-surface storage isolation: no GCS write added to ERPHome path, no ERPNext write added to Genie path
- [ ] Dev smoke: assert `get_storage_client()` returns `GCSStorageClient` (not silent Local fallback) when bucket configured
- [ ] Run `ruff format --check`, `ruff check`, `mypy app/`, backend pytest (75% gate), FE lint/type-check/test (≥85% new)
- [ ] Run `/code-review` and `/security-review` after each pass

---

I have everything confirmed. Writing the section.

## 2. Tech Stack, Dependencies & Versions

This feature is deliberately **zero-new-runtime-dependency** for the required (in-spec) scope. Every library the compression + rebuild + classifier code needs is already pinned in `erpsense-backend/requirements.txt` and already proven to install in the production image (`docker/Dockerfile`). The only *optional, behind-a-flag, off-by-default* path (full native-PDF stream/object recompression) is the single place where a new pinned dependency (`pikepdf`) *could* be added — and the recommendation below is to **defer it** and ship the optional rebuild on the already-present `PyMuPDF` path, so that no new wheel, no new system library, and no new attack surface enters the image in pass one.

### 2.1 Libraries already present (the full toolset for in-scope work)

All confirmed in `erpsense-backend/requirements.txt` (verified verbatim):

| Library | Pin in `requirements.txt` | Role in this feature | Where it is already used (don't re-add) |
|---|---|---|---|
| **Pillow** | `Pillow>=10.0.0` | Image recompression (`Image.open(BytesIO)` → resize to ≤1200px longest edge → re-encode JPEG/PNG/WebP to ≤200 KB on a **copy** of in-memory `content`). Must replicate the `RGBA`/`P`→`RGB` conversion that OCR already does. | `image_handler.py:_resize_if_needed` (`img.save(file_path)` at line 162 — the in-place overwrite footgun); `BrandingService` logo resize. |
| **PyMuPDF (fitz)** | `PyMuPDF>=1.23.0` | (a) page-count + per-page text density for the native-vs-scanned classifier; (b) scanned-PDF rasterization at 200 DPI for per-page recompression; (c) the **optional** native-PDF rebuild (`doc.save(..., garbage=4, deflate=True)`), which needs no extra dependency. | `pdf_handler.py:_pdf_to_images` (200 DPI, `fitz.Matrix(zoom)`); `pdf_handler.py:_has_sufficient_text`; PDF page-count validation in `upload.py:954-1019`. |
| **reportlab** | `reportlab>=4.1` | Rebuild a single PDF from the recompressed page images (scanned-PDF path), since `img2pdf` is **not** available. Already in the image for audit-trail PDF export. | audit-trail PDF export; `weasyprint` co-resident. |
| **chardet** | `chardet>=5.2.0` | Encoding sniffing for any text-side guards (not strictly needed; native text/CSV/DOCX/XLSX are pass-through). | text handlers. |
| **aiofiles** | `aiofiles>=23.0.0` | The temp-file write at `upload.py:948-949` stays untouched; compression itself runs in-memory + `asyncio.to_thread`, so no new async-IO lib. | `upload.py` temp write. |
| **google-cloud-storage** | `google-cloud-storage>=2.10.0` | The persist sink. Compressed bytes flow through the **single existing** `storage.upload_bytes(path, data, content_type)` at `storage_client.py:246-252` / called from `upload.py:_save_original_file` line 258. No new storage lib, no signed-URL lib. | `GCSStorageClient`, `LocalStorageClient`, `get_storage_client()`. |

CPU-bound calls (Pillow encode, fitz rasterize, reportlab build) are synchronous and **must** be wrapped in `asyncio.to_thread(...)` from inside `upload_file()` (execution is synchronous-inline per corrected decision 6; "parallel" = `asyncio.gather` over `asyncio.to_thread` within the request), so they never block the event loop.

### 2.2 Libraries explicitly NOT installed — and the decision on each

Confirmed absent from `requirements.txt` (grep returned zero matches in `app/` and the requirements file):

- **`pikepdf`** — the "nice" native-PDF object/stream recompressor. **Decision: do NOT add in pass one.** The in-scope spec compresses *only* images (≤200 KB / ≤1200px) and scanned PDFs (≤300 KB/page); native PDF/DOCX/XLSX/CSV are stored **as-is** (decision 7). The optional native-PDF container rebuild is **behind a flag, off by default**, and is implemented on `PyMuPDF` (`doc.save(garbage=4, deflate=True, clean=True)`) — which is already installed. Adding `pikepdf` would pull in a `qpdf` native dependency (a new C++ toolchain surface, new CVE feed to track, larger image) for an optional path; that contradicts the "no new attack surface unless required" posture. If product later wants aggressive native-PDF shrink, add it **then**, pinned `pikepdf>=9.0,<10` with a `qpdf` apt line in the Dockerfile, gated behind the same off-by-default flag.
- **`img2pdf`** — would be the clean way to assemble images→PDF for the scanned-PDF rebuild. **Decision: do NOT add.** Use the already-present `reportlab` (canvas + `drawImage`) to assemble recompressed page images into the rebuilt PDF. Same output, zero new dependency.
- **`python-magic` / `lxml`** — **Decision: do NOT add.** Content-type / format detection for the compression branch keys off the **filename suffix + the `content_type`** already threaded through both surfaces (matches the suffix-derived serve content-type at `upload.py:serve_original_file` `MIME_TYPE_MAP`, and the multipart 3rd-element on the ERPHome adapter `_attachments.py:108/196`). Pillow/fitz themselves validate the bytes on open and raise on a malformed/spoofed file — which the fallback (decision 9: store original, no retry, logged) catches. No libmagic native dep needed.

### 2.3 Files to CREATE (coverage-bearing, kept OUT of the omit list)

The Genie OCR seam files (`upload.py`, `library_gemini.py`, `pdf_handler.py`, `image_handler.py`, the whole `ocr/` + `handlers/` tree) are in `pyproject.toml [tool.coverage.run].omit` (verified lines 122-142). **New compression logic must NOT live in those files or it contributes 0% to the 75% CI gate.** Create new, non-omitted modules:

- **`app/services/file_processing/compression.py`** — the compressor. Pure, dependency-light, ≤500 lines (modularity cap). Public surface:
  ```python
  @dataclass(frozen=True)
  class CompressionResult:
      data: bytes            # compressed OR original (fallback)
      original_size: int
      stored_size: int
      was_compressed: bool   # False on skip/fallback/regression
      strategy: str          # "image" | "scanned_pdf" | "native_pdf_rebuild" | "skip" | "fallback"
      fallback_reason: str | None

  def compress_image(content: bytes, suffix: str, *, max_px: int = 1200, target_kb: int = 200) -> CompressionResult: ...
  def compress_scanned_pdf(content: bytes, *, target_kb_per_page: int = 300, dpi: int = 200) -> CompressionResult: ...
  def rebuild_native_pdf(content: bytes) -> CompressionResult: ...   # optional, flag-gated by caller
  async def compress_for_storage(content: bytes, filename: str, content_type: str | None, *,
                                 settings: Settings) -> CompressionResult: ...  # orchestrator; to_thread inside; NEVER raises
  ```
  Hard contract (decision 9): **never raises** — any exception → `CompressionResult(data=content, was_compressed=False, strategy="fallback", fallback_reason=...)`. If `compressed >= original` → return original (`was_compressed=False`, `strategy="...", fallback_reason="size_regression"`). Operates **only on the in-memory `content` copy**, never `temp_path` (the `image_handler` in-place-overwrite + the OCR/page-count temp reads make `temp_path` off-limits).
- **`app/services/file_processing/_pdf_classifier.py`** — the single shared `is_native_pdf(...)` helper that both OCR classifiers delegate to, **parameterized to preserve each site's existing semantics** (`pdf_handler` = total `len(text) >= 50*page_count`, branch; `library_gemini` = per-page average `< 50`, raise). The compressor calls the same helper so "compress only images + scanned PDFs" stays in lock-step with what OCR considers native vs scanned. Not in the omit list → counts toward coverage.
- **`app/services/file_processing/compression_metrics.py`** — log-based metrics (mirrors `app/services/tally_sync/diagnostic_metrics.py`): thin `logger.info("compression.decision" | "compression.fallback" | "gcs.upload", extra={...})` wrappers (filename, original/stored bytes, saved_pct, strategy, storage_backend, duration_ms — never bytes/PII). No registry wiring required.
- **Tests** (decision: lay them where they're measured) — `tests/unit/file_processing/test_compression.py`, `test_pdf_classifier.py` (pure logic, no DB); extend `tests/api/test_upload_cancel.py` pattern for the persist-compressed-bytes assertion; ERPHome adapter compression test under `tests/api/v1/` with `respx`.

### 2.4 Files to MODIFY (no behavior break)

- **`app/api/v1/endpoints/upload.py`** — at the persist seam (call site `1024-1025`): compute `result = await compress_for_storage(content, original_filename, content_type, settings=...)` **after** `content = await file.read()` (864) and **after** PDF page-count validation (ends 1019), then pass `result.data` into `_save_original_file` (single `upload_bytes` write at 258 — overwrite the bytes, do **not** add a second write or the dup-filename suffixer renames it). OCR call at `1044-1047` keeps reading the untouched `temp_path`. Replace the f-string log at line 259 with structured `logger.info("upload.original_saved", extra={...})`. No new import beyond the new modules.
- **`app/adapters/erpnext/_attachments.py`** — same compressor, applied to a **copy** of `file_content` immediately before the multipart build (`upload_attachment` line 108; `upload_file_standalone` line 196). Add `compress: bool = True` kwarg to both; provisioning logo push (`_steps_company.py:335`) passes `compress=False`. ERPHome persists to ERPNext only — no GCS.
- **`app/config.py`** — add **Settings** fields (knobs belong in config, never `os.getenv` per R03): `image_compression_enabled: bool = True`, `image_compress_max_px: int = 1200`, `image_compress_target_kb: int = 200`, `scanned_pdf_compress_target_kb_per_page: int = 300`, `compression_min_exempt_bytes: int = 500 * 1024` (decision 8: <500 KB exempt), `native_pdf_rebuild_enabled: bool = False` (decision 7: optional path off by default). These are env/infra knobs, named after the subsystem — **not** `feature_*` (reserved) and **not** per-tenant `FeatureService` gates.
- **`requirements.txt`** — **no change** for in-scope work. (Documented here so a reviewer doesn't "fix" a missing pin: the toolset is already complete.) Only if/when the `pikepdf` native-PDF path is greenlit later: add `pikepdf>=9.0,<10` plus a `qpdf` apt line — out of scope for this pass.
- **`requirements-dev.txt`** — **no change**; `pytest`, `respx`, `fakeredis`, `Pillow`/`PyMuPDF`/`reportlab` (inherited via `-r requirements.txt`) cover the tests.
- **`docker/Dockerfile`** — **no change required.** Pillow/PyMuPDF/reportlab ship manylinux/cp311 wheels installed by the existing `pip install -r requirements.txt` (line 28) on `python:3.11-slim`; the system libs already present (`libpango*`, `libgdk-pixbuf`, `fonts-noto-core` — lines 16-21, for weasyprint) are a superset of anything reportlab needs, and PyMuPDF/Pillow wheels are self-contained (no extra apt). The build is multi-stage and copies site-packages forward (lines 36-37), so the new pure-Python modules need nothing.

### 2.5 Security posture of each dependency (all already in the image)

- **Pillow** (`>=10.0.0`): historically the highest-CVE library here (decompression-bomb / malformed-image RCEs). Mitigations baked into the compression path: open from `BytesIO` (never a path the attacker controls on disk), the existing per-file size caps (Genie 30 MB, ERPHome 10 MB) bound the input, and a **decompression-bomb guard** (set `Image.MAX_IMAGE_PIXELS` defensively or check `img.size` before resize) — on any `PIL.UnidentifiedImageError`/`DecompressionBombError` the fallback stores the original and logs `fallback_reason`. Recommend bumping the floor to a current patched line (`Pillow>=10.3.0`) when this PR lands, since 10.0.0 has known fixes downstream.
- **PyMuPDF (fitz)** (`>=1.23.0`): bundles MuPDF (C). Risk is malformed-PDF parsing; bounded by the 30 MB cap + the existing page-count validation already done before compression, and any `fitz` exception → fallback-to-original. No network, no shell-out.
- **reportlab** (`>=4.1`): pure-Python PDF *writer* (we only ever **write** a PDF from images we ourselves decoded via Pillow — never parse untrusted PDF with it), so its attack surface in this path is effectively nil.
- **google-cloud-storage / ADC**: unchanged. Auth is ADC (local `gcloud auth application-default login`; deployed Cloud Run base SA). IAM is least-privilege (`objectCreator` + `objectViewer` only). The dup-filename suffixer already avoids overwrite (denied by `objectCreator`), and compression keeps the same filename, so persist still writes a **fresh** path on first save. No new IAM, no `signBlob` (the dead signed-URL helper stays dead).
- **No new wheels, no new native libs, no new CVE feeds** introduced in pass one — the single most important security statement for this section.

### 2.6 How the dev image installs them

The dev/prod image is the same `docker/Dockerfile` (`python:3.11-slim`, multi-stage). `pip install --no-cache-dir -r requirements.txt` (line 28) already resolves Pillow/PyMuPDF/reportlab from PyPI manylinux wheels; the dependencies stage's site-packages are copied into the application stage (lines 36-37). Because the feature adds **no** new line to `requirements.txt` (in-scope), the dev image is byte-for-byte identical on the dependency axis — the only change is new pure-Python source under `app/services/file_processing/`. Local dev (`venv` + `pip install -r requirements-dev.txt`) likewise already has the full toolset. The "works in DEV with GCS configured" requirement (decision 10) is a config/ADC concern (`GCS_UPLOAD_BUCKET=erpsense-backend-uploads-dev`, ADC login), not a dependency concern — nothing to install.

### 2.x Checklist

- [ ] Confirm no new pin in `requirements.txt` for in-scope work; add a one-line comment near the File Processing block noting Pillow/PyMuPDF/reportlab cover compression so a reviewer doesn't "add a missing dep".
- [ ] (Recommended) Bump `Pillow>=10.0.0` → `Pillow>=10.3.0` to pick up patched decompression/parsing CVEs; verify the wheel still resolves on `python:3.11-slim` via a clean `docker build`.
- [ ] Do NOT add `pikepdf` or `img2pdf` in this pass; record the deferral (native-PDF rebuild uses `PyMuPDF doc.save(garbage=4, deflate=True, clean=True)`; scanned-PDF rebuild uses `reportlab`).
- [ ] Do NOT add `python-magic`/`lxml`; format detection keys off filename suffix + threaded `content_type`, with Pillow/fitz open-time validation + fallback.
- [ ] Create `app/services/file_processing/compression.py` (≤500 lines) with `CompressionResult`, `compress_image`, `compress_scanned_pdf`, `rebuild_native_pdf`, async `compress_for_storage` — **never raises**, operates on in-memory `content` copy only, `asyncio.to_thread` for all CPU-bound calls.
- [ ] Create `app/services/file_processing/_pdf_classifier.py` with parameterized `is_native_pdf(...)`; delegate from `pdf_handler.py:_has_sufficient_text` (preserve total-chars branch) and `library_gemini.py:_extract_pdf_native` (preserve per-page-average raise) without changing thresholds, DPI, or control flow.
- [ ] Create `app/services/file_processing/compression_metrics.py` (log-based, mirror `diagnostic_metrics.py`); event names `compression.decision` / `compression.fallback` / `gcs.upload`; no bytes/PII in `extra`.
- [ ] Verify all four new modules are NOT in `pyproject.toml [tool.coverage.run].omit` (the OCR/handlers tree IS omitted — lines 122-142).
- [ ] Add Settings to `app/config.py`: `image_compression_enabled`, `image_compress_max_px=1200`, `image_compress_target_kb=200`, `scanned_pdf_compress_target_kb_per_page=300`, `compression_min_exempt_bytes=512000`, `native_pdf_rebuild_enabled=False` — subsystem-named, not `feature_*`, not `os.getenv`.
- [ ] Wire `compress_for_storage` into `upload.py` between line 1019 and 1024; pass `result.data` to `_save_original_file` (overwrite the single `upload_bytes` at 258; no 2nd write); keep OCR reading `temp_path`; preserve filename+extension.
- [ ] Replace the f-string log at `upload.py:259` with `logger.info("upload.original_saved", extra={...})`; do not extend the `'='*60` banner logs.
- [ ] Wire compression into `_attachments.py` `upload_attachment` (before line 108) and `upload_file_standalone` (before line 196) on a copy of `file_content`; add `compress: bool = True`; pass `compress=False` from `_steps_company.py:335`.
- [ ] Add decompression-bomb guard in `compress_image` (`Image.MAX_IMAGE_PIXELS` / size check) and verify the `RGBA`/`P`→`RGB` conversion matches `image_handler._resize_if_needed`.
- [ ] Confirm `docker/Dockerfile` needs no apt/pip change for the in-scope path; run a clean `docker build` to prove Pillow/PyMuPDF/reportlab wheels still install on `python:3.11-slim`.
- [ ] Confirm dev install path (`pip install -r requirements-dev.txt`) provides the full toolset; no new dev pin.
- [ ] Document (in the optional-path section of the plan only) the future `pikepdf>=9.0,<10` + `qpdf` apt addition, gated behind `native_pdf_rebuild_enabled`, as explicitly out of scope for this pass.

---
Files referenced (all absolute): `/Users/bhags/Desktop/erpsense_all/erpsense-backend/requirements.txt`, `/Users/bhags/Desktop/erpsense_all/erpsense-backend/requirements-dev.txt`, `/Users/bhags/Desktop/erpsense_all/erpsense-backend/docker/Dockerfile`, `/Users/bhags/Desktop/erpsense_all/erpsense-backend/pyproject.toml`.

---

All anchors verified. I now have everything needed to write Section 3 with concrete, grounded detail.

## 3. Configuration, Feature Flags & Limit Reconciliation

All compression behavior must be controllable from one place per layer: backend knobs live in `app/config.py` (`Settings`, an `env_file=".env"`, `case_sensitive=False` Pydantic-Settings model — confirmed at `app/config.py:38-44`), and frontend copy/limits live in `erpsense-frontend/src/lib/constants.ts`. Per CLAUDE.md "Infra / environment kill-switches … stay as `Settings` fields … Do NOT use the `feature_*` prefix" — compression is an operational/environment kill-switch (it is not a per-tenant plan capability), so it does **not** go through `FeatureService`/`FeatureKey`. It is named after its subsystem (`compression_*`), exactly like the existing `brs_orphan_cleanup_enabled` precedent (`app/config.py:523`). The master flag defaults **off** so merging this PR changes nothing at runtime until an env explicitly flips it.

### 3.1 Why a config block (not magic numbers, not `os.getenv`)

The seams document that compression thresholds are scattered conceptually but must be single-sourced. Two hard constraints drive the design:

- **R03 (no `os.getenv` outside `app/config.py`)** — the `be-logging` seam flagged `diagnostic_metrics.py` as a violation we must NOT replicate. Every compression knob (flag, thresholds, target sizes) is a typed `Settings` field, read via the existing `get_settings()` accessor, never `os.getenv`.
- **Coverage placement** — `app/api/v1/endpoints/upload.py` is in `[tool.coverage.run].omit` (`pyproject.toml:145`, per `tests-ci` seam). Config reads are cheap, but the *compression logic* that consumes these settings must live in a NEW non-omitted module (`app/services/file_processing/compression.py`, defined in the implementation section). Section 3 only owns the knobs; it deliberately keeps them as plain data so both the omitted call site (`upload.py`) and the covered compressor read the same `Settings`.

### 3.2 Backend config — `app/config.py` (MODIFY)

Add one cohesive block. Place it immediately after the existing OCR/GCS group (`app/config.py:238-249`, the `# OCR Settings` + `# GCS Storage (for OCR file uploads)` section) so it sits with the surface it serves. All fields are typed primitives with safe defaults; none are required, so an env that sets nothing inherits "off".

```python
# ── Upload compression (Genie OCR + ERPHome attach surfaces) ──────────────
# Master kill-switch. OFF by default: merging this feature is a no-op until an
# env flips it. Mirrors the brs_orphan_cleanup_enabled safe-off precedent.
compression_enabled: bool = False
# Per-surface enables (only consulted when compression_enabled is True).
compression_genie_enabled: bool = True       # POST /api/v1/upload originals → GCS
compression_erphome_enabled: bool = True     # ERPNext attach adapter sink
# Per-type enables.
compression_images_enabled: bool = True      # png/jpg/jpeg/gif/bmp/tiff/webp
compression_scanned_pdf_enabled: bool = True # PDFs OCR classifies as scanned
# MIN exemption: files smaller than this are never compressed (net-new rule,
# decision 8). 500 KB. Cheap files aren't worth the CPU or quality loss.
compression_min_bytes: int = 500 * 1024
# Image targets (decision 7): aim for <=200 KB and <=1200 px longest edge.
compression_image_target_bytes: int = 200 * 1024
compression_image_max_dimension: int = 1200
compression_image_jpeg_quality: int = 80     # starting JPEG quality, stepped down to hit target
# Scanned-PDF target: <=300 KB per page (decision 7).
compression_scanned_pdf_target_bytes_per_page: int = 300 * 1024
compression_scanned_pdf_dpi: int = 150        # raster DPI for rebuild (OCR rasters at 200; rebuild can be lower)
# OPTIONAL native-container rebuild (PDF/DOCX/XLSX). OFF by default, guarded
# (decision 7). When False, native files are stored as-is — no behavior change.
compression_native_rebuild_enabled: bool = False
```

Notes that keep existing behavior intact:

- **No existing field is renamed or repurposed.** `gcs_bucket_name` (`:244`), `gcs_project_id` (`:245`), `gcs_upload_bucket` (`:517`), `upload_dir` (`:516`) are untouched — storage selection via `get_storage_client()` (`storage_client.py:399-418`) is unchanged. Compression writes through the same `upload_bytes(path, data, content_type)` contract.
- **The two pre-existing size-limit sets stay as-is.** The Genie OCR module constants `MAX_FILE_SIZE=30MB / MAX_TOTAL_SIZE=32MB / MAX_FILES_PER_SESSION=5 / MAX_PAGES_PER_PDF=10` (`upload.py:81-84`) and the import-surface `max_file_size_bytes=10MB / max_files_per_session=50` (`config.py:525-526`) are **independent** and remain so. Compression adds knobs; it does not unify limits. The `compression_min_bytes` floor is purely additive and gates *inside* compression, never as an upload rejection.
- **`@field_validator`s (optional, recommended).** Add bounds so a fat-fingered env can't produce a 0-DPI raster or a quality of 500: clamp `compression_image_jpeg_quality` to 10–95, `compression_image_max_dimension` to >=256, `compression_scanned_pdf_dpi` to 72–300, and assert `compression_image_target_bytes < compression_min_bytes`-independence (targets may exceed min; only the floor gates eligibility). Validators raise on import (config is constructed once), surfacing misconfig at boot — never mid-request.

### 3.3 Reading the config — accessor contract

The compressor (`app/services/file_processing/compression.py`, implemented in the build section) takes a small immutable params object derived from `Settings`, so it is pure-testable without env juggling:

```python
@dataclass(frozen=True)
class CompressionConfig:
    enabled: bool
    images_enabled: bool
    scanned_pdf_enabled: bool
    native_rebuild_enabled: bool
    min_bytes: int
    image_target_bytes: int
    image_max_dimension: int
    image_jpeg_quality: int
    scanned_pdf_target_bytes_per_page: int
    scanned_pdf_dpi: int

    @classmethod
    def from_settings(cls, s: Settings) -> "CompressionConfig": ...
```

`upload.py` (Genie) passes `compression_genie_enabled and compression_enabled`; the ERPNext adapter (`_attachments.py`) passes `compression_erphome_enabled and compression_enabled`. Per-surface gating happens at the call sites; the compressor itself only sees a single resolved `enabled`. This keeps the omitted `upload.py` to a one-line guard and puts the testable branching in the covered module.

### 3.4 Frontend config — `erpsense-frontend/src/lib/constants.ts` (MODIFY)

**Limit reconciliation is already satisfied on the Genie surface** (per `fe` seam): `OCR_MAX_FILE_SIZE = 30MB`, `OCR_MAX_TOTAL_SIZE = 32MB`, `OCR_MAX_FILES = 5`, `OCR_MAX_PAGES_PER_PDF = 10` (`constants.ts:1893-1896`) — these already match `upload.py:81-84` exactly. The ground-truth worry that the FE says "10MB" is false for this surface; the only `10MB` constant on the FE is `attachments-panel.tsx:25 MAX_BYTES` for the unrelated production-floor FileStorage sink. **Do NOT lower the Genie limit to 10MB.** No FE change is required for upload/preview to keep functioning.

What we add to the FE is **additive transparency copy only** (off-by-default-flag-aware, no behavior change):

```ts
/** Upload compression (mirrors BE app/config.py compression_* defaults; UI copy only).
 *  BE is the source of truth — these inform hint text and the limit badge, they do
 *  NOT gate uploads. The 30MB/32MB/5/10 limits above are unchanged. */
export const COMPRESSION_MIN_BYTES = 500 * 1024;          // matches compression_min_bytes
export const COMPRESSION_IMAGE_TARGET_BYTES = 200 * 1024; // matches compression_image_target_bytes
```

- The FE has **no compression enable flag of its own** — it learns whether a file was actually compressed from the BE response. Per the `fe` seam, surfacing this requires extending `UploadResponse`/`ExtractionMetadata` (`types/chat.ts:442-464`) with optional `original_size?`, `stored_size?`, `was_compressed?`, then rendering "compressed from X" in `chat/file-preview-modal.tsx:164-167`. That is observability work owned by a later section; Section 3 only lands the two mirror constants and reserves the i18n keys.
- **i18n:** any new user-facing string (e.g. "Large images are compressed automatically") goes under `chat:file_preview` in `public/locales/*/chat.json` — never hardcoded English. Note the `fe` seam flags `chat-input.tsx:214/247/271/298` as pre-existing hardcoded-English size errors; do not add to that debt.
- **Filename/extension invariant (decision 4):** the FE preview path keys content-type off filename+extension (`chat-utils.ts buildPreviewUrl` + `EXT_TO_MIME`, and BE `serve_original_file` MIME map is suffix-derived). Compression must preserve filename+extension. This is a constraint the FE config relies on but does not enforce — it is enforced BE-side and asserted in tests.

### 3.5 ERPHome surface — explicit non-reconciliation

Per `be-erphome` seam: the ERPHome attach cap is genuinely **10MB** (`_shared.py:111 _MAX_ATTACHMENT_SIZE_BYTES = 10*1024*1024`), enforced on both `doc_social.py:upload_attachment` and `file_proxy.py:upload_erp_file`. The "30MB not 10MB" reconciliation is the Genie surface only. **Do NOT raise `_MAX_ATTACHMENT_SIZE_BYTES` to 30MB** — that is a different product surface with its own intent and 413 messages. Compression on this surface is gated by `compression_erphome_enabled and compression_enabled`, applied to a copy of `file_content` inside the adapter, and excluded for the provisioning logo push via the `compress=False` kwarg (`_steps_company.py:335`).

### 3.6 How dev / prod set these

| Setting | Local `.env` / `env.example` | Dev (Cloud Run / Terraform) | Staging / Prod |
|---|---|---|---|
| `COMPRESSION_ENABLED` | unset → `False` for normal dev; set `=true` to manually smoke compression | inject `=true` once verified | inject `=true` after dev soak; this is the rollout switch |
| `COMPRESSION_GENIE_ENABLED` / `COMPRESSION_ERPHOME_ENABLED` | default `True` (only matters when master on) | default `True` | default `True`; per-surface kill if one regresses |
| `COMPRESSION_NATIVE_REBUILD_ENABLED` | `False` | `False` | `False` until container-rebuild is independently validated |
| thresholds (`COMPRESSION_*_BYTES`, `*_DIMENSION`, `*_DPI`, `*_QUALITY`) | rely on defaults; override only to tune | defaults | defaults |

Mechanics, grounded in the storage/terraform seams:

- **Storage selection is independent of the compression flag.** `get_storage_client()` picks GCS when `gcs_bucket_name or gcs_upload_bucket` is set (`storage_client.py:407`), else `LocalStorageClient` with a WARNING. Local `.env` sets `GCS_BUCKET_NAME=erpsense-ocr-dev` and `GCS_UPLOAD_BUCKET=erpsense-backend-uploads-dev` (verified) → GCS path in dev (decision 10 satisfied). Deployed envs inject only `GCS_UPLOAD_BUCKET=var.gcs_upload_bucket_name` via the `phase2_enabled` dynamic env block (`terraform/modules/cloud-run/main.tf:610-617`), and `phase2_enabled=true` in `dev.tfvars:161`, `staging.tfvars:146`, `prod.tfvars:118` — so deployed envs resolve to `gcs_upload_bucket` (the OCR `GCS_BUCKET_NAME` is never injected).
- **No new Terraform env wiring is mandatory.** `compression_*` are plain Pydantic fields with safe defaults; an env that adds nothing gets the off-by-default behavior. To turn compression on in a deployed env, add `COMPRESSION_ENABLED` (and any overrides) to the same Cloud Run env mechanism that already injects `GCS_UPLOAD_BUCKET`. This is an ops action, not a code dependency — listed in the deploy/runbook section, not a blocker for merge.
- **`env.example` (MODIFY):** add a documented, commented `COMPRESSION_ENABLED=false` block (with the threshold knobs as commented examples) so operators discover the switch, mirroring how `env.example:54-58` documents the GCS block.
- **DEV-GCS smoke caveat (`be-storage` gotcha):** local `.env` `GCS_BUCKET_NAME=erpsense-ocr-dev` may not be provisioned by Terraform; the reliably-provisioned bucket is `erpsense-backend-uploads-dev`. The dev smoke must assert the GCS client is actually selected (not silently degraded to Local) before trusting "works with GCS". This is a verification note for the smoke section, not a config change.

### 3.7 Tests for this section

- **`tests/unit/core/test_compression_config.py` (CREATE)** — pure-logic unit lane (`tests-ci` seam: `tests/unit/` is DB-free, auto-`@pytest.mark.unit`). Assert: defaults (`compression_enabled is False`, `compression_min_bytes == 500*1024`, all targets/dimensions/DPI exact); `CompressionConfig.from_settings()` round-trips every field; validators reject out-of-range (`jpeg_quality=200`, `dpi=0`, `image_max_dimension=10`) by raising on `Settings(...)` construction. Lives in a non-omitted path so it counts toward the 75% gate (`config.py` is not in `[tool.coverage.run].omit`).
- **FE `tests/unit/lib/...` (extend):** assert `OCR_MAX_FILE_SIZE`/`OCR_MAX_TOTAL_SIZE` unchanged at 30/32MB (guards against an accidental "fix to 10MB") and that the two new mirror constants equal their BE counterparts.

### 3.x Checklist

- [ ] In `erpsense-backend/app/config.py`, add the `# Upload compression` block (13 `compression_*` fields) after the OCR/GCS group (`:238-249`), defaulting `compression_enabled=False`.
- [ ] Set per-surface defaults `compression_genie_enabled=True`, `compression_erphome_enabled=True`.
- [ ] Set per-type defaults `compression_images_enabled=True`, `compression_scanned_pdf_enabled=True`.
- [ ] Set `compression_min_bytes=500*1024` (500KB MIN-exempt, decision 8).
- [ ] Set image targets `compression_image_target_bytes=200*1024`, `compression_image_max_dimension=1200`, `compression_image_jpeg_quality=80`.
- [ ] Set scanned-PDF targets `compression_scanned_pdf_target_bytes_per_page=300*1024`, `compression_scanned_pdf_dpi=150`.
- [ ] Set `compression_native_rebuild_enabled=False` (optional rebuild, guarded, decision 7).
- [ ] Add `@field_validator`s clamping `jpeg_quality` (10–95), `image_max_dimension` (>=256), `scanned_pdf_dpi` (72–300); raise on out-of-range at construction.
- [ ] Confirm NO existing `config.py` field renamed/repurposed (`gcs_bucket_name`, `gcs_upload_bucket`, `upload_dir`, `max_file_size_bytes` all untouched).
- [ ] Confirm Genie module constants `upload.py:81-84` and import-surface `config.py:525-526` limits are left unchanged (compression adds knobs, does not unify limits).
- [ ] Define `CompressionConfig` frozen dataclass + `from_settings()` in the new (non-omitted) `app/services/file_processing/compression.py`; never read `os.getenv` (R03).
- [ ] Wire per-surface resolution: Genie passes `compression_enabled and compression_genie_enabled`; ERPNext adapter passes `compression_enabled and compression_erphome_enabled`.
- [ ] In `erpsense-backend/env.example`, add a commented `COMPRESSION_ENABLED=false` block + commented threshold examples (mirror the `:54-58` GCS block style).
- [ ] In `erpsense-frontend/src/lib/constants.ts`, add `COMPRESSION_MIN_BYTES` + `COMPRESSION_IMAGE_TARGET_BYTES` mirror constants (UI copy only).
- [ ] Verify FE `OCR_MAX_FILE_SIZE=30MB`/`OCR_MAX_TOTAL_SIZE=32MB`/`OCR_MAX_FILES=5`/`OCR_MAX_PAGES_PER_PDF=10` are left as-is (`constants.ts:1893-1896`); do NOT lower to 10MB.
- [ ] Confirm ERPHome `_shared.py:111 _MAX_ATTACHMENT_SIZE_BYTES = 10MB` is NOT raised to 30MB (separate surface).
- [ ] Reserve i18n keys under `chat:file_preview` in `public/locales/*/chat.json` for future "compressed from X" copy (no hardcoded English).
- [ ] Document dev/prod rollout: `COMPRESSION_ENABLED` injected via the same Cloud Run env mechanism as `GCS_UPLOAD_BUCKET` (`cloud-run/main.tf:610-617`, `phase2_enabled=true` in all tfvars); off-by-default means no Terraform change is required to merge.
- [ ] Add `tests/unit/core/test_compression_config.py`: defaults, `from_settings()` round-trip, validator rejections (placed in non-omitted path for the 75% gate).
- [ ] Extend FE unit test to assert OCR max constants unchanged and new mirror constants equal BE values.

Relevant files: `/Users/bhags/Desktop/erpsense_all/erpsense-backend/app/config.py`, `/Users/bhags/Desktop/erpsense_all/erpsense-backend/env.example`, `/Users/bhags/Desktop/erpsense_all/erpsense-backend/app/services/file_processing/compression.py` (new), `/Users/bhags/Desktop/erpsense_all/erpsense-backend/tests/unit/core/test_compression_config.py` (new), `/Users/bhags/Desktop/erpsense_all/erpsense-frontend/src/lib/constants.ts`.

---

I have everything confirmed. The IAM grant is `objectCreator` + `objectViewer` only — which matters because the dup-filename suffixer means we never overwrite (overwrite is denied by `objectCreator`). Writing the section now.

## 4. GCS Connection, Storage Layer & Dev-Env Setup

This section specifies how compressed-original bytes are persisted to Google Cloud Storage on the **Genie Chat OCR surface only** (per corrected decision 5: Genie → GCS, ERPHome → ERPNext, never both), how the existing storage abstraction is reused without modification, how the single persist write is overwritten in place, the retry/timeout/error-handling envelope, and how it works in **DEV (GCS-configured), not just local** (corrected decision 10). The storage *transport* layer (`app/services/storage_client.py`) is treated as generic and is **not** modified for compression — all compression/observability wiring lands at the caller seam in `upload.py`, keeping the storage layer transport-only (one-axis-of-change rule).

### 4.1 How storage selection works today (do not change the contract)

The factory `get_storage_client()` (`app/services/storage_client.py:399-418`, `@lru_cache(maxsize=1)`) resolves:

```python
bucket  = settings.gcs_bucket_name or settings.gcs_upload_bucket   # OCR bucket first
project = settings.gcs_project_id or ""
return GCSStorageClient(bucket, project) if bucket else LocalStorageClient(settings.upload_dir)
```

- `GCSStorageClient.__init__` (`storage_client.py:222-232`) lazily imports `google.cloud.storage`, builds `storage.Client(project=project_id or None)` (project falls through to ADC inference when empty), and **already logs** `"GCSStorageClient initialized: bucket=%s, project=%s"` at INFO.
- The single persist sink for Genie originals is `upload_bytes(path, data: bytes, content_type="")` → `GCSStorageClient._upload_bytes_sync` (`storage_client.py:246-252`) → `blob.upload_from_string(data, content_type=... or "application/octet-stream")`, run off-thread via `_run_sync` → `loop.run_in_executor`.
- Preview is served by the **byte proxy** `GCSStorageClient.get_serve_response` (`storage_client.py:378-396`): it calls `exists()` then `download_bytes()` and returns a `Response` with `Content-Disposition: inline` + `Cache-Control: private, max-age=600`. It does **not** redirect to a signed URL (CORS-avoidance). The dead `_generate_signed_url_sync` (`storage_client.py:335-376`) stays dead — do not wire it (it needs `iam.serviceAccounts.signBlob`, granted nowhere).

**Reuse rule:** compression does not introduce a new storage client, a new factory, or a second write. It feeds compressed `bytes` through the exact same `upload_bytes(path, data, content_type)` contract that already runs for the original.

### 4.2 The single persist seam — overwrite bytes in place at `_save_original_file`

The only write of the Genie original is `_save_original_file()` (`app/api/v1/endpoints/upload.py:225-264`), whose single `await storage.upload_bytes(storage_path, content, content_type=content_type)` is at **line 258**. Per corrected decision 3, we **overwrite this one write’s payload** — we do **not** add a second `upload_bytes` call (the dup-filename suffixer at `upload.py:249-255` would rename a second write to `{stem}_1{suffix}` and create a phantom duplicate that the cancel-purge would then orphan).

**Wiring choice (recommended):** compute compressed bytes at the call site (`upload.py:1024-1025`, `if session_id: await _save_original_file(...)`) — *after* `content = await file.read()` (line 864) and *after* all size/page validation (the PDF page block ends ~line 1019) and *before* OCR (line 1044) and the `finally` temp-unlink (line 1359). Deferral is therefore already satisfied; no reordering is needed. The compressor is a **new, non-omitted module** (see §4.3) so its logic is coverage-measured; `_save_original_file` itself lives in coverage-omitted `upload.py`, so keep `_save_original_file` a thin pass-through and put all decision logic in the new module.

**FILES TO MODIFY**

- `/Users/bhags/Desktop/erpsense_all/erpsense-backend/app/api/v1/endpoints/upload.py`
  - Change `_save_original_file` signature to accept already-decided bytes plus telemetry context, keeping it transport-thin:
    ```python
    async def _save_original_file(
        session_id: str,
        original_filename: str,
        content: bytes,                 # bytes to persist (compressed-or-original, decided upstream)
        *,
        original_size: int | None = None,   # for the structured persist log only
        strategy: str = "none",             # "image" | "scanned_pdf" | "none"
    ) -> str | None:
    ```
  - At the call site (`upload.py:1024-1025`), call the new compressor **on a COPY of in-memory `content`** (never `temp_path`; corrected decisions 1-2, and the `ImageHandler._resize_if_needed` in-place-overwrite hazard at `image_handler.py:162`), then pass the result down:
    ```python
    if session_id:
        result = await maybe_compress_for_storage(content, original_filename, content_type=...)
        await _save_original_file(
            session_id, original_filename, result.bytes_to_store,
            original_size=len(content), strategy=result.strategy,
        )
    ```
    `content` (the OCR input written to `temp_path` at 948-949) is **untouched** — OCR at line 1044 still reads the original.
  - **Replace the f-string persist log at `upload.py:259`** (a CLAUDE.md violation) with a structured event (details in the Logging section), e.g. `logger.info("upload.original_saved", extra={...})`. Do **not** extend the `"="*60` banner logs (323-372).
  - Keep `safe_name` / suffix logic (241-244) and the dup-filename suffixer (249-255) **unchanged** — the compressed file MUST keep the same filename+extension (corrected decision 4; serve content-type is suffix-derived at `serve_original_file` `upload.py:1985-1988`).
  - Preserve the **non-blocking contract**: the existing `try/except Exception → logger.warning → return None` (262-264) stays; a storage failure must never block OCR. A compression failure also never raises (handled inside the compressor — see §4.4).

### 4.3 New module placement (coverage-bearing, transport stays generic)

`upload.py`, `library_gemini.py`, `pdf_handler.py`, `image_handler.py` are in `[tool.coverage.run].omit` (`pyproject.toml:86-181`) — code there is **not** measured against the 75% gate. Therefore:

**FILES TO CREATE**

- `/Users/bhags/Desktop/erpsense_all/erpsense-backend/app/services/file_processing/compression.py` — the compressor + `maybe_compress_for_storage()` orchestrator (image/scanned-PDF logic detailed in the Compression section). It returns a small result object the storage seam consumes:
  ```python
  @dataclass(frozen=True)
  class CompressionResult:
      bytes_to_store: bytes      # compressed bytes, or original on fallback
      original_size: int
      stored_size: int
      strategy: str              # "image" | "scanned_pdf" | "none"
      compressed: bool           # True only if a smaller payload was produced
      fallback_reason: str | None  # "error" | "regression" | "below_min" | "unsupported" | None
  ```
- `/Users/bhags/Desktop/erpsense_all/erpsense-backend/app/services/file_processing/compression_metrics.py` — log-based + registry metric helpers including a **`gcs.upload`** event wrapper (see Logging section); this is where storage-op observability is emitted, **not** inside `storage_client.py` (keeps the transport layer generic; corrected decision: logging at the caller seam).

This module is **not** added to the omit list, so its branches count toward the gate; the thin `_save_original_file` in `upload.py` remains uncounted but logic-free.

### 4.4 Retries, timeouts, and error handling for GCS ops

The Genie request executes synchronously inline (no worker; corrected decision 6), so the GCS write happens within the request before the 300s OCR call. The `google-cloud-storage` client already performs internal idempotent retries on transient 5xx/429 for `upload_from_string` (DEFAULT_RETRY), so we do **not** add an application retry loop for the original-persist write (corrected decision 9: **no retry** at our layer; on failure store original / log).

Error-handling envelope, all enforced at the **caller** seam (transport layer untouched):

1. **Compression failure or `compressed >= original`** → fall back to original `content`, set `fallback_reason`, emit a `WARNING` `compression.fallback`, then persist the original (decision 9). Never raises.
2. **GCS `upload_bytes` failure** (network, IAM 403, bucket missing) → caught by the existing `_save_original_file` `except Exception` (262), logged `WARNING`, returns `None`; OCR proceeds (decision: persist is non-blocking). Add an explicit timeout guard around the persist write so a wedged GCS socket cannot stall the whole request:
   ```python
   await asyncio.wait_for(
       storage.upload_bytes(storage_path, content, content_type=content_type),
       timeout=settings.original_persist_timeout_seconds,   # new Settings field, default 60
   )
   ```
   On `asyncio.TimeoutError` → same `WARNING` + `return None` path.
3. **IAM least-privilege gotcha (prod/dev GCS):** the Cloud Run SA holds `roles/storage.objectCreator` + `roles/storage.objectViewer` **only** (`terraform/modules/phase2/main.tf:35-37`). `objectCreator` permits *create-new* but **denies overwrite/delete** of an existing object. The dup-filename suffixer (`upload.py:249-255`) already guarantees a fresh path on collision, so the persist never attempts an overwrite — this is why we must keep the suffixer and must never “re-PUT to an existing path.” Document this invariant in a code comment at the seam.
4. **Emit `gcs.upload` telemetry** (status=`ok`/`error`/`timeout`, backend=`type(storage).__name__`, `original_size`, `stored_size`, `saved_pct`, `duration_ms`, `storage_path`) from `compression_metrics.py`, called at the seam — never logging file bytes/PII.

### 4.5 Config / Settings (knobs live in `app/config.py`, never `os.getenv`)

**FILE TO MODIFY:** `/Users/bhags/Desktop/erpsense_all/erpsense-backend/app/config.py` — add Settings fields beside the existing GCS block (`config.py:243-245`) and Phase-2 block (`config.py:515-532`). All compression knobs and the persist timeout are Pydantic Settings (R03: no `os.getenv` outside `config.py`):

```python
# Genie OCR original-file compression (Genie chat surface only)
compression_enabled: bool = True
compression_image_max_bytes: int = 200 * 1024            # target ≤200KB/image (decision 7)
compression_image_max_dimension: int = 1200              # ≤1200px longest edge (decision 7)
compression_scanned_pdf_max_bytes_per_page: int = 300 * 1024  # ≤300KB/page (decision 7)
compression_min_exempt_bytes: int = 500 * 1024           # <500KB skipped (decision 8: MIN exempt)
compression_pdf_rebuild_enabled: bool = False            # native-PDF/container rebuild OFF by default (decision 7, guarded)
original_persist_timeout_seconds: int = 60               # §4.4 wedged-GCS guard
```

Existing GCS selection fields are reused unchanged: `gcs_bucket_name` (244), `gcs_project_id` (245), `gcs_upload_bucket` (517), `upload_dir` (516). **No new bucket** is introduced — compressed originals persist to the same `sessions/{sid}/originals/{filename}` prefix in whichever bucket the factory already selects.

### 4.6 Dev-env: authentication, bucket & IAM (must work in DEV, not only local)

Authentication is **ADC everywhere** — no key file, no `GOOGLE_APPLICATION_CREDENTIALS`. `GCSStorageClient` calls `storage.Client(project=...)` which uses ADC implicitly.

- **Local dev (GCS path):** `gcloud auth application-default login` (per `env.example:56`). `env.example:54-57` sets `GCS_BUCKET_NAME=erpsense-ocr-dev`, `GCS_PROJECT_ID=erpsense-dev`. **CRITICAL gotcha to surface in the PR:** no Terraform (backend or platform) provisions an `erpsense-ocr-dev` bucket — phase2 only creates `${name_prefix}-uploads-{env}` (`= erpsense-backend-uploads-dev`, `phase2/main.tf:99-126`). So a local `.env` pointed at `erpsense-ocr-dev` will 404/permission-fail on every GCS op **unless that bucket exists out-of-band and ADC has access**. The realistically-working dev bucket is `erpsense-backend-uploads-dev` — verify the bucket exists before relying on dev GCS, or set `GCS_UPLOAD_BUCKET=erpsense-backend-uploads-dev` and leave `GCS_BUCKET_NAME` empty so the factory falls through to it.
- **Deployed DEV (Cloud Run):** Cloud Run runs as the platform base SA (`terraform/main.tf:91`). It injects **only** `GCS_UPLOAD_BUCKET=var.gcs_upload_bucket_name` (`cloud-run/main.tf:609-622`), gated on `phase2_enabled`; `GCS_BUCKET_NAME` (OCR) is **never** set in deployed envs. Net: deployed DEV uses `gcs_upload_bucket = erpsense-backend-uploads-dev` via the factory’s `gcs_bucket_name or gcs_upload_bucket` fallback. **No IAM change is required** — compressed `upload_bytes` (create-new) is covered by `objectCreator`; `download_bytes`/`exists` (preview) are covered by `objectViewer` (`phase2/main.tf:35-37`).
- **Silent-degrade hazard:** when no bucket is configured, `get_storage_client()` only emits a `WARNING` and returns `LocalStorageClient` (ephemeral, lost on restart) — it does **not** fail (`storage_client.py:412-417`). The dev smoke check must **assert the GCS client is actually selected** (e.g. `isinstance(get_storage_client(), GCSStorageClient)`), not merely that uploads “succeed.” `phase2_enabled` defaults `false` (`variables.tf:364`) but dev/staging/prod tfvars set it true — a new env that forgets it falls back to Local silently.

**FILES TO MODIFY (env/infra):**

- `/Users/bhags/Desktop/erpsense_all/erpsense-backend/env.example` — clarify the `GCS_BUCKET_NAME` comment: for a working local-GCS path either point at a bucket that exists in your ADC project, or set `GCS_UPLOAD_BUCKET=erpsense-backend-uploads-dev` and leave `GCS_BUCKET_NAME` empty; reaffirm `gcloud auth application-default login`.
- No new Terraform bucket/IAM is required for this feature (same prefix, same bucket, create-new-only access already granted). If product wants a dedicated OCR bucket, that is a **separate** phase2 module change (out of scope here) — flag it but do not bundle it.

### 4.7 Why existing behavior stays unbroken

- **Single write preserved:** only the *payload* of the existing `upload_bytes` at `upload.py:258` changes; no second write, suffixer untouched → cancel-purge (`_purge_upload_artifacts`, `upload.py:1366-1510`, matches `{stem}.{ext}`/`{stem}_{n}.{ext}`) keeps working.
- **Filename + extension unchanged** → `serve_original_file` (`upload.py:1927-2004`) suffix-derived MIME and FE `buildPreviewUrl`/`EXT_TO_MIME` preview both keep resolving (corrected decision 4).
- **OCR reads the original** (`temp_path`) → no quality/accuracy change; compression only affects the persisted/preview copy.
- **Non-blocking persist** → OCR, response, and the `finally` temp-unlink (`upload.py:1353-1363`) are unaffected by any compression or GCS error.
- **Transport layer untouched** → `storage_client.py` keeps its generic `upload_bytes`/`download_bytes`/`get_serve_response` contract; other callers (`get_archival_storage_client`, `get_user_uploads_storage_client`, BRS, import, branding, FileStorageService) are unaffected.
- **ERPHome surface not touched** here — its bytes go to ERPNext only (corrected decision 5); GCS persist is Genie-only.

### 4.x Checklist

- [ ] Confirm `get_storage_client()` selection in deployed DEV resolves to `GCSStorageClient` over `erpsense-backend-uploads-dev` (assert `isinstance`, not just “upload works”).
- [ ] Verify `erpsense-ocr-dev` either exists with ADC access locally, or switch local `.env` to `GCS_UPLOAD_BUCKET=erpsense-backend-uploads-dev` + empty `GCS_BUCKET_NAME`.
- [ ] Run `gcloud auth application-default login` for local-GCS dev; confirm no key file / `GOOGLE_APPLICATION_CREDENTIALS` is needed or added.
- [ ] Create `app/services/file_processing/compression.py` with `maybe_compress_for_storage()` + `CompressionResult` (NOT added to `pyproject.toml` `[tool.coverage.run].omit`).
- [ ] Create `app/services/file_processing/compression_metrics.py` with the `gcs.upload` event helper (emit at the caller seam, not in `storage_client.py`).
- [ ] Modify `_save_original_file` (`upload.py:225-264`): accept pre-decided `content` + `original_size`/`strategy` kwargs; keep it transport-thin and non-blocking.
- [ ] At `upload.py:1024-1025`, call the compressor on a **COPY of in-memory `content`** (never `temp_path`), then pass `result.bytes_to_store` into `_save_original_file`.
- [ ] Keep `content`/`temp_path` untouched so OCR at `upload.py:1044` still reads the original.
- [ ] Overwrite **only** the single `upload_bytes` payload at `upload.py:258`; do NOT add a second write; leave the dup-filename suffixer (249-255) intact.
- [ ] Preserve the original filename + extension end-to-end (no rename) so suffix-derived serve MIME + FE preview keep working.
- [ ] Replace the f-string persist log at `upload.py:259` with structured `logger.info("upload.original_saved", extra={filename, storage_path, original_size_bytes, stored_size_bytes, saved_pct, strategy, compressed, storage_backend})`.
- [ ] Wrap the persist `upload_bytes` in `asyncio.wait_for(..., timeout=settings.original_persist_timeout_seconds)`; on `TimeoutError` → WARNING + `return None`.
- [ ] Do NOT add an application retry loop for the GCS write (rely on the client’s internal idempotent retry; decision 9 = no retry, fallback to original).
- [ ] Confirm the persist path never overwrites an existing object (objectCreator denies overwrite); add a code comment documenting the suffixer invariant + IAM least-privilege.
- [ ] Add Settings fields in `app/config.py` (`compression_enabled`, image/scanned-PDF thresholds, `compression_min_exempt_bytes=500KB`, `compression_pdf_rebuild_enabled=False`, `original_persist_timeout_seconds=60`) — no `os.getenv` outside config.
- [ ] On compression failure OR `compressed >= original` OR `< 500KB`: store ORIGINAL bytes, set `fallback_reason`, emit WARNING `compression.fallback`; never raise (OCR must not be blocked).
- [ ] Emit `gcs.upload` telemetry (backend, status ok/error/timeout, original/stored size, saved_pct, duration_ms, storage_path) — never log file bytes/PII; do not pass auto-injected ctx ids in `extra`.
- [ ] Do NOT wire the dead `_generate_signed_url_sync` / signed URLs; preview stays on the existing byte proxy `get_serve_response`.
- [ ] Update `env.example:54-57` comment to clarify the working local-GCS bucket choice + ADC login.
- [ ] Verify no Terraform bucket/IAM change is required (same bucket, same prefix, create-new-only access already granted); flag any dedicated-OCR-bucket request as separate/out-of-scope.
- [ ] Confirm ERPHome attachment surface is NOT touched by this GCS persist path (Genie → GCS only; corrected decision 5).
- [ ] Confirm cancel-purge (`_purge_upload_artifacts`, `upload.py:1366-1510`) still matches the persisted compressed file by unchanged filename pattern.

---

I have confirmed all the key seams. The symbols, line numbers, and semantics match the ground truth and seam reports. I have enough to write a deep, concrete section.

## 5. Compression Engine: Module Structure & Interfaces

This section specifies a self-contained `app/services/compression/` package that turns in-memory upload bytes into smaller-but-equivalent bytes, with a deterministic pure-bytes contract, an eligibility gate, per-format compressors, the shared `is_native_pdf()` classifier, and a validation/fallback wrapper that can **never** raise into a caller's hot path. It is consumed by exactly two surfaces (per CORRECTED DECISION #5): the Genie OCR persist seam at `upload.py:_save_original_file` call site (`upload.py:1024-1025`) and the ERPHome adapter hooks at `_attachments.py:AttachmentsMixin.upload_attachment` (line ~108) / `upload_file_standalone` (line ~196). The engine itself knows nothing about storage, OCR, ERPNext, or HTTP — callers wire it in.

### 5.1 Design constraints carried from the seams

These are load-bearing and are enforced by the module's shape, not by caller discipline:

- **Never mutate inputs / never touch `temp_path`.** `engine.compress_bytes()` takes `bytes` and returns new `bytes`; it never receives or opens the OCR temp file. This is the guard against the in-place-overwrite footgun in `ImageHandler._resize_if_needed` (`image_handler.py:162`, `img.save(file_path)` overwrites the temp original *before* OCR) and against `PDFHandler` / page-count validation / OCR all reading the same `temp_path`. OCR keeps reading the untouched original; storage receives the compressed copy.
- **Pure, deterministic, side-effect-free core.** Given the same bytes + filename + surface, the same `CompressionResult` comes back. No clock, no randomness, no global mutable state. (The reportlab/PyMuPDF rebuild path must therefore pin any embedded metadata/timestamps off — see 5.6.)
- **CPU work runs in `asyncio.to_thread`.** Execution is synchronous-inline in the request (CORRECTED DECISION #6); Pillow encode and PyMuPDF rasterize/rebuild are blocking, so the async entrypoint offloads them. "Parallel" multi-page work = `asyncio.gather(*[asyncio.to_thread(...)])` inside the engine, bounded by a semaphore.
- **Same filename + extension preserved by contract.** The engine returns only bytes; the caller keeps `safe_name` unchanged (`_save_original_file` line 241-246) so the suffix-derived serve content-type (`serve_original_file` MIME_TYPE_MAP, `upload.py:1908-1924/1985-1988`) and FE preview routing (`chat-utils.ts` `EXT_TO_MIME` / `buildPreviewUrl` PREVIEWABLE_EXTENSIONS) keep working (CORRECTED DECISION #4). Recompressing a JPEG stays JPEG, a PNG stays PNG, a PDF stays PDF — the engine asserts `result.bytes` is decodable in the same format, else it falls back.
- **Fallback is in-band, not exceptional.** On any failure, unsupported type, sub-threshold size, or `new_size >= original_size`, the engine returns a `CompressionResult` with `changed=False` and `data == original content` plus a machine-readable `reason`. It must not raise (CORRECTED DECISION #9; preserves the non-blocking contract of `_save_original_file` which returns `None` on failure at `upload.py:262-264`, and the `{success: False}` envelope discipline of the ERPHome adapter methods).
- **Knobs live in `app/config.py` Settings**, never `os.getenv` (R03), never module constants that drift from the upload-limit constants. Defaults below.
- **Coverage-bearing placement.** The engine must NOT live in a `[tool.coverage.run].omit` path. `upload.py`, `library_gemini.py`, `pdf_handler.py`, `image_handler.py` are all omitted (`pyproject.toml:145+`). The new `app/services/compression/` package is unlisted and therefore measured — keep it out of `omit` so it counts toward the 75% gate.

### 5.2 Files to CREATE (full paths)

All under `/Users/bhags/Desktop/erpsense_all/erpsense-backend/`:

| File | Responsibility | Approx LOC budget (≤500 each, modularity rule) |
|---|---|---|
| `app/services/compression/__init__.py` | Public re-exports: `compress_bytes`, `CompressionResult`, `CompressionSurface`, `is_native_pdf`. The ONLY import surface callers use. | ~30 |
| `app/services/compression/types.py` | `CompressionResult` dataclass, `CompressionSurface` enum, `CompressionDecision`/`reason` literals, `CompressionConfig` snapshot. No logic. | ~90 |
| `app/services/compression/engine.py` | `async def compress_bytes(...)` orchestrator: eligibility gate → classify → dispatch → validate → fallback wrapper → emit telemetry. | ~220 |
| `app/services/compression/eligibility.py` | `classify(content, filename, content_type, surface) -> EligibilityVerdict`: min-size exempt, format detection (magic bytes + suffix), image vs scanned-PDF vs native/pass-through. | ~160 |
| `app/services/compression/image_compressor.py` | `compress_image(content, fmt, cfg) -> bytes | None` on a `BytesIO` copy: downscale to `<=max_px`, RGBA/P→RGB, re-encode JPEG/PNG/WebP to `<=target_kb`. Mirrors `image_handler.py:159-161` conversion. | ~180 |
| `app/services/compression/pdf_compressor.py` | `compress_scanned_pdf(content, cfg) -> bytes | None`: PyMuPDF rasterize 200 DPI → Pillow per-page compress → reportlab/PyMuPDF rebuild to `<=target_kb/page`. Uses shared classifier. | ~240 |
| `app/services/compression/pdf_classifier.py` | The shared `is_native_pdf()` + `avg_chars_per_page()` helpers (see 5.5). The single source of truth both OCR call sites delegate to. | ~70 |
| `app/services/compression/metrics.py` | Log-based telemetry helpers `record_compression_decision()`, `record_compression_fallback()` (mirrors `tally_sync/diagnostic_metrics.py` shape). Optional `_init_compression_metrics()` registration is wired in `app/core/metrics.py` (see 5.8). | ~110 |

Test files (placement per the tests-ci seam — must be in non-omitted, correct lanes):

| File | Lane |
|---|---|
| `tests/unit/compression/test_eligibility.py` | unit (pure) |
| `tests/unit/compression/test_image_compressor.py` | unit (pure) |
| `tests/unit/compression/test_pdf_compressor.py` | unit (pure) |
| `tests/unit/compression/test_pdf_classifier.py` | unit (pure; asserts BOTH OCR semantics preserved) |
| `tests/unit/compression/test_engine.py` | unit (pure; fallback/size-regression/never-raise) |

(Endpoint/integration tests for the wired-in surfaces — Genie `tests/api/` mirroring `test_upload_cancel.py`, ERPHome `tests/api/` with respx — are specified in the wiring/testing sections, not here.)

### 5.3 Files to MODIFY (for completeness; detailed wiring lives in later sections)

- `app/config.py` — add `CompressionConfig` Settings fields (5.7). MODIFY only; additive.
- `app/api/v1/endpoints/upload.py` — at the `_save_original_file` call (`1024-1025`) compute compressed bytes via the engine and pass them in; replace the f-string log at line 259 with a structured event (logging seam). No reordering needed (persist already precedes OCR at 1044 and unlink at 1359).
- `app/services/file_processing/handlers/pdf_handler.py` — `_has_sufficient_text` (`302-308`) becomes a thin delegate to `pdf_classifier.is_native_pdf(..., mode="total")`. Behavior preserved.
- `app/services/file_processing/ocr/library_gemini.py` — `_extract_pdf_native` (`589-618`) inline `avg < MIN_PDF_TEXT_PER_PAGE` check becomes `pdf_classifier.is_native_pdf(..., mode="average")`; keep the `raise FileProcessingException` control-flow.
- `app/adapters/erpnext/_attachments.py` — call the engine before the multipart build in both methods; add `compress: bool = True` kwarg.
- `app/services/erpnext_provisioning/_steps_company.py` — pass `compress=False` on the logo push (line ~335).
- `app/core/metrics.py` — optional `_init_compression_metrics()` registration (5.8).

### 5.4 Public interfaces (`types.py` + `engine.py`)

```python
# app/services/compression/types.py
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import Literal

class CompressionSurface(str, Enum):
    GENIE_OCR = "genie_ocr"        # upload.py persist seam
    ERP_ATTACHMENT = "erp_attachment"  # _attachments.py adapter hooks

# Stable, low-cardinality strings — safe as metric labels and log fields.
CompressionDecision = Literal["compressed", "skipped", "fallback"]
CompressionReason = Literal[
    "compressed_image",
    "compressed_scanned_pdf",
    "below_min_size",          # < min_compress_bytes (500KB default) -> exempt
    "unsupported_type",        # native pdf / docx / xlsx / csv / txt -> pass-through
    "native_pdf",              # PDF classified as text-native -> pass-through
    "size_regression",         # new >= original -> store original
    "compression_error",       # exception caught -> store original
    "empty_input",
    "disabled",                # feature flag off
]

@dataclass(frozen=True)
class CompressionResult:
    data: bytes                  # ALWAYS the bytes to persist (compressed or original)
    changed: bool                # True only when data is a smaller, valid re-encode
    decision: CompressionDecision
    reason: CompressionReason
    original_size: int
    new_size: int                # == original_size when not changed
    strategy: str                # "image" | "scanned_pdf" | "none"

    @property
    def saved_bytes(self) -> int: ...
    @property
    def saved_pct(self) -> float: ...   # 0.0 when not changed
```

```python
# app/services/compression/engine.py
import logging
from app.config import settings
from app.services.compression.types import CompressionResult, CompressionSurface
from app.services.compression.eligibility import classify
from app.services.compression import image_compressor, pdf_compressor, metrics

logger = logging.getLogger(__name__)

async def compress_bytes(
    content: bytes,
    *,
    filename: str,
    surface: CompressionSurface,
    content_type: str | None = None,
) -> CompressionResult:
    """
    Pure-bytes compression entrypoint. NEVER raises, NEVER mutates `content`,
    NEVER touches any temp file. Returns a CompressionResult whose `.data` is
    ALWAYS safe to persist (compressed when changed, original otherwise).
    """
    original_size = len(content)
    # Fast guards (kill-switch, empty, below-min) -> pass-through results.
    if not settings.compression_enabled:
        return _passthrough(content, "disabled", original_size)
    if original_size == 0:
        return _passthrough(content, "empty_input", original_size)

    verdict = classify(content, filename, content_type, surface)
    if not verdict.eligible:
        return _passthrough(content, verdict.reason, original_size)

    try:
        if verdict.strategy == "image":
            new_bytes = await asyncio.to_thread(
                image_compressor.compress_image, content, verdict.image_format, _cfg()
            )
        else:  # "scanned_pdf"
            new_bytes = await pdf_compressor.compress_scanned_pdf_async(content, _cfg())
    except Exception:  # noqa: BLE001 — fallback is the contract, never propagate
        logger.warning("compression.error", extra={"filename": filename,
                       "strategy": verdict.strategy, "original_size_bytes": original_size})
        result = _passthrough(content, "compression_error", original_size, verdict.strategy)
        metrics.record_compression_fallback(result, surface)
        return result

    # Validation + size-regression fallback (DECISION 9).
    if new_bytes is None or len(new_bytes) >= original_size or not _is_valid_same_format(new_bytes, verdict):
        reason = "size_regression" if new_bytes is not None else "compression_error"
        result = _passthrough(content, reason, original_size, verdict.strategy)
        metrics.record_compression_fallback(result, surface)
        return result

    result = CompressionResult(data=new_bytes, changed=True,
                              decision="compressed", reason=verdict.success_reason,
                              original_size=original_size, new_size=len(new_bytes),
                              strategy=verdict.strategy)
    metrics.record_compression_decision(result, surface)
    return result
```

`_passthrough(...)` builds a `changed=False` result with `data=content` and `new_size=original_size`. `_is_valid_same_format()` re-opens `new_bytes` with Pillow/PyMuPDF to prove the re-encode is decodable in the SAME format (defends the suffix-derived content-type guarantee).

**Caller contract** (so existing behavior is provably unbroken): every caller persists `result.data` and **always succeeds** — there is no error path to add. Genie: `await _save_original_file(session_id, original_filename, result.data)` — one-line swap of the bytes arg at `upload.py:1024-1025`, leaving the single `upload_bytes` write at line 258 and the dup-suffixer at 249-254 untouched (no 2nd write — CORRECTED DECISION #3). ERPHome: build the multipart tuple from `result.data` keeping `content_type` as the 3rd tuple element (not a header).

### 5.5 Shared `is_native_pdf()` (`pdf_classifier.py`) — reuse without changing OCR behavior

The two existing classifiers have **different semantics and control-flow** that MUST be preserved (gotcha from be-ocr): `PDFHandler._has_sufficient_text` uses TOTAL `len(text) >= 50 * page_count` and *branches* (`pdf_handler.py:302-308`); `library_gemini._extract_pdf_native` uses PER-PAGE AVERAGE `avg < 50` and *raises* (`library_gemini.py:613-618`). The helper parameterizes on mode so neither site's verdict changes on mixed-density PDFs:

```python
# app/services/compression/pdf_classifier.py
from typing import Literal

MIN_PDF_TEXT_PER_PAGE = 50  # mirrors PDFHandler.MIN_TEXT_LENGTH & library_gemini.MIN_PDF_TEXT_PER_PAGE

def avg_chars_per_page(total_chars: int, page_count: int) -> float:
    return total_chars / page_count if page_count > 0 else 0.0

def is_native_pdf(
    text: str,
    page_count: int,
    *,
    mode: Literal["total", "average"] = "total",
    threshold: int = MIN_PDF_TEXT_PER_PAGE,
) -> bool:
    """True => text-native (skip OCR fallback / skip compression).
       mode='total'  -> len(text) >= threshold*page_count   (PDFHandler semantics)
       mode='average'-> avg_chars_per_page >= threshold       (library_gemini semantics)"""
    if not text or page_count <= 0:
        return False
    if mode == "total":
        return len(text) >= threshold * page_count
    return avg_chars_per_page(len(text), page_count) >= threshold
```

Delegation, preserving each site exactly:

- `pdf_handler.py:302-308` → `return is_native_pdf(text, page_count, mode="total")` (method kept, value identical).
- `library_gemini.py:613-618` → `if not is_native_pdf(combined_text, page_count, mode="average"): raise FileProcessingException(...)` (raise control-flow kept). Note the library path concatenates only non-empty pages into `pages_text`; pass the same `total_chars`/`combined_text` semantics so the average is byte-identical to today's `total_chars/page_count`.

The compression PDF path calls `is_native_pdf(..., mode="average")` to gate "scanned ⇒ compress, native ⇒ pass-through", keeping the spec's "compress only images + scanned PDFs" in lock-step with the ACTIVE (`library_gemini`) approach's verdict — `ocr_features.json` makes `library_gemini` the live path, so matching its `average` semantics avoids a classifier disagreement between what OCR rasterizes and what compression compresses.

Unchanged: `MIN_TEXT_LENGTH`/`MIN_PDF_TEXT_PER_PAGE` values, 200 DPI, the raise-vs-branch flow. `test_pdf_classifier.py` asserts a mixed-density fixture (one dense page + N empty pages) yields the SAME verdict per-mode as the pre-refactor inline logic.

### 5.6 Per-format compressors

`image_compressor.compress_image(content, fmt, cfg) -> bytes | None`:
- Operates on `BytesIO(content)` — a copy; never the temp file.
- `Image.open` → if `max(w,h) > cfg.image_max_px` (default 1200) downscale by aspect-ratio with `LANCZOS` (same approach as `image_handler.py:152-156`).
- Convert `RGBA`/`P` → `RGB` for JPEG re-encode (mirrors `image_handler.py:159-161`) — but only when target format is JPEG; PNG/WebP keep alpha.
- Re-encode preserving the ORIGINAL container format (JPEG→JPEG quality ladder down to target ≤200KB; PNG→`optimize=True`, fall back to keeping PNG; WebP→quality ladder). Returns `None` if it cannot get under target without format change (caller treats `None` as fallback — never silently changes extension).
- Strip EXIF/metadata for determinism + size (`exif=b""`); deterministic encoder settings only.

`pdf_compressor.compress_scanned_pdf(content, cfg) -> bytes | None` (sync core; `compress_scanned_pdf_async` wraps with `to_thread`/`gather`):
- `fitz.open(stream=content, filetype="pdf")` from the in-memory copy (NOT a path).
- Classify via `is_native_pdf(mode="average")` on extracted text; native ⇒ return `None` (pass-through) so native PDFs are never rebuilt (spec scope: native PDF as-is; container rebuild is the OPTIONAL flag, off by default — `cfg.pdf_rebuild_enabled`).
- Scanned ⇒ rasterize each page at 200 DPI (reuse the `zoom = dpi/72`, `fitz.Matrix` pattern from `pdf_handler.py:310-326`), compress each page image via `image_compressor` to ≤`cfg.pdf_target_kb_per_page` (default 300KB), then rebuild a PDF. Rebuild uses PyMuPDF (`fitz`) page-insert of the compressed JPEGs (pikepdf/img2pdf are NOT available — confirmed; reportlab is a fallback rebuilder). Per-page compression is the `asyncio.gather(*[asyncio.to_thread(...)])` fan-out, bounded by a semaphore sized from `cfg.pdf_max_parallel_pages`.
- Returns `None` on any rebuild failure or if rebuilt ≥ original.

Both compressors are pure functions of (bytes, cfg); no I/O, no logging inside the hot loop (the engine emits one decision/fallback event per call).

### 5.7 Config knobs (`app/config.py`, additive)

```python
# Compression engine (Genie OCR persist + ERPHome attachment surfaces)
compression_enabled: bool = True
compression_min_bytes: int = 500 * 1024        # DECISION 8: < 500KB exempt
compression_image_max_px: int = 1200
compression_image_target_kb: int = 200          # DECISION 7 image cap
compression_pdf_target_kb_per_page: int = 300   # DECISION 7 scanned-PDF cap
compression_pdf_rebuild_enabled: bool = False   # DECISION 7: native-PDF rebuild OFF by default, flag-guarded
compression_pdf_max_parallel_pages: int = 4
compression_pdf_dpi: int = 200
```

These are infra kill-switch / operational-policy booleans and numeric knobs → `Settings` fields, not `FeatureKey` (compression is not a per-tenant customer capability). The 30MB MAX is the upload-side constant (`upload.py:81`), NOT duplicated here; the engine's only size gate is the 500KB MIN. `_cfg()` snapshots these into an immutable `CompressionConfig` so a compress call is deterministic for its duration.

### 5.8 Telemetry (`metrics.py`) — house idioms only

Mirror `tally_sync/diagnostic_metrics.py` (the forward-looking, log-based idiom) and `feature_service.py:328-340` / `metrics.log_form_generated` (`metrics.py:281`) for the combined emit. Do NOT copy `upload.py`/`storage_client.py` styles (f-strings + `'='*60` banners + %-style DEBUG are CLAUDE.md violations):

```python
logger = logging.getLogger(__name__)

def record_compression_decision(result: CompressionResult, surface: CompressionSurface) -> None:
    metrics.compression_decisions_total.inc(strategy=result.strategy, decision=result.decision)
    metrics.compression_bytes_saved.observe(result.saved_bytes)
    logger.info("compression.decision", extra={
        "filename_strategy": result.strategy, "decision": result.decision,
        "reason": result.reason, "surface": surface.value,
        "original_size_bytes": result.original_size, "new_size_bytes": result.new_size,
        "saved_bytes": result.saved_bytes, "saved_pct": round(result.saved_pct, 2)})

def record_compression_fallback(result: CompressionResult, surface: CompressionSurface) -> None:
    metrics.compression_fallback_total.inc(reason=result.reason)
    logger.warning("compression.fallback", extra={...})  # WARNING = recoverable (DECISION 9)
```

Event name first, data in `extra={}`, NO f-strings. Do NOT pass `request_id/tenant_id/user_id/session_id/layer/agent` in `extra` — they auto-inject via `request_context.get_log_context()` / `JSONFormatter`. Log only filename/sizes/strategy/decision — never bytes, tokens, GSTIN, PII. Optional Prometheus registry: add `MetricsRegistry._init_compression_metrics()` (mirror `_init_brs_metrics`, `metrics.py:864`) with `compression_decisions_total{strategy,decision}`, `compression_fallback_total{reason}`, `Histogram compression_bytes_saved`; call it from BOTH `__post_init__` AND `reset_metrics()` (`metrics.py:490/980`) so pytest doesn't leak. Never use `tenant_id` as a registry label (high cardinality).

### 5.9 How existing behavior stays unbroken (explicit)

- OCR untouched: the engine never sees `temp_path`; OCR at `upload.py:1044` still reads the original temp file. `ImageHandler._resize_if_needed` still overwrites the temp (its own copy), not the persisted bytes.
- Single write preserved: only the existing `upload_bytes` at `upload.py:258` runs; the dup-suffixer is not re-triggered (no 2nd write).
- Preview/serve unbroken: filename+extension preserved by the bytes-only contract; `_is_valid_same_format` proves the re-encode is the same container.
- Classifier verdicts unchanged: `mode` parameter keeps total-vs-average + branch-vs-raise per site; tests assert parity.
- ERPHome storage stays ERPNext-only (no GCS sink added); provisioning logo opted out via `compress=False`.
- Failure isolation: engine never raises; both caller surfaces keep their non-blocking / success-envelope contracts.

### 5.10 Checklist

- [ ] Create `app/services/compression/__init__.py` re-exporting `compress_bytes`, `CompressionResult`, `CompressionSurface`, `is_native_pdf`.
- [ ] Create `app/services/compression/types.py` with `CompressionResult` (frozen dataclass, `saved_bytes`/`saved_pct` props), `CompressionSurface` enum, `CompressionDecision`/`CompressionReason` Literals, `CompressionConfig` snapshot.
- [ ] Create `app/services/compression/pdf_classifier.py` with `avg_chars_per_page()` + `is_native_pdf(text, page_count, mode, threshold)`; `MIN_PDF_TEXT_PER_PAGE=50`.
- [ ] Create `app/services/compression/eligibility.py` with `classify(content, filename, content_type, surface)` → verdict: min-size exempt (500KB), magic-byte + suffix format detection, image vs scanned-PDF (delegates to `is_native_pdf(mode="average")`) vs pass-through.
- [ ] Create `app/services/compression/image_compressor.py` with `compress_image(content, fmt, cfg)` on a `BytesIO` copy: LANCZOS downscale to `image_max_px`, RGBA/P→RGB for JPEG, format-preserving quality-ladder re-encode to `image_target_kb`, strip metadata, return `None` if it can't beat target without changing format.
- [ ] Create `app/services/compression/pdf_compressor.py` with sync `compress_scanned_pdf` + async wrapper: `fitz.open(stream=...)` from bytes, native→`None`, scanned→200-DPI rasterize + per-page `image_compressor` to `pdf_target_kb_per_page`, PyMuPDF rebuild (reportlab fallback), `asyncio.gather`+`to_thread` fan-out bounded by `pdf_max_parallel_pages`, `None` on failure/regression; guard rebuild of native PDFs behind `pdf_rebuild_enabled` (default off).
- [ ] Create `app/services/compression/engine.py` `async def compress_bytes(content, *, filename, surface, content_type=None)`: kill-switch/empty/min guards, classify, dispatch via `asyncio.to_thread`, `_is_valid_same_format` revalidation, size-regression + error fallback to original, never raises, never mutates input, emits exactly one telemetry event.
- [ ] Create `app/services/compression/metrics.py` with `record_compression_decision()` / `record_compression_fallback()` (log-based, event-name + `extra`, no f-strings, no request-scoped ids, no PII).
- [ ] Confirm `app/services/compression/` is NOT added to `pyproject.toml [tool.coverage.run].omit` (must be coverage-measured for the 75% gate).
- [ ] Add Settings fields to `app/config.py`: `compression_enabled`, `compression_min_bytes` (500KB), `compression_image_max_px` (1200), `compression_image_target_kb` (200), `compression_pdf_target_kb_per_page` (300), `compression_pdf_rebuild_enabled` (False), `compression_pdf_max_parallel_pages` (4), `compression_pdf_dpi` (200); wire into `CompressionConfig` snapshot.
- [ ] Delegate `pdf_handler.py:_has_sufficient_text` (302-308) to `is_native_pdf(text, page_count, mode="total")` — keep the method, identical return.
- [ ] Delegate `library_gemini.py:_extract_pdf_native` (613-618) average check to `is_native_pdf(combined_text, page_count, mode="average")` — keep the `raise FileProcessingException` control-flow and message.
- [ ] (Optional) Add `MetricsRegistry._init_compression_metrics()` in `app/core/metrics.py`; call from `__post_init__` AND `reset_metrics()`; low-cardinality labels only (`strategy`, `decision`, `reason`); no `tenant_id` label.
- [ ] Create `tests/unit/compression/test_pdf_classifier.py`: assert `mode="total"` vs `mode="average"` parity with pre-refactor logic on a mixed-density fixture (dense page + empty pages); empty text / zero pages → `False`.
- [ ] Create `tests/unit/compression/test_eligibility.py`: <500KB → `below_min_size`; native PDF / docx / xlsx / csv / txt → pass-through; large image → `strategy="image"`; scanned PDF → `strategy="scanned_pdf"`; magic-byte vs suffix mismatch handling.
- [ ] Create `tests/unit/compression/test_image_compressor.py`: large JPEG/PNG/WebP shrink + decodable in SAME format + same extension contract; alpha preserved for PNG/WebP; already-tiny image → `None`; corrupt bytes → `None` (no raise).
- [ ] Create `tests/unit/compression/test_pdf_compressor.py`: synthetic scanned PDF (image-only pages) shrinks and reopens as a valid PDF with same page count; native (text) PDF → `None`; rebuild-disabled flag respected; corrupt PDF → `None`.
- [ ] Create `tests/unit/compression/test_engine.py`: input bytes never mutated (assert identity/content of original); `changed=False` returns `data == original`; size-regression → `size_regression` fallback; compressor exception → `compression_error` fallback and NEVER raises; `compression_enabled=False` → `disabled` pass-through; empty input → `empty_input`; `CompressionResult.saved_pct`/`saved_bytes` math.
- [ ] Run `ruff format --check app/ tests/ && ruff check app/ tests/ && mypy app/ --ignore-missing-imports && pytest tests/unit/compression -p no:warnings` clean before wiring callers.

---

I have confirmed all the seam symbols, library availability, and classifier semantics. Writing the section now.

## 6. Core Compressors: Image + Scanned PDF

This section specifies the two pure-logic compressors that turn over-sized in-memory upload bytes into smaller bytes of the **same extension/content-type**, plus the classifier/dispatch helper that decides *whether* to compress. All compressors operate on a **copy of in-memory `content`/`file_content` bytes via `io.BytesIO`** — never `temp_path` — because `ImageHandler._resize_if_needed` (`image_handler.py:139-167`, `img.save(file_path)` at line 162) overwrites the OCR temp file in place before OCR, and the PDF page-count validation + OCR both read `temp_path`. Honoring CORRECTED DECISIONS 1-4 + 9: compress a copy, OCR reads the original, storage saves the compressed bytes, same filename/extension preserved, and any failure/size-regression silently falls back to the original bytes (logged, never raised).

These modules are **net-new and standalone** (no behavioral edit to OCR handlers in this section — only a delegate refactor of the shared classifier, described in §6.3). They must live OUTSIDE `app/api/v1/endpoints/upload.py` and the OCR engine/handler files, all of which are in `pyproject.toml [tool.coverage.run].omit` (so code placed there is not coverage-measured). The package `app/services/file_processing/compression/` is NOT in the omit list, so logic there counts toward the 75% gate.

### 6.1 Files to CREATE / MODIFY

CREATE (new package — package-first per modularity rule, ≥3 sub-domains: classifier, image, pdf):

- `/Users/bhags/Desktop/erpsense_all/erpsense-backend/app/services/file_processing/compression/__init__.py` — re-exports the public surface: `compress_upload`, `CompressionResult`, `CompressionStrategy`, `should_compress`, `MediaKind`, `classify_media`.
- `/Users/bhags/Desktop/erpsense_all/erpsense-backend/app/services/file_processing/compression/models.py` — `CompressionResult` dataclass + `CompressionStrategy`/`MediaKind` enums + `CompressionError` (subclass of `FileProcessingException`).
- `/Users/bhags/Desktop/erpsense_all/erpsense-backend/app/services/file_processing/compression/config.py` — module-level thresholds re-read from `app/config.py` Settings (NO `os.getenv` — R03). Constants: target sizes, max dimensions, quality ladders, image-bomb caps, DPI.
- `/Users/bhags/Desktop/erpsense_all/erpsense-backend/app/services/file_processing/compression/classifier.py` — `MediaKind` classification (`image` / `scanned_pdf` / `native_pdf` / `passthrough`) + `should_compress()` + the shared `is_native_pdf()`/`avg_chars_per_page()` helpers (the de-dup target for §6.3). Estimated ~120 lines.
- `/Users/bhags/Desktop/erpsense_all/erpsense-backend/app/services/file_processing/compression/image_compressor.py` — `compress_image(content: bytes, suffix: str) -> CompressionResult`. ~150 lines.
- `/Users/bhags/Desktop/erpsense_all/erpsense-backend/app/services/file_processing/compression/pdf_compressor.py` — `compress_scanned_pdf(content: bytes) -> CompressionResult`. ~170 lines.
- `/Users/bhags/Desktop/erpsense_all/erpsense-backend/app/services/file_processing/compression/orchestrator.py` — `compress_upload(content, filename, content_type) -> CompressionResult` (classify → dispatch → fallback → telemetry), plus the `asyncio.to_thread`/`asyncio.gather` async wrappers so the synchronous-inline request (DECISION 6) does not block the event loop. ~140 lines.

MODIFY (config + the de-dup refactor):

- `/Users/bhags/Desktop/erpsense_all/erpsense-backend/app/config.py` — add Settings fields (§6.2). The two existing limit surfaces (`max_upload_size_mb=50` at 240, `max_file_size_bytes=10MB` at 525) are NOT touched here; compression thresholds are new, named after the subsystem (`compression_*`), default ON.
- `/Users/bhags/Desktop/erpsense_all/erpsense-backend/app/services/file_processing/handlers/pdf_handler.py` — `PDFHandler._has_sufficient_text` (line 302-308) body replaced with a thin delegate to `classifier.is_native_pdf(text, page_count, mode="total")` keeping `len(text) >= MIN_TEXT_LENGTH * page_count` semantics EXACTLY. Constants and control flow untouched.
- `/Users/bhags/Desktop/erpsense_all/erpsense-backend/app/services/file_processing/ocr/library_gemini.py` — the inline `avg_chars_per_page < self.MIN_PDF_TEXT_PER_PAGE` check (line 613-614) computes via `classifier.avg_chars_per_page(text, page_count)` then still `raise FileProcessingException(...)` (PER-PAGE-AVERAGE semantics + raise control-flow preserved per the be-ocr gotcha).

NOTE — wiring of `compress_upload()` into the Genie persist seam (`upload.py:_save_original_file` call at 1024-1025) and the ERPHome adapter hook (`_attachments.py`) is covered in the integration sections of the plan, NOT here. §6 produces the pure compressors those seams call.

### 6.2 Config (`compression/config.py` + `app/config.py`)

Add to `app/config.py` Settings (defaults satisfy the spec; the rebuild flag is OFF per DECISION 7):

```python
# --- Upload compression (Genie chat originals + ERPHome attachments) ---
compression_enabled: bool = True                 # master kill-switch
compression_min_bytes: int = 500 * 1024          # DECISION 8: <500KB exempt (net-new)
compression_image_target_bytes: int = 200 * 1024 # spec: images <=200KB
compression_image_max_dimension: int = 1200      # spec: images <=1200px longest side
compression_pdf_target_bytes_per_page: int = 300 * 1024  # spec: scanned PDF <=300KB/page
compression_pdf_dpi: int = 200                   # matches PDFHandler._pdf_to_images dpi
compression_container_rebuild_enabled: bool = False  # DECISION 7: native PDF/Office rebuild guarded, OFF
compression_max_image_pixels: int = 64_000_000   # image-bomb guard (~8000x8000)
compression_max_pdf_pages: int = 50              # decompression-bomb page cap for compressor
```

`compression/config.py` imports `from app.config import settings` and exposes module-level constants derived from those fields plus the JPEG quality ladder `IMAGE_JPEG_QUALITY_LADDER = (85, 75, 65, 55, 45)` and `PNG_QUANTIZE_COLORS = 256`. Reading via `app.config` keeps it consistent with the be-logging gotcha (no `os.getenv` outside config).

### 6.3 Classifier + shared `is_native_pdf` (`compression/classifier.py`)

The classifier decides which compressor (if any) runs, and is the single home for the native-vs-scanned PDF semantics that today live in TWO places with TWO different semantics:

- `PDFHandler._has_sufficient_text` (`pdf_handler.py:302-308`) — TOTAL: `len(text) >= MIN_TEXT_LENGTH * page_count`, branches (no raise).
- `LibraryGeminiProcessor._extract_pdf_native` (`library_gemini.py:613-614`) — PER-PAGE AVERAGE: `avg = total_chars/page_count < MIN_PDF_TEXT_PER_PAGE`, raises `FileProcessingException`.

Per the be-ocr gotcha, a naive single helper would silently change one site's behavior on mixed-density PDFs. The helper is therefore **parameterized on mode** so each call site keeps its exact semantics:

```python
def avg_chars_per_page(text: str, page_count: int) -> float:
    return (len(text) / page_count) if page_count > 0 else 0.0

def is_native_pdf(text: str, page_count: int, *, threshold: int = 50,
                  mode: Literal["total", "average"] = "total") -> bool:
    if not text:
        return False
    if mode == "average":
        return avg_chars_per_page(text, page_count) >= threshold
    return len(text) >= threshold * page_count   # PDFHandler semantics
```

`PDFHandler._has_sufficient_text` becomes `return is_native_pdf(text, page_count, threshold=self.MIN_TEXT_LENGTH, mode="total")`; `library_gemini` keeps `if avg_chars_per_page(text, page_count) < self.MIN_PDF_TEXT_PER_PAGE: raise FileProcessingException(...)`. Values (`MIN_TEXT_LENGTH=50`, `MIN_PDF_TEXT_PER_PAGE=50`), the raise-vs-branch flow, and 200 DPI are all unchanged — OCR behavior is byte-for-byte preserved.

`classify_media(filename, content_type, content) -> MediaKind`:

1. Resolve `suffix = Path(filename).suffix.lower()`; resolve a normalized kind from suffix first, `content_type` as tiebreaker.
2. Image suffixes (mirror `ImageHandler.SUPPORTED_EXTENSIONS` at `image_handler.py:39`: `.png .jpg .jpeg .gif .bmp .tiff .webp`) → `MediaKind.IMAGE`.
3. `.pdf` → open a copy via `fitz.open(stream=content, filetype="pdf")` in a thread, extract text per page, compute native-vs-scanned with `is_native_pdf(..., mode="total")` (compression uses TOTAL semantics — conservative: a PDF the OCR PDFHandler considers native will not be re-rastered). Scanned → `MediaKind.SCANNED_PDF`; native → `MediaKind.NATIVE_PDF`.
4. Everything else (`.txt .md .csv .docx .xlsx .xls .tex`) → `MediaKind.PASSTHROUGH`.

`should_compress(filename, content_type, content) -> bool`:

```python
if not settings.compression_enabled:            return False
if len(content) < settings.compression_min_bytes:  return False   # DECISION 8: 500KB exempt
kind = classify_media(...)
if kind is MediaKind.IMAGE:        return True
if kind is MediaKind.SCANNED_PDF:  return True
if kind is MediaKind.NATIVE_PDF:   return settings.compression_container_rebuild_enabled  # OFF by default
return False   # PASSTHROUGH (native PDF/DOCX/XLSX/CSV) — DECISION 7
```

### 6.4 Image compressor (`compression/image_compressor.py`)

Pure function on a `BytesIO` copy. Replicates `ImageHandler._resize_if_needed`'s RGBA/P→RGB conversion idiom (`image_handler.py:159-161`) so JPEG output stays valid, but writes to a fresh buffer — never the temp file.

```python
def compress_image(content: bytes, suffix: str) -> CompressionResult:
    # Guard: decompression / image-bomb before decode
    Image.MAX_IMAGE_PIXELS = settings.compression_max_image_pixels   # raises DecompressionBombError above
    with Image.open(io.BytesIO(content)) as img:
        img.load()                                  # force decode under the pixel cap
        img = ImageOps.exif_transpose(img)          # apply EXIF orientation, then drop it
        # Resize longest side to <=1200px (LANCZOS), preserve aspect ratio
        longest = max(img.width, img.height)
        if longest > settings.compression_image_max_dimension:
            ratio = settings.compression_image_max_dimension / longest
            img = img.resize((round(img.width*ratio), round(img.height*ratio)),
                             Image.Resampling.LANCZOS)
        ...
```

Format-specific encode (same extension always — preview/serve content-type is suffix-derived per be-storage `serve_original_file` and FE `EXT_TO_MIME`/`buildPreviewUrl`):

- **JPEG (`.jpg/.jpeg`)**: flatten transparency onto white if `RGBA/P/LA` → `RGB`; encode walking `IMAGE_JPEG_QUALITY_LADDER` (85→45) with `optimize=True, progressive=True`, stop at the first quality whose output `<= compression_image_target_bytes`. If even q=45 is over target, keep the q=45 result (best effort; the >=original fallback in §6.6 still applies).
- **PNG (`.png`)**: first try `save(optimize=True)`. If still over target, `img.quantize(colors=256, method=Image.Quantize.FASTOCTREE)` then `save(optimize=True)`. PNG stays PNG (lossless-ish) — never silently re-encoded to JPEG (would change the extension/content-type and break preview).
- **WEBP (`.webp`)**: `save(format="WEBP", quality=…)` walking the ladder; `method=6`.
- **GIF (`.gif`)**: if animated (`getattr(img, "is_animated", False)` or `n_frames > 1`) → **passthrough** (return original; resizing animated GIFs frame-by-frame is out of spec). Static GIF → resize + `save(optimize=True)`.
- **BMP (`.bmp`) / TIFF (`.tiff`)**: re-encode in the same format with the resize applied; if no size win, fallback.

Return `CompressionResult(data=out_bytes, strategy=IMAGE, original_size, compressed_size, applied=…)`. The `applied` flag is set in the orchestrator after the size comparison (§6.6).

Edge cases handled: corrupt/undecodable image → `Image.open`/`load()` raises → caught by orchestrator → fallback to original. `DecompressionBombError` / `DecompressionBombWarning` (Pillow) → fallback (never OOM). Zero-frame / 1×1 images → resize is a no-op, encode still runs, size check decides.

### 6.5 Scanned-PDF compressor (`compression/pdf_compressor.py`)

pikepdf and img2pdf are NOT available (confirmed in requirements.txt); rebuild uses **PyMuPDF (fitz) + Pillow** only (reportlab is available but PyMuPDF rebuild is simpler and keeps a real PDF). Per-page rasterize at 200 DPI (matching `PDFHandler._pdf_to_images` at `pdf_handler.py:310-326`), JPEG-compress each page to ≤300KB, rebuild a fresh single-PDF, validate it opens.

```python
def compress_scanned_pdf(content: bytes) -> CompressionResult:
    src = fitz.open(stream=content, filetype="pdf")
    try:
        page_count = src.page_count
        if page_count == 0 or page_count > settings.compression_max_pdf_pages:
            raise CompressionError("page count out of bounds")   # -> fallback
        out = fitz.open()                              # new empty PDF
        zoom = settings.compression_pdf_dpi / 72       # 200 DPI matrix (== PDFHandler)
        matrix = fitz.Matrix(zoom, zoom)
        for page in src:
            pix = page.get_pixmap(matrix=matrix, alpha=False)
            with Image.frombytes("RGB", (pix.width, pix.height), pix.samples) as pil:
                jpeg = _encode_page_under_target(pil)  # quality ladder -> <=300KB/page
            rect = page.rect                           # preserve original page geometry
            new_page = out.new_page(width=rect.width, height=rect.height)
            new_page.insert_image(rect, stream=jpeg)
        rebuilt = out.tobytes(garbage=4, deflate=True)
    finally:
        src.close(); out.close()
    # Validate the rebuilt PDF actually opens (else fallback)
    with fitz.open(stream=rebuilt, filetype="pdf") as check:
        if check.page_count != page_count:
            raise CompressionError("rebuilt page-count mismatch")
    return CompressionResult(data=rebuilt, strategy=SCANNED_PDF, ...)
```

`_encode_page_under_target(pil)` walks `IMAGE_JPEG_QUALITY_LADDER` (85→45) until the JPEG bytes are `<= compression_pdf_target_bytes_per_page`; keeps the lowest-quality result if still over (best-effort). Pages stay JPEG-in-PDF, suffix stays `.pdf`.

Edge cases: encrypted/password PDF → `fitz.open` raises `fitz.FileDataError` (or `needs_pass`) → caught → fallback. Corrupt PDF → fallback. Page-count over `compression_max_pdf_pages` (50) → fallback (decompression-bomb / OOM guard; the upload surface already caps at `MAX_PAGES_PER_PDF=10` so this is a defense-in-depth cap). Pixmap with alpha is forced off (`alpha=False`) so `Image.frombytes("RGB", …)` matches the existing rasterizer's RGB assumption. `get_pixmap` failing on a malformed page → exception → whole-file fallback (no partial PDF).

### 6.6 Orchestrator + async wrapping + fallback (`compression/orchestrator.py`)

`compress_upload` is the single entry point both seams call. It is **async** so it can offload CPU-bound Pillow/PyMuPDF work off the event loop within the synchronous-inline request (DECISION 6); "parallel" = `asyncio.to_thread` (and `asyncio.gather` when a caller compresses multiple attachments, e.g. employee_expense/employee_travel loops).

```python
async def compress_upload(content: bytes, filename: str,
                          content_type: str | None) -> CompressionResult:
    original_size = len(content)
    if not should_compress(filename, content_type, content):
        return CompressionResult.passthrough(content, reason="not_eligible")
    kind = classify_media(filename, content_type, content)
    try:
        if kind is MediaKind.IMAGE:
            result = await asyncio.to_thread(compress_image, content, Path(filename).suffix.lower())
        elif kind is MediaKind.SCANNED_PDF:
            result = await asyncio.to_thread(compress_scanned_pdf, content)
        else:
            return CompressionResult.passthrough(content, reason="not_eligible")
    except Exception as exc:                       # DECISION 9: never raise, log + fallback
        record_compression_fallback(reason="error", filename=filename,
                                    original_size=original_size, error=type(exc).__name__)
        return CompressionResult.passthrough(content, reason="error")
    # DECISION 9: compressed >= original -> store original
    if result.compressed_size >= original_size:
        record_compression_fallback(reason="size_regression", filename=filename,
                                    original_size=original_size,
                                    compressed_size=result.compressed_size)
        return CompressionResult.passthrough(content, reason="size_regression")
    record_compression_decision(strategy=result.strategy.value, filename=filename,
                                original_size=original_size,
                                compressed_size=result.compressed_size)
    return result
```

`CompressionResult.passthrough(content, reason)` returns `data=content, applied=False, strategy=PASSTHROUGH` so the calling seam always gets bytes to persist (the original on any fallback). The telemetry helpers `record_compression_decision` / `record_compression_fallback` are the be-logging-style structured emitters (event names `compression.decision` / `compression.fallback`, sizes/strategy/saved_pct in `extra={}`, no PII, no f-strings) — defined in the logging/observability section, imported here. Crucially, `compress_upload` **never raises**: it is called inside `_save_original_file` (already non-blocking, returns None on failure) and inside the ERPHome adapter try/except, so a compression failure must never block OCR or the attachment POST.

How existing behavior stays unbroken:

- OCR pipeline untouched: only `_has_sufficient_text` (delegate) and the `library_gemini` avg check (delegate) change, both byte-for-byte equivalent. `temp_path` is never read or written by any compressor — `compress_upload` takes `bytes`, decodes a `BytesIO`/`fitz` stream copy. So `ImageHandler._resize_if_needed`'s in-place overwrite (the most dangerous seam) is irrelevant to this module.
- Filename + extension preserved end to end → `serve_original_file` (`upload.py`) MIME_TYPE_MAP (suffix-derived), the dup-filename suffixer (`_save_original_file:249-254`), the cancel-purge matcher (`_purge_upload_artifacts`), and the FE `buildPreviewUrl`/`EXT_TO_MIME` preview routing all keep working unchanged.
- PASSTHROUGH default for native PDF/DOCX/XLSX/CSV → those files persist exactly as today (the rebuild path is behind `compression_container_rebuild_enabled=False`).
- `compression_enabled=False` short-circuits to passthrough → a clean kill-switch that restores 100% current behavior.

### 6.7 Checklist

- [ ] Create package dir `app/services/file_processing/compression/` with `__init__.py` re-exporting `compress_upload`, `CompressionResult`, `CompressionStrategy`, `MediaKind`, `classify_media`, `should_compress`, `is_native_pdf`, `avg_chars_per_page`.
- [ ] Add `compression/models.py`: `CompressionResult` dataclass (`data: bytes`, `strategy`, `original_size`, `compressed_size`, `applied`, `reason`) with `passthrough()` classmethod; `CompressionStrategy`/`MediaKind` enums; `CompressionError(FileProcessingException)`.
- [ ] Add Settings fields to `app/config.py`: `compression_enabled`, `compression_min_bytes`, `compression_image_target_bytes`, `compression_image_max_dimension`, `compression_pdf_target_bytes_per_page`, `compression_pdf_dpi`, `compression_container_rebuild_enabled`, `compression_max_image_pixels`, `compression_max_pdf_pages`.
- [ ] Add `compression/config.py` deriving module constants from `settings` (no `os.getenv`); define `IMAGE_JPEG_QUALITY_LADDER=(85,75,65,55,45)`, `PNG_QUANTIZE_COLORS=256`.
- [ ] Add `compression/classifier.py`: `avg_chars_per_page`, parameterized `is_native_pdf(..., mode="total"|"average")`, `classify_media`, `should_compress` (enforces `compression_enabled` + 500KB min + per-kind gating, rebuild flag for native PDF).
- [ ] Image suffix set in classifier must mirror `ImageHandler.SUPPORTED_EXTENSIONS` (`image_handler.py:39`).
- [ ] PDF classification opens a `fitz.open(stream=..., filetype="pdf")` COPY in a thread; never reads a temp path.
- [ ] Add `compression/image_compressor.py`: `compress_image(content, suffix)` — `Image.MAX_IMAGE_PIXELS` bomb guard, `ImageOps.exif_transpose`, RGBA/P→RGB flatten for JPEG, LANCZOS resize to ≤1200px longest side, JPEG quality ladder to ≤200KB, PNG `optimize`→`quantize(256)` fallback, WEBP/BMP/TIFF re-encode, animated-GIF passthrough.
- [ ] Image compressor always keeps the original extension; PNG never silently converted to JPEG.
- [ ] Add `compression/pdf_compressor.py`: `compress_scanned_pdf(content)` — page cap ≤50, 200 DPI `fitz.Matrix` (matches `PDFHandler._pdf_to_images`), `get_pixmap(alpha=False)`→`Image.frombytes("RGB",…)`, per-page JPEG ladder to ≤300KB, rebuild new PDF preserving page geometry, validate rebuilt PDF opens + page-count matches.
- [ ] PDF compressor handles encrypted/corrupt/zero-page/over-cap PDFs by raising `CompressionError`/letting fitz raise → orchestrator fallback.
- [ ] Add `compression/orchestrator.py`: async `compress_upload(content, filename, content_type)` — `should_compress` gate, classify, `asyncio.to_thread` dispatch, try/except → fallback (DECISION 9 error), `compressed_size >= original_size` → fallback (DECISION 9 regression), emit `compression.decision`/`compression.fallback` telemetry, NEVER raise.
- [ ] `CompressionResult.passthrough` returns original bytes with `applied=False` on every fallback path so callers always persist valid bytes.
- [ ] MODIFY `pdf_handler.py:_has_sufficient_text` (302-308) → `return classifier.is_native_pdf(text, page_count, threshold=self.MIN_TEXT_LENGTH, mode="total")`; keep `MIN_TEXT_LENGTH` and branch control flow.
- [ ] MODIFY `library_gemini.py` (613-614) → compute via `classifier.avg_chars_per_page(text, page_count)`, keep `MIN_PDF_TEXT_PER_PAGE` threshold AND the `raise FileProcessingException(...)` control flow.
- [ ] Verify the delegate refactor changes no OCR thresholds (50/50), no DPI (200), no raise-vs-branch behavior — run existing OCR tests if any reach these (note OCR files are coverage-omitted).
- [ ] Confirm no compressor reads/writes `temp_path`; all operate on `io.BytesIO`/`fitz` stream copies of in-memory bytes.
- [ ] Confirm the compression package is NOT added to `pyproject.toml [tool.coverage.run].omit` (so it counts toward the 75% gate).
- [ ] Confirm filename + extension are preserved by every compressor (suffix-derived serve content-type + FE preview depend on it).
- [ ] Run `ruff format/check app/services/file_processing/compression/` and `mypy app/ --ignore-missing-imports` clean on the new package.

---

I have what I need. The config uses subsystem-named env booleans (not `feature_*` prefix), reportlab/Pillow/PyMuPDF/python-docx/openpyxl are present, pikepdf/img2pdf are not. Writing the section now.

## 7. Optional Container Rebuild Engine (DOCX / XLSX / native-PDF) [flagged, default OFF]

> **Status: documented design, default OFF, out of the v1 happy-path.** The spec scope (corrected decision #7) is *compress ONLY images (≤200KB, ≤1200px) + scanned PDFs (≤300KB/page); native PDF/DOCX/XLSX/CSV are stored as-is.* This section specifies the **optional** container-rebuild engine for the as-is types so it can be turned on later behind a flag **without touching the v1 wiring** at `app/api/v1/endpoints/upload.py:_save_original_file` (225-264) or the adapter hook at `app/adapters/erpnext/_attachments.py`. When the flag is OFF (the default), this engine is never imported into the request path and these container types pass through byte-for-byte exactly as today. CSV is intentionally excluded from rebuild (it is already minimal text; the only "compression" would be gzip, which breaks the suffix-derived content-type and is therefore disallowed by corrected decision #4).

### 7.1 Why this is a separate engine (and separate from image/scanned-PDF compression)

The image + scanned-PDF compressor in Sections 4-5 operates on a **single media stream** decoded into a `BytesIO` and re-encoded (Pillow for images; PyMuPDF 200-DPI rasterize + reportlab rebuild for scanned PDFs, mirroring `PDFHandler._pdf_to_images` at `app/services/file_processing/handlers/pdf_handler.py:310-326`). OOXML containers (DOCX, XLSX) and native (text-layer) PDFs are **structurally different**: they are zip archives (OOXML) or object/xref graphs (PDF) whose bulk is usually embedded media, not the text. You cannot "re-encode" them as one stream; you must **open the container, recompress the media members, and repack the same container format**. That is a distinct algorithm with its own corruption surface, which is exactly why it is flag-gated and validate-or-fallback.

This engine reuses the native-vs-scanned verdict from the shared classifier introduced in Section 5 (`app/services/file_processing/handlers/_pdf_classifier.py:is_native_pdf`, delegated from `PDFHandler._has_sufficient_text` at `pdf_handler.py:302-308` and `LibraryGeminiProcessor._extract_pdf_native` at `app/services/file_processing/ocr/library_gemini.py:589-618`). A PDF only reaches **this** engine's PDF branch when `is_native_pdf(...) is True`; scanned PDFs continue to flow through the Section 5 rasterize-rebuild path. There is no double-classification: the compression orchestrator calls `is_native_pdf` exactly once per PDF and routes to scanned-compressor OR (if the flag is on) this native-rebuilder, never both.

### 7.2 Library reality (confirmed against `requirements.txt`)

Confirmed present: `PyMuPDF>=1.23.0` (line 73), `python-docx>=1.0.0` (74), `openpyxl>=3.1.0` (75), `Pillow>=10.0.0` (77), `reportlab>=4.1` (90). **Confirmed absent:** `pikepdf`, `img2pdf` (0 matches). Consequences for this engine:

- **OOXML rebuild** needs only the **stdlib `zipfile`** (unzip/repack) + **Pillow** (recompress `word/media/*`, `xl/media/*`) + **`python-docx` / `openpyxl`** (open-to-validate). No new dependency.
- **Native-PDF stream/font optimization** is the one branch that genuinely wants `pikepdf` (lossless object-stream + xref compaction, font subsetting). Since `pikepdf` is **not** in requirements, the native-PDF rebuild branch is split:
  - **Tier A (no new dep, ships under the flag):** PyMuPDF lossless save — `doc.save(out, garbage=4, deflate=True, deflate_images=True, deflate_fonts=True, clean=True)`. This is a real win on PDFs with uncompressed streams and dead objects, uses only `fitz` (already present), and never rasterizes a native PDF (text layer preserved → OCR still classifies it native).
  - **Tier B (requires adding `pikepdf>=8` to `requirements.txt` — itself gated):** deeper object-stream rebuild + font subsetting. Tier B is **double-gated**: off unless both the rebuild flag AND a `pikepdf` availability check pass (`try: import pikepdf except ImportError: tier_b = False`). Adding the dependency is its own reviewed change (see global-tech-stack); the engine degrades gracefully to Tier A when `pikepdf` is unavailable.

### 7.3 Config flags (subsystem-named env booleans in `app/config.py`, per CLAUDE.md)

Per the logging/config rule (be-logging gotcha: knobs belong in `app/config.py`, never `os.getenv`; and per CLAUDE.md the `feature_*` prefix is reserved for the deprecated per-tenant pattern — new env kill-switches are named after the subsystem). **MODIFY** `/Users/bhags/Desktop/erpsense_all/erpsense-backend/app/config.py` — add to `class Settings` (mirroring the existing `bool` defaults around lines 354-471):

```python
# --- Optional container-rebuild compression (DOCX / XLSX / native-PDF). Default OFF. ---
container_rebuild_enabled: bool = False            # master gate; OFF = pass-through, engine never imported
container_rebuild_docx_enabled: bool = False       # recompress word/media/* in .docx zip
container_rebuild_xlsx_enabled: bool = False       # recompress xl/media/* in .xlsx zip
container_rebuild_native_pdf_enabled: bool = False # PyMuPDF lossless save (Tier A); Tier B needs pikepdf
container_rebuild_pikepdf_tier_b_enabled: bool = False  # double-gated; degrades to Tier A if pikepdf absent
container_rebuild_media_quality: int = Field(82, ge=40, le=95)   # JPEG quality for recompressed media
container_rebuild_media_max_px: int = Field(1600, ge=512, le=4096)  # downscale ceiling for embedded media
container_rebuild_min_saving_pct: float = Field(10.0, ge=0.0, le=90.0)  # min % saved to accept rebuild
container_rebuild_timeout_seconds: float = Field(20.0, ge=1.0, le=120.0)  # wall-clock budget per file
```

The master `container_rebuild_enabled` is checked **before importing** the engine module, so when OFF there is zero import-time/runtime cost and zero new failure surface on the hot path. The per-type sub-flags allow enabling DOCX while leaving the riskier native-PDF off. `min_saving_pct` enforces corrected decision #9 at a stricter-than-zero threshold (a 1% saving is not worth the corruption risk for a container rebuild). The quality/px/timeout knobs are validated with `Field(ge=, le=)` per global-validation.

### 7.4 Files to CREATE

All new files live **outside** the coverage-omit set (be-storage / tests-ci: `upload.py` and the OCR handlers under `file_processing/handlers/` + `file_processing/ocr/` are `[tool.coverage.run].omit` in `pyproject.toml:145+`). To count toward the 75% gate, the engine goes in a **new, non-omitted** subpackage `app/services/file_processing/compression/`:

- **`app/services/file_processing/compression/__init__.py`** — re-exports `rebuild_container`, `RebuildResult`, `RebuildSkipped`. Package-first per global-modularity (≥3 sub-concerns: docx, xlsx, pdf).
- **`app/services/file_processing/compression/container_rebuild.py`** (≤500 lines — modularity cap for new files) — the orchestrator `rebuild_container(...)` + the generic 8-step OOXML routine + dispatch by extension. Pure-logic, no FastAPI, no DB; takes/returns `bytes`.
- **`app/services/file_processing/compression/_ooxml.py`** — the shared OOXML zip walk (`_recompress_zip_members`), media filter, and per-member Pillow recompress (reuses the RGBA/P→RGB conversion pattern from `ImageHandler._resize_if_needed` at `app/services/file_processing/handlers/image_handler.py:159-161`, but on `BytesIO`, never a temp path).
- **`app/services/file_processing/compression/_native_pdf.py`** — Tier A (`fitz` lossless save) + optional Tier B (`pikepdf`, import-guarded).
- **`app/services/file_processing/compression/_validate.py`** — post-rebuild open-validators: `validate_docx(bytes)` via `python-docx`, `validate_xlsx(bytes)` via `openpyxl`, `validate_pdf(bytes)` via `fitz` (assert page-count parity + every page opens). Any exception ⇒ rebuild rejected.
- **`app/services/file_processing/compression/compression_metrics.py`** — log-based metric wrappers (mirrors `app/services/tally_sync/diagnostic_metrics.py`): `record_rebuild(...)` emitting event names `compression.rebuild.decision` / `compression.rebuild.fallback` with `extra={original_size, rebuilt_size, saved_pct, container, tier, decision, fallback_reason, duration_ms}`. Event-name-first, all data in `extra={}`, no f-strings, no request-scoped ids (auto-injected) — per be-logging.

> Note: the **image + scanned-PDF** compressor from Sections 4-5 (`compress_image`, `compress_scanned_pdf`) also lives in this `compression/` package — this engine is the optional sibling, sharing the package, the metrics module, and the classifier delegate.

### 7.5 Files to MODIFY (single integration seam — guarded, so OFF is a no-op)

There is exactly **one** wiring point, and it is the same place the v1 image/scanned-PDF compressor plugs in (be-upload integration point), so the rebuild engine adds **no new seam**:

- **`/Users/bhags/Desktop/erpsense_all/erpsense-backend/app/api/v1/endpoints/upload.py`** — inside the compression orchestrator call that runs after `content = await file.read()` (line 864) and after the PDF page-count validation block (ends ~1019), feeding `_save_original_file` at lines 1024-1025. The orchestrator decides: image → Section 4; scanned PDF → Section 5; **native PDF / DOCX / XLSX → IF `settings.container_rebuild_enabled` and the matching sub-flag → call `rebuild_container(...)`, ELSE pass `content` through unchanged.** The compressed-or-original bytes are then handed to `_save_original_file` (the single `upload_bytes` write at line 258), exactly as decided in #3. OCR at line 1044 still reads the untouched `temp_path` (#2). When the master flag is OFF, the `rebuild_container` import is not executed and the branch is a straight pass-through — **existing native-PDF/DOCX/XLSX behavior is byte-identical to today.**

- **`/Users/bhags/Desktop/erpsense_all/erpsense-backend/app/config.py`** — the flags in §7.3.

- **(Optional, only if enabling on the ERPHome surface too)** `/Users/bhags/Desktop/erpsense_all/erpsense-backend/app/adapters/erpnext/_attachments.py` — the same `if settings.container_rebuild_enabled` guard at the `compress` hook before the multipart build in `upload_attachment` (108) / `upload_file_standalone` (196), honoring the existing `compress: bool = True` opt-out (be-erphome) so the provisioning logo push at `_steps_company.py:335` is excluded. **Recommended: leave OFF on ERPHome** — that surface persists to ERPNext only (#5) and its 10MB cap (`_shared.py:111`) makes container rebuild low-value.

No new endpoint, no migration, no router change → the Alembic single-head gate and route-registration order are untouched.

### 7.6 The generic 8-step OOXML rebuild (DOCX / XLSX)

`rebuild_container(content: bytes, filename: str, content_type: str) -> RebuildResult` runs entirely on an **in-memory copy** of `content` (corrected decision #1; never the temp path — the `ImageHandler._resize_if_needed` in-place-overwrite footgun at `image_handler.py:162` must not be repeated). Signature:

```python
@dataclass(frozen=True)
class RebuildResult:
    bytes_out: bytes          # rebuilt OR original (caller stores this)
    rebuilt: bool             # True only if rebuild applied AND accepted
    original_size: int
    output_size: int
    fallback_reason: str | None  # set when rebuilt is False but a rebuild was attempted

def rebuild_container(content: bytes, filename: str, content_type: str) -> RebuildResult: ...
```

The 8 steps (DOCX/XLSX share them; the PDF branch in §7.7 is different):

1. **Guard + classify.** If master/sub-flag off → return `RebuildResult(content, rebuilt=False, ...)` immediately (pass-through). Derive container kind from the **suffix** (`.docx`/`.xlsx`) — must match the value `_save_original_file` will persist so the suffix-derived serve content-type (`serve_original_file` at `upload.py:1927-2004`, `MIME_TYPE_MAP`) stays correct (#4).
2. **Open the zip read-only from a `BytesIO(content)`** with `zipfile.ZipFile`. **Corruption guard #1:** wrap in `try/except (zipfile.BadZipFile, OSError)` → fallback (return original, `fallback_reason="not_a_zip"`). Run `zf.testzip()`; a non-`None` result means a CRC-bad member → fallback.
3. **Enumerate members; identify media** by prefix (`word/media/`, `xl/media/`, `ppt/media/`) and image extension. **Corruption guard #2 (zip-bomb / path-traversal):** reject any member whose name contains `..`, is absolute, or whose `ZipInfo.file_size` exceeds a sane per-member ceiling (e.g. 64MB) or whose total decompressed size exceeds a whole-archive ceiling → fallback. This is the standard decompression-bomb defense; the engine never trusts the archive's declared sizes blindly.
4. **Recompress each media member on a `BytesIO` copy** via `_ooxml._recompress_media`: open with Pillow, convert RGBA/P→RGB for JPEG targets (mirroring `image_handler.py:159-161`), downscale to `container_rebuild_media_max_px`, re-encode at `container_rebuild_media_quality`. **Per-member fallback:** if Pillow raises or the recompressed member is **not smaller**, keep the **original member bytes** (never grow a member). PNG with alpha or already-optimized media is left untouched. SVG/EMF/WMF vector members are passed through verbatim (rasterizing them would degrade quality and is out of scope).
5. **Repack into a fresh `BytesIO` `zipfile.ZipFile(mode="w")`**, preserving every member's **archive name and order**, copying non-media members **byte-for-byte** (do not re-deflate XML — re-deflating can subtly change `[Content_Types].xml` ordering and gains little). Media members get the recompressed bytes; all others get the verbatim original bytes. **Critical: write `[Content_Types].xml` and `_rels/` exactly as-is** — these are the OOXML manifest; corrupting them breaks Word/Excel/Frappe preview.
6. **Same extension, same filename.** The output keeps `.docx`/`.xlsx` (NEVER `.zip`) so the suffix-derived content-type and the FE `EXT_TO_MIME`/`PREVIEWABLE_EXTENSIONS` routing (`chat-utils.ts`) keep working (#4, fe seam).
7. **Validate before accepting** (`_validate`): re-open the rebuilt bytes with `python-docx` (`Document(BytesIO(out))`) for DOCX or `openpyxl` (`load_workbook(BytesIO(out), read_only=True)`) for XLSX. **Corruption guard #3:** any exception ⇒ reject the rebuild, return original, `fallback_reason="validate_failed"`. This is the load-bearing safety net — a structurally-valid-but-app-broken file is caught here, not by the user.
8. **Size + saving gate, then emit.** Accept the rebuild **only if** `output_size < original_size` **and** `saved_pct >= container_rebuild_min_saving_pct` (#9, stricter). Otherwise return original with `fallback_reason="no_saving"`. Emit `compression.rebuild.decision` (accepted) or `compression.rebuild.fallback` (any fallback branch) via `compression_metrics.record_rebuild(...)`.

The whole routine runs inside `asyncio.to_thread(...)` from the orchestrator (CPU-bound zip+Pillow work off the event loop — be-upload decision #6: synchronous inline, "parallel" = `asyncio.gather`+`asyncio.to_thread`) under an `asyncio.wait_for(..., timeout=settings.container_rebuild_timeout_seconds)`. **Corruption guard #4 (DoS / pathological archive):** on `TimeoutError`, fall back to original, `fallback_reason="timeout"` — never block OCR or the request.

### 7.7 Native-PDF branch (`_native_pdf.py`)

Only entered when `is_native_pdf(...) is True` (so a text-layer PDF is never rasterized — preserving OCR's native classification and the $0 native text path). Tier A is the default-shippable branch:

- **Tier A (PyMuPDF, no new dep):** `doc = fitz.open(stream=content, filetype="pdf")`; assert it opens and `doc.page_count >= 1` (else fallback). Save to a `BytesIO` with `doc.save(out, garbage=4, deflate=True, deflate_images=True, deflate_fonts=True, clean=True)`. This compacts the xref, deflates uncompressed streams, and drops orphaned objects — **lossless**, text layer intact. Validate by re-opening the output with `fitz` and asserting `out_doc.page_count == doc.page_count` and every page `.get_text(...)` returns without raising (corruption guard). Accept only if smaller by `min_saving_pct`.
- **Tier B (pikepdf, double-gated):** entered only if `container_rebuild_pikepdf_tier_b_enabled` AND `import pikepdf` succeeds. `pikepdf.open(BytesIO(content))` → `pdf.save(out, object_stream_mode=ObjectStreamMode.generate, compress_streams=True, recompress_flate=True)`, optional font subsetting. Same page-parity + open validation. On any `ImportError`/exception, **degrade to Tier A** (or to original), never raise. Adding `pikepdf>=8` to `requirements.txt` is a separate reviewed change (global-tech-stack) and is *not* required to ship the flag OFF or to ship Tier A.

> Encrypted/password-protected PDFs, PDFs with digital signatures (rebuild invalidates the signature), and AcroForm/XFA forms are **never rebuilt** — detect via `doc.is_encrypted` / `doc.needs_pass` / signature flags and fall back to original. Rewriting a signed PDF would silently break its signature; that is a hard skip.

### 7.8 How existing behavior stays unbroken

- **Flag OFF (default) = byte-identical pass-through.** The orchestrator never imports the engine; native-PDF/DOCX/XLSX bytes flow straight into `_save_original_file` exactly as today. No behavior change, no new failure mode, no perf cost.
- **Single write preserved.** The engine returns `bytes`; the caller substitutes them into the **one** `upload_bytes` write at `upload.py:258` (#3). No second write, so the dup-filename suffixer (`upload.py:249-254`) does not fire and the cancel-purge path (`_purge_upload_artifacts`, be-upload integration) still matches `{stem}.{ext}` unchanged.
- **OCR untouched.** OCR reads `temp_path` (the original, line 1044); the engine only ever touches an in-memory copy of `content` (#1/#2). Native PDFs are never rasterized (Tier A/B are lossless) so the native classifier verdict and the $0 native text extraction are unchanged.
- **Preview untouched.** Same filename + extension out (#4); `serve_original_file` content-type, the chat `buildPreviewUrl`/`EXT_TO_MIME` routing, and the ERP `fetchFileBlob` proxy all keep working.
- **Fallback never raises.** Every corruption guard, validate failure, no-saving, timeout, and unexpected exception returns the **original** bytes with a `fallback_reason`, mirroring `_save_original_file`'s non-blocking contract (#9). A rebuild failure can never block OCR or fail the upload.
- **Coverage + CI.** Engine lives in the non-omitted `compression/` package so its lines count toward the 75% gate; no migration ⇒ Alembic single-head gate untouched; no endpoint/router change.

### 7.9 Tests (where they go — per tests-ci)

Pure-logic unit tests in **`tests/unit/file_processing/test_container_rebuild.py`** (DB-free lane — the strict `tests/unit/` conftest warns on `db_session`/`client`): build a tiny `.docx`/`.xlsx` in-memory with `python-docx`/`openpyxl` embedding a large PNG, assert rebuilt is smaller AND re-opens AND keeps the same extension; assert a CRC-corrupted zip, a `..`-traversal member, an oversized decompressed member, and a Pillow-unreadable media member each fall back to original with the right `fallback_reason`; assert flag-OFF returns the input object unchanged (`rebuilt is False`). Native-PDF: build a multi-page PDF with `fitz`, assert Tier A page-parity + lossless text, assert encrypted/signed PDFs skip, assert `pikepdf`-absent degrades to Tier A. A `monkeypatch` that makes the rebuild raise must produce a fallback, not a 500 (mirrors `tests/unit/compliance/brs/test_brs_gcs_upload.py:148-160`). One `tests/api/` test (mirroring `tests/api/test_upload_cancel.py`) with the flag toggled on via `monkeypatch.setattr(settings, ...)` asserting the persisted bytes (`get_storage_client().download_bytes(...)`) are the rebuilt-or-original payload and OCR still ran on the original.

### 7.x Checklist

- [ ] Add `container_rebuild_*` flags (master + 3 sub-flags + 5 knobs) to `app/config.py` `Settings`, subsystem-named (no `feature_*` prefix), with `Field(ge=, le=)` bounds; default master = `False`.
- [ ] Create package `app/services/file_processing/compression/__init__.py` re-exporting `rebuild_container`, `RebuildResult`.
- [ ] Create `compression/container_rebuild.py` with `rebuild_container(content, filename, content_type) -> RebuildResult` + `RebuildResult` dataclass + extension dispatch (≤500 lines).
- [ ] Create `compression/_ooxml.py`: zip walk, media-member filter (`word/media/`, `xl/media/`, `ppt/media/`), per-member Pillow recompress on `BytesIO` with RGBA/P→RGB (mirror `image_handler.py:159-161`), per-member keep-smaller fallback.
- [ ] Create `compression/_native_pdf.py`: Tier A `fitz` lossless `doc.save(garbage=4, deflate=…, clean=True)`; import-guarded Tier B `pikepdf`; encrypted/signed/needs-pass hard-skip.
- [ ] Create `compression/_validate.py`: `validate_docx` (python-docx), `validate_xlsx` (openpyxl read_only), `validate_pdf` (fitz page-parity + per-page open).
- [ ] Create `compression/compression_metrics.py`: `record_rebuild(...)` emitting `compression.rebuild.decision` / `compression.rebuild.fallback`, event-name-first, all data in `extra={}`, no f-strings, no request-scoped ids.
- [ ] Implement guard #1: `BadZipFile`/`OSError` on open + `testzip()` CRC check → fallback `not_a_zip`.
- [ ] Implement guard #2: reject `..`/absolute member names, per-member and whole-archive decompressed-size ceilings (zip-bomb defense) → fallback.
- [ ] Implement guard #3: post-rebuild open-validate every container; any exception → reject rebuild, return original `validate_failed`.
- [ ] Implement guard #4: run rebuild in `asyncio.to_thread` under `asyncio.wait_for(timeout=container_rebuild_timeout_seconds)`; `TimeoutError` → fallback `timeout`.
- [ ] Enforce repack discipline: same member names + order, copy `[Content_Types].xml` / `_rels/` / all XML byte-for-byte, only swap recompressed media, NEVER `.zip` extension.
- [ ] Enforce size gate: accept only if `output_size < original_size` AND `saved_pct >= container_rebuild_min_saving_pct`; else return original (#9).
- [ ] Wire the single guarded branch in `upload.py` compression orchestrator (after read@864 + page-validation@~1019, before `_save_original_file`@1024-1025): `if settings.container_rebuild_enabled and <sub-flag>` → `rebuild_container`, else pass `content` through; substitute result into the single `upload_bytes` write @258 (no 2nd write).
- [ ] Confirm OCR still reads untouched `temp_path`@1044 and the engine only touches an in-memory copy of `content` (never `temp_path`).
- [ ] Confirm flag-OFF path does not import the engine and is byte-identical pass-through for native-PDF/DOCX/XLSX.
- [ ] (Optional, default OFF) add the same `container_rebuild_enabled` guard at `_attachments.py` `compress` hook, honoring `compress=False` for the provisioning logo push (`_steps_company.py:335`); leave OFF on ERPHome by default.
- [ ] Tier B: add `pikepdf>=8` to `requirements.txt` ONLY as a separately-reviewed change; engine must degrade to Tier A when `pikepdf` import fails.
- [ ] Place engine in the non-omitted `compression/` package (NOT inside `upload.py` or the OCR handler/`omit` set) so lines count toward the 75% coverage gate.
- [ ] Unit tests `tests/unit/file_processing/test_container_rebuild.py`: smaller-and-valid DOCX/XLSX rebuild, same-extension assertion, each corruption guard → correct `fallback_reason`, flag-OFF no-op, native-PDF Tier A page-parity + lossless text, encrypted/signed skip, pikepdf-absent degrade, raise→fallback (no 500).
- [ ] API test `tests/api/` (mirror `test_upload_cancel.py`) with flag monkeypatched on: persisted bytes = rebuilt-or-original via `get_storage_client().download_bytes(...)`; OCR ran on original.
- [ ] Verify no migration / no router / no endpoint added → Alembic single-head gate and route-registration order untouched.
- [ ] Run `ruff format --check`, `ruff check`, `mypy app/ --ignore-missing-imports`, and the new unit lane before declaring done.

---

All anchors confirmed against the freshly-pulled code. Now writing the section.

## 8. Genie OCR-Flow Integration (upload.py)

This section wires the compression module into the **single** Genie Chat OCR upload handler so that **OCR reads the untouched original** while **GCS/Local storage persists a compressed copy**, with a strict fall-back to the original on any failure or size regression. It honors corrected decisions 1–6 and 9, and changes no observable behavior when `session_id` is absent.

### 8.1 The exact integration point

There is exactly **one** persist sink for the Genie original and exactly **one** OCR call, both inside `upload_file()` (`app/api/v1/endpoints/upload.py:upload_file()`, def line 650, `try` opens 856, `finally` at 1353):

- **In-memory read** — `content = await file.read()` at `upload.py:864`, inside the `with NamedTemporaryFile(...)` block (858–949). `content: bytes` is pre-initialized to `b""` at 845 so early-validation `except` handlers can reference `len(content)`.
- **OCR input** — `temp_path` (`upload.py:858-859`), written from `content` at 948–949; consumed by `extraction = await asyncio.wait_for(file_router.process(temp_path, languages=languages), timeout=_OCR_PROCESSING_TIMEOUT)` at `upload.py:1044-1047`.
- **Persist seam** — `if session_id: await _save_original_file(session_id, original_filename, content)` at `upload.py:1024-1025`, which performs the **single** `storage.upload_bytes(...)` write at `upload.py:258` inside `_save_original_file()` (225–264).
- **Temp cleanup** — `finally:` at `upload.py:1353-1363` unlinks `temp_path` via `asyncio.to_thread(temp_path.unlink)`.

The persist call at 1024 already runs **after** the read (864) and **after** all size/page validation (cumulative-size 869–916, per-file size 919–946, PDF page-count 954–1019), and **before** both OCR (1044) and the temp unlink (1359). Deferral is therefore **already satisfied** — no statement reordering is required. The wiring is: compute compressed bytes after validation (after 1019, before 1024), then feed them into the persist call. The OCR call at 1044 keeps reading `temp_path` (the original) untouched, satisfying decisions 1–2.

Why not compress `temp_path`: `ImageHandler._resize_if_needed` (`image_handler.py:139-167`, `img.save(file_path)` at 162) **overwrites the temp file in place** for images >4096px before OCR; PDF page-count validation (957–963) and OCR both read `temp_path`. Compression therefore acts only on a **copy of in-memory `content`**, never on `temp_path` (decision 1).

Why overwrite the existing single write (not add a second): `_save_original_file` has a dup-filename suffixer at `upload.py:249-254` that renames on collision to `{stem}_{counter}{suffix}`. A second `upload_bytes` would trip the suffixer and create a phantom `{filename}_1` (and, in prod, the bucket IAM is `objectCreator`-only — overwriting an existing path 403s). We mutate the bytes passed into the **one** write at 258 (decision 3). The compressed file keeps the **same filename + extension** because both the serve content-type (`serve_original_file` MIME_TYPE_MAP, suffix-derived, `upload.py:1927-2004`) and the cancel-purge matcher (`_purge_upload_artifacts`, matches `{stem}.{ext}`/`{stem}_{counter}.{ext}`) key off the suffix/stem (decision 4).

### 8.2 Files to CREATE

- **`/Users/bhags/Desktop/erpsense_all/erpsense-backend/app/services/file_processing/compression.py`** — the compressor (≤500 lines, NOT in `pyproject.toml [tool.coverage.run].omit`, so it counts toward the 75% gate; `upload.py` itself is omitted). Pure, synchronous, blocking-CPU functions designed to run under `asyncio.to_thread`. Public surface:

  ```python
  # app/services/file_processing/compression.py
  import io, logging
  from dataclasses import dataclass
  logger = logging.getLogger(__name__)  # module-level, no f-strings in messages

  @dataclass(frozen=True)
  class CompressionResult:
      data: bytes              # compressed-or-original (caller persists this verbatim)
      original_size: int
      stored_size: int
      strategy: str            # "image" | "scanned_pdf" | "none"
      decision: str            # "compressed" | "fallback_original" | "skipped"
      fallback_reason: str | None

  def compress_content(
      content: bytes, *, filename: str, content_type: str | None,
  ) -> CompressionResult:
      """Compress a COPY of in-memory bytes. NEVER raises. NEVER touches temp_path.
      Returns original bytes (decision==fallback_original/skipped) on any failure,
      on compressed>=original, or when below the 500KB exemption."""
  ```

  Internals (all on `io.BytesIO(content)` copies):
  - **Classification** routes via suffix/`content_type` and reuses the shared native/scanned helper — introduce `app/services/file_processing/handlers/_pdf_classifier.py:is_native_pdf(text, page_count, *, mode)` and delegate from BOTH `PDFHandler._has_sufficient_text` (`pdf_handler.py:302-308`, TOTAL `len(text) >= 50*page_count`) and `LibraryGeminiProcessor._extract_pdf_native` (`library_gemini.py:589-618`, PER-PAGE AVG `< 50`, raises) — parameterized so neither call site changes semantics or raise-vs-branch control flow (be-ocr gotchas).
  - **Images** (`.png/.jpg/.jpeg/.gif/.bmp/.tiff/.webp`): Pillow, target ≤1200px longest edge + recompress to ≤200KB; replicate `ImageHandler._resize_if_needed`'s RGBA/P→RGB conversion (`image_handler.py:159-161`) for JPEG compatibility; preserve original format/extension.
  - **Scanned PDFs** only (native PDF/DOCX/XLSX/CSV pass through, decision 7): PyMuPDF (`fitz`) rasterize 200 DPI (mirror `PDFHandler._pdf_to_images`, `pdf_handler.py:310-326`), Pillow-compress each page to ≤300KB/page, rebuild with PyMuPDF/reportlab (pikepdf/img2pdf are NOT in requirements — be-ocr).
  - **Exemptions/guards**: `len(content) < 500*1024` → `decision="skipped"` (decision 8: min 500KB exempt). Native containers → `decision="skipped"`. The optional container-rebuild path is behind a config flag, off by default (decision 7).

- **`/Users/bhags/Desktop/erpsense_all/erpsense-backend/app/services/file_processing/handlers/_pdf_classifier.py`** — the shared `is_native_pdf()` helper described above (be-ocr reuse plan; non-omitted module).

- **`/Users/bhags/Desktop/erpsense_all/erpsense-backend/app/services/file_processing/compression_metrics.py`** — log-based telemetry, mirroring `app/services/tally_sync/diagnostic_metrics.py`. Thin `logger.info("<event>", extra={...})` wrappers (`compression.decision`, `compression.fallback`); optional registry `_init_compression_metrics()` is a separate concern handled in the logging section. No new contextvar (`session_id`/`tenant_id`/`request_id` are auto-injected by `JSONFormatter`).

### 8.3 Files to MODIFY

- **`/Users/bhags/Desktop/erpsense_all/erpsense-backend/app/api/v1/endpoints/upload.py`**
  1. **`_save_original_file()` (225–264)** — change signature to accept the already-compressed bytes plus telemetry, keeping the **single** write at 258 and the suffixer at 249–254 intact:
     ```python
     async def _save_original_file(
         session_id: str, original_filename: str, content: bytes,
         *, original_size: int | None = None, strategy: str = "none",
         decision: str = "stored",
     ) -> str | None:
     ```
     The body keeps `safe_name` sanitization (241–244), the suffixer, `content_type = mimetypes.guess_type(safe_name)[0] or "application/octet-stream"` (257), and `await storage.upload_bytes(storage_path, content, content_type=content_type)` (258). Replace the f-string log at line 259 with a compliant structured event (be-logging):
     ```python
     logger.info("upload.original_saved", extra={
         "filename": safe_name, "storage_path": storage_path,
         "stored_size_bytes": len(content),
         "original_size_bytes": original_size if original_size is not None else len(content),
         "strategy": strategy, "decision": decision,
         "storage_backend": type(storage).__name__,
     })
     ```
     Keep the non-blocking contract: `except Exception` (262) still returns `None` and never raises (decision 9). Do **not** extend the `"="*70` banner logs (1027–1041) — CLAUDE.md violation.

  2. **The persist seam (1024–1025)** — replace with compression-then-persist, run in parallel with OCR per decision 6 (synchronous inline, no worker → `asyncio.gather` + `asyncio.to_thread`).

- **`/Users/bhags/Desktop/erpsense_all/erpsense-backend/app/config.py`** — add `Settings` fields for compression knobs (decisions 7/8; NEVER `os.getenv`, be-logging): `compression_enabled: bool = True`, `compression_min_exempt_bytes: int = 500*1024`, `compression_image_max_px: int = 1200`, `compression_image_target_bytes: int = 200*1024`, `compression_pdf_page_target_bytes: int = 300*1024`, `compression_pdf_dpi: int = 200`, `compression_rebuild_containers: bool = False` (off-by-default guarded container path).

### 8.4 Before / after pseudocode (anchored)

**BEFORE** (`upload.py`, ~954–1047, abbreviated):

```python
        # PDF page-count validation ...                       # 954–1019  (unchanged)

        # SAVE ORIGINAL FILE (for preview from chat history)  # 1021–1025
        if session_id:
            await _save_original_file(session_id, original_filename, content)

        logger.info("=" * 70); ...                            # 1027–1041 (banner; leave as-is)

        extraction = await asyncio.wait_for(                  # 1044–1047
            file_router.process(temp_path, languages=languages),
            timeout=_OCR_PROCESSING_TIMEOUT,
        )
```

**AFTER** (substitute compressed bytes; run compression concurrently with OCR; OCR still reads `temp_path`):

```python
        # PDF page-count validation ... (unchanged, still reads temp_path)   # 954–1019

        # ---- Compress a COPY of in-memory `content` IN PARALLEL with OCR ----
        # temp_path (original) is the OCR input and is NEVER mutated here.
        # compression runs on bytes only; OCR reads temp_path concurrently.
        async def _run_ocr():
            return await asyncio.wait_for(
                file_router.process(temp_path, languages=languages),   # original, untouched
                timeout=_OCR_PROCESSING_TIMEOUT,
            )

        async def _run_compression() -> CompressionResult | None:
            if not (session_id and settings.compression_enabled):
                return None                              # no-op when session_id absent
            try:
                return await asyncio.to_thread(
                    compress_content,
                    content,                             # COPY-of-bytes contract inside compressor
                    filename=original_filename,
                    content_type=file.content_type,
                )
            except Exception as exc:                     # compressor shouldn't raise, but belt-and-suspenders
                logger.warning("compression.fallback", extra={
                    "filename": original_filename,
                    "original_size_bytes": len(content),
                    "fallback_reason": type(exc).__name__,
                })
                return None                              # → store original below

        # gather: OCR failure must still surface as the existing 422/timeout path;
        # compression failure must NEVER block OCR (decision 9).
        ocr_result, comp_result = await asyncio.gather(
            _run_ocr(), _run_compression(), return_exceptions=True,
        )
        if isinstance(ocr_result, BaseException):
            raise ocr_result                             # preserve existing TimeoutError/FileProcessing handling
        extraction = ocr_result

        # ---- Choose compressed-or-original, then the SINGLE persist write ----
        if session_id:
            if isinstance(comp_result, CompressionResult) and comp_result.decision == "compressed":
                await _save_original_file(
                    session_id, original_filename, comp_result.data,
                    original_size=comp_result.original_size,
                    strategy=comp_result.strategy, decision="compressed",
                )
            else:
                # fallback: skipped / fallback_original / None / exception → store ORIGINAL, logged, no retry
                await _save_original_file(
                    session_id, original_filename, content,
                    original_size=len(content), strategy="none", decision="fallback_original",
                )

        logger.info("=" * 70); ...                       # existing banner block kept verbatim
        # (OCR already done above; remove the standalone wait_for at 1044–1047)
```

Notes tying the pseudocode to invariants:
- `asyncio.gather(..., return_exceptions=True)` ensures a compression exception cannot abort OCR; the OCR branch is re-raised to keep the **exact** existing failure semantics (timeout → 422 path at 1349–1352; `file_router` exceptions handled by the existing `except` ladder). This satisfies decision 9 and the "compression never blocks OCR" gotcha.
- Persist still happens **after** validation and OCR-scheduling and **before** the `finally` unlink (1359), so the deferral guarantee and `temp_path` lifetime are unchanged.
- When `session_id` is `None`, `_run_compression` returns `None` and **no** persist occurs — byte-for-byte identical to today's legacy path (no behavior change without a session).
- `comp_result.decision == "compressed"` is the **only** path that persists compressed bytes; every other outcome (`skipped` for <500KB or native containers, `fallback_original` for compressed≥original or error, `None` for disabled/no-session) persists the **original** — single write either way (decision 3), same filename/extension (decision 4).

### 8.5 How existing behavior stays unbroken

- **OCR fidelity**: `file_router.process(temp_path, ...)` (`router.py:109`) reads the original `temp_path`; compression touches only `content` copies — the `ImageHandler` in-place-overwrite footgun (`image_handler.py:162`) is avoided, and native/scanned routing under the active `library_gemini` approach (`library_gemini.py`, `ocr_features.json` priority 1) is unaffected because we reuse, not change, the classifier.
- **Preview**: same filename+extension → `serve_original_file` MIME_TYPE_MAP suffix lookup (`upload.py:1927-2004`) and FE `buildPreviewUrl`/`EXT_TO_MIME` (be/fe seams) keep resolving content-type; the byte-proxy `get_serve_response` (`storage_client.py:378-396`) serves the compressed bytes transparently.
- **Cancel/purge**: `_purge_upload_artifacts` / `cancel_upload` match `{stem}.{ext}`/`{stem}_{counter}.{ext}` — unchanged because filename is preserved.
- **Limits**: no change to `MAX_FILE_SIZE=30MB` (81) etc.; validation still short-circuits before any save. The 500KB-min exemption lives inside the compressor only.
- **Storage selection**: unchanged — `get_storage_client()` (`storage_client.py:399-418`) returns GCS in DEV/prod when `gcs_bucket_name`/`gcs_upload_bucket` is set (decision 10), Local otherwise; compressed bytes flow through the same `upload_bytes(path, data, content_type)` contract.
- **Logging compliance**: the only edit to existing logging replaces the f-string at 259 with a structured event; the pre-existing banner lines (1027–1041) and `storage_client.py` `%`-style DEBUG logs are left as-is (not extended), per the "no double logging / no new f-strings" rules.
- **ERPHome surface untouched**: this seam is Genie-only (`upload.py`); the ERPNext adapter path (`_attachments.py`) is a separate feature (decision 5) and is not modified here.

### 8.6 Checklist

- [ ] Create `app/services/file_processing/compression.py` with `compress_content(content, *, filename, content_type) -> CompressionResult` that operates only on `io.BytesIO` copies and **never raises**.
- [ ] Define `CompressionResult` dataclass (`data`, `original_size`, `stored_size`, `strategy`, `decision`, `fallback_reason`).
- [ ] Implement image path: Pillow resize ≤1200px + recompress ≤200KB, RGBA/P→RGB conversion (mirror `image_handler.py:159-161`), preserve original format/extension.
- [ ] Implement scanned-PDF path: `fitz` rasterize 200 DPI (mirror `pdf_handler.py:310-326`), per-page Pillow compress ≤300KB/page, rebuild via PyMuPDF/reportlab (no pikepdf/img2pdf).
- [ ] Enforce 500KB min-exemption → `decision="skipped"`; native PDF/DOCX/XLSX/CSV pass-through → `decision="skipped"`.
- [ ] On error OR compressed≥original → return original bytes with `decision="fallback_original"` + `fallback_reason`; no retry.
- [ ] Create `app/services/file_processing/handlers/_pdf_classifier.py:is_native_pdf(text, page_count, *, mode)` parameterized for TOTAL vs PER-PAGE semantics.
- [ ] Delegate `PDFHandler._has_sufficient_text` (`pdf_handler.py:302-308`) to the helper, preserving TOTAL semantics and branch control-flow.
- [ ] Delegate `LibraryGeminiProcessor._extract_pdf_native` (`library_gemini.py:589-618`) to the helper, preserving PER-PAGE AVG semantics and the `FileProcessingException` raise.
- [ ] Create `app/services/file_processing/compression_metrics.py` mirroring `diagnostic_metrics.py` (events `compression.decision`, `compression.fallback`; data in `extra`, no f-strings, no manual contextvars).
- [ ] Add compression `Settings` fields in `app/config.py` (`compression_enabled`, `compression_min_exempt_bytes`, `compression_image_max_px`, `compression_image_target_bytes`, `compression_pdf_page_target_bytes`, `compression_pdf_dpi`, `compression_rebuild_containers=False`); no `os.getenv`.
- [ ] In `upload.py`, import `compress_content`/`CompressionResult`/`settings` and add `_run_ocr()` + `_run_compression()` locals inside `upload_file()`.
- [ ] Replace the standalone OCR `await asyncio.wait_for(...)` (1044–1047) with `asyncio.gather(_run_ocr(), _run_compression(), return_exceptions=True)` and re-raise the OCR exception to preserve existing 422/timeout handling.
- [ ] Ensure `_run_compression()` returns `None` (no-op) when `session_id is None` or `settings.compression_enabled is False`.
- [ ] Replace persist call (1024–1025) with compressed-or-original selection: persist `comp_result.data` only when `decision=="compressed"`, else persist original `content`.
- [ ] Update `_save_original_file()` signature (225–227) to accept `*, original_size`, `strategy`, `decision`; keep the single `upload_bytes` write at 258 and the dup-suffixer (249–254) intact; do NOT add a second write.
- [ ] Replace the f-string log at `upload.py:259` with `logger.info("upload.original_saved", extra={...})` (filename, storage_path, stored/original size, strategy, decision, storage_backend); keep the `except` non-blocking (returns `None`, never raises).
- [ ] Verify `temp_path` is never passed to the compressor and is unlinked only in the existing `finally` (1359–1361); confirm OCR still reads `temp_path`.
- [ ] Confirm compressed output keeps the same `safe_name`+suffix so `serve_original_file` MIME map (1927–2004) and `_purge_upload_artifacts` matcher keep working.
- [ ] Confirm no edits to the banner logs (1027–1041), to `storage_client.py`, or to the ERPHome adapter (`_attachments.py`) from this seam.
- [ ] Confirm legacy (`session_id is None`) path is byte-for-byte unchanged (no compression, no persist).

---

I have all the confirmation I need. Key findings to fold into the plan:
- Branch `feat/auto-compression`, no compression module exists yet (clean slate).
- `_attachments.py` and the employee services are NOT in the coverage omit list, so adapter-level code IS coverage-measured. The OCR handlers (`pdf_handler.py`, `image_handler.py`) and `upload.py` ARE omitted.
- All libs (PyMuPDF, Pillow, reportlab) present; pikepdf/img2pdf absent.
- Confirmed both ERPHome sinks, the provisioning logo opt-out call, the `_MAX_ATTACHMENT_SIZE_BYTES = 10MB` shared cap, and the 5 in-scope callers.

Now writing the section.

## 9. ERPHome Attachment + OCR-Attach Entity Integration

This section wires **pre-POST byte compression** into the ERPHome attachment surfaces so that the bytes ERPNext persists in the Frappe `File` doctype are already optimized, **without OCR** (OCR belongs solely to the Genie chat surface), **without a second GCS sink** (ERPHome storage is per-surface = ERPNext only, per corrected decision 5), and **without breaking** any of the five in-scope callers, the provisioning logo push, or the three out-of-scope sinks (onboarding import, branding logo, FileStorageService). The "OCR-attach entity" feature — attaching files to an ERP entity and (optionally, on the Genie side) running OCR — is wired end-to-end by reusing the existing Genie OCR persist seam plus a shared compressor.

### 9.1 Architecture: one compressor, two call families

There is exactly **one** new compression engine, consumed by **two** distinct surfaces that never share a storage sink:

| Surface | Entry | Sink | OCR? | Compression hook |
|---|---|---|---|---|
| ERPHome attach-to-doc | `doc_social.py:upload_attachment` (`POST /erp/{doctype}/{name}/attachment`) | ERPNext `File` doctype | No | adapter-level |
| ERPHome standalone inline-field | `file_proxy.py:upload_erp_file` (`POST /erp/files/upload`) | ERPNext `File` doctype | No | adapter-level |
| ERPHome expense receipts (x2) | `employee_expense_service.py:upload_attachments`/`attach_line_receipt` | ERPNext `File` (Expense Claim) | No | adapter-level (inherited) |
| ERPHome travel receipts | `employee_travel_service.py:upload_attachments` | ERPNext `File` (Travel Request) | No | adapter-level (inherited) |
| Genie chat OCR | `upload.py:upload_file` (`POST /api/v1/upload`) | GCS / Local via `_save_original_file` | Yes | already covered in plan §(compression) at `upload.py:1024` |

The **single recommended ERPHome hook** is at the adapter mixin (`AttachmentsMixin.upload_attachment` and `AttachmentsMixin.upload_file_standalone` in `app/adapters/erpnext/_attachments.py`), because it covers all **five** ERPHome callers in **one** place with no per-caller edits. The only adapter caller that must opt out is the provisioning branding-logo push at `app/services/erpnext_provisioning/_steps_company.py:335` (`is_private=False`, image already optimized by `BrandingService`).

### 9.2 Files to CREATE

**`app/services/file_processing/compression.py`** (NEW, non-omitted → coverage-counted). The shared, pure-logic compressor. Must stay ≤500 lines (modularity BLOCKER). It owns NO storage/network/temp-file I/O — it takes bytes + a content hint and returns bytes. This is the *same* engine the Genie side calls (single source of truth for "compress only images + scanned PDFs"). Confirmed libs available: `PyMuPDF (fitz)`, `Pillow`, `reportlab`; `pikepdf`/`img2pdf` are NOT present, so any PDF rebuild uses fitz+Pillow only.

```python
# app/services/file_processing/compression.py
import io, logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class CompressionResult:
    data: bytes              # compressed OR original (fallback)
    original_size: int
    compressed_size: int
    strategy: str            # "image" | "scanned_pdf" | "skip_native" | "skip_small" | "skip_unsupported"
    decision: str            # "compressed" | "fallback_original" | "skipped"
    fallback_reason: str | None  # "error" | "regression" | None

    @property
    def saved_bytes(self) -> int: return max(0, self.original_size - self.compressed_size)
    @property
    def saved_pct(self) -> float:
        return round(100 * self.saved_bytes / self.original_size, 2) if self.original_size else 0.0

def compress_bytes(
    content: bytes,
    *,
    filename: str,
    content_type: str | None,
) -> CompressionResult:
    """Compress a COPY of `content`. NEVER mutate the caller's bytes.
    Returns ORIGINAL bytes on any failure or size-regression (decision 9).
    Preserves filename+extension implicitly (caller keeps same name)."""
```

Internal helpers (private, in the same module):
- `_classify(filename, content_type) -> "image" | "scanned_pdf" | "native" | "unsupported"` — uses suffix + content_type, and for `.pdf` reads page text via `fitz` from a `BytesIO` copy and calls the shared classifier `is_native_pdf()` (see §9.3). Native PDF / DOCX / XLSX / CSV / TXT → not compressed (spec scope 7).
- `_compress_image(content) -> bytes` — replicates `ImageHandler._resize_if_needed`'s RGBA/P→RGB conversion (`image_handler.py:159-161`) **on a `BytesIO` copy**, downscales to `MAX_DIM` (default 1200px longest edge, decision 7), re-encodes JPEG/WebP/PNG by original format with quality from config; targets ≤200KB. Keeps the **same extension/format family** so the suffix-derived content-type still matches (decisions 4 & 5).
- `_compress_scanned_pdf(content) -> bytes` — rasterizes each page at config DPI via `fitz` (mirror `PDFHandler._pdf_to_images` 200-DPI pattern, `pdf_handler.py:310-326`) into compressed JPEGs, rebuilds a PDF (fitz `insert_image` or reportlab canvas), targets ≤300KB/page (decision 7). On any page failure → raise internally → caught → original fallback.
- Guards: **MIN exempt** — if `len(content) < settings.compression_min_bytes` (default 500KB, decision 8) → `decision="skipped"`, `strategy="skip_small"`. **Regression guard** — if `compressed >= original` → return original, `decision="fallback_original"`, `fallback_reason="regression"` (decision 9). **Never raises** to the caller.

**`app/services/file_processing/_pdf_classifier.py`** (NEW, non-omitted → coverage-counted). The shared native-vs-scanned helper extracted so the compression decision stays in lock-step with what OCR considers native/scanned (`be-ocr` integration point). It exposes:

```python
def avg_chars_per_page(text: str, page_count: int) -> float: ...
def is_native_pdf(text: str, page_count: int, *, mode: str = "total") -> bool:
    """mode="total"  -> len(text) >= MIN_TEXT_LENGTH * page_count   (PDFHandler semantics)
       mode="average"-> avg_chars_per_page(text, page_count) >= MIN  (library_gemini semantics)"""
```
This parameterizes on `mode` so neither existing classifier's behavior changes (the two have **different semantics** — `PDFHandler._has_sufficient_text` at `pdf_handler.py:302-308` uses TOTAL `len(text) >= 50*page_count`; `LibraryGeminiProcessor._extract_pdf_native` at `library_gemini.py:589-618` uses PER-PAGE AVERAGE and *raises* `FileProcessingException`). Threshold constant stays `MIN_TEXT_LENGTH=50`. Compression's `_classify` calls `is_native_pdf(text, pages, mode="average")` for its native-skip decision. **Note:** `pdf_handler.py` and `library_gemini.py` are in the coverage `omit` list (pyproject.toml:127,140), so delegating to this helper from them does not regress coverage; the helper itself is non-omitted and unit-tested directly.

**`app/services/file_processing/compression_metrics.py`** (NEW, non-omitted). Log-based metrics module mirroring `app/services/tally_sync/diagnostic_metrics.py` (the house-current, forward-looking idiom). Thin `logger.info(event, extra={...})` wrappers — Cloud Monitoring derives counters/distributions from the event name + extra. No registry wiring, no test-reset needed.
```python
def record_compression(surface, result: CompressionResult, *, filename, content_type, duration_ms): ...
    # logger.info("compression.decision", extra={surface, filename, content_type,
    #   original_size_bytes, compressed_size_bytes, saved_pct, strategy, decision,
    #   fallback_reason, duration_ms})  -- NEVER request_id/tenant_id (auto-injected)
```
`surface` label ∈ `{"erphome_attachment","erphome_standalone","genie_ocr"}`. **No raw bytes, no PII** — only filenames/sizes/strategy (logging seam). Knobs (`compression_min_bytes`, `compression_image_max_dim`, `compression_image_max_kb`, `compression_pdf_per_page_kb`, `compression_pdf_dpi`, `compression_pdf_rebuild_enabled` default `False`) live in `app/config.py` Settings — **never `os.getenv`** (R03).

**Optional (guarded, off by default):** `MetricsRegistry._init_compression_metrics()` in `app/core/metrics.py` for a Prometheus `/api/v1/metrics` scrape (`compression_decisions_total{surface,strategy,decision}`, `compression_fallback_total{reason}`, `compression_bytes_saved` histogram). If added, it MUST be called from both `__post_init__` AND `reset_metrics()` (else pytest leaks counter state). Low-cardinality labels only; never `tenant_id` as a label.

### 9.3 Files to MODIFY

**`app/adapters/erpnext/_attachments.py`** — `AttachmentsMixin.upload_attachment` (lines 77-168) and `AttachmentsMixin.upload_file_standalone` (lines 170-233):
1. Add `compress: bool = True` kwarg to **both** signatures.
2. Immediately after the empty-content guard (after line 106 / 194), before the multipart build (line 108 / 196), compress a COPY of the incoming bytes:
   ```python
   if compress:
       import time
       from app.services.file_processing.compression import compress_bytes
       from app.services.file_processing.compression_metrics import record_compression
       _t0 = time.monotonic()
       _res = compress_bytes(file_content, filename=file_name, content_type=content_type)
       record_compression("erphome_attachment", _res, filename=file_name,
                           content_type=content_type, duration_ms=round((time.monotonic()-_t0)*1000, 2))
       file_content = _res.data   # compressed OR original (fallback never raises)
   ```
   The `files = {"file": (file_name, file_content, content_type or ...)}` multipart tuple is unchanged below — same filename, same `content_type` 3rd element (preview is suffix/header-derived, decisions 4 & 5). Compression **must not** alter the `content_type` request-header handling (the `del`-Content-Type / multipart-boundary logic at lines 130-142 / 205-217 stays untouched).
3. The existing `logger.info("Attachment uploaded"/"Standalone file uploaded", extra={size:...})` (lines 147 / 222) keeps the `size` field as `len(file_content)` (now the *stored/compressed* size) — `record_compression` carries the before/after pair, so no duplicate logging.
4. A compression exception can never bubble (`compress_bytes` swallows + falls back), so each method's existing `try/except` returning the `{"success": False, ...}` envelope is preserved unchanged.

**`app/services/erpnext_provisioning/_steps_company.py`** — at the logo push (line 335), pass `compress=False` so the already-optimized branding logo is not re-touched (this is the single in-adapter call that is functionally out-of-scope):
```python
upload_resp = await self.admin_adapter.upload_file_standalone(
    file_name=derived_name, file_content=logo_bytes,
    is_private=False, content_type=content_type, compress=False,
)
```
The surrounding `try/except` (lines 334-350, non-fatal) is unchanged.

**`app/services/file_processing/handlers/pdf_handler.py`** — `PDFHandler._has_sufficient_text` (302-308): replace the body with `return is_native_pdf(text, page_count, mode="total")` (delegate, keep the method + caller at line 263). Behavior identical (TOTAL semantics preserved).

**`app/services/file_processing/ocr/library_gemini.py`** — `_extract_pdf_native` (589-618): replace the inline `avg < MIN` check with `is_native_pdf(text, page_count, mode="average")`, **keeping the `raise FileProcessingException` control-flow** (do not collapse raise→branch). Behavior identical (AVERAGE semantics + raise preserved).

**`app/api/v1/endpoints/upload.py`** (Genie side, OCR-attach entity) — at the `_save_original_file` call (1024-1025), substitute compressed bytes for `content` and call `record_compression("genie_ocr", ...)`. OCR at line 1044 still reads the **untouched** `temp_path` (OCR reads original; storage saves compressed). This is detailed in the Genie-compression plan section; cross-referenced here because it consumes the **same** `compress_bytes`. (`upload.py` is coverage-omitted, so its compression *logic* is exercised via the new module's unit tests + the `tests/api/test_upload_cancel.py`-style endpoint test.)

**`app/config.py`** — add the six `compression_*` Settings fields (defaults above; `compression_pdf_rebuild_enabled=False` per decision 7's guarded/off-by-default container rebuild).

**`app/core/exceptions/`** — no new exception class is required: `compress_bytes` is non-raising by contract (decision 9). If a future caller wants a hard-fail path, a `FILE_*`-prefixed class would go in the file-processing exceptions module — not needed now.

### 9.4 What is explicitly NOT touched (out-of-scope sinks)

These never call `AttachmentsMixin.upload_attachment`/`upload_file_standalone`, so the adapter hook structurally cannot intercept them — no guard code needed beyond the one provisioning opt-out:
- `app/api/v1/endpoints/import_files/upload.py:upload_file` → `FileProcessingService` → `storage.upload_bytes` (onboarding master-data import, content-hash keyed). **Skip.**
- `app/api/v1/endpoints/files.py:upload_logo` → `BrandingService.process_logo` (Pillow already resizes). **Skip.**
- `app/api/v1/endpoints/attachments.py` → `FileStorageService` → GCS + `erpsense_files` DB (production-floor/employee workflow). **Skip** — different sink; do NOT add compression here despite the seam note mentioning it, since adding a GCS-side compressor to this sink is a separate feature and conflates surfaces (decision 5 / per-surface storage).

### 9.5 Size-cap reconciliation (ERPHome ≠ Genie)

`_MAX_ATTACHMENT_SIZE_BYTES = 10MB` at `app/api/v1/endpoints/erp_documents/_shared.py:111` is the **genuine** ERPHome cap, enforced on **both** routes (`doc_social.py:298,321` and `file_proxy.py:153,173`). Decision 8's "MAX is really 30MB not 10MB" applies **only** to the Genie chat surface (`upload.py:MAX_FILE_SIZE=30MB`); the ERPHome adapter path legitimately stays **10MB**. Do **not** raise `_MAX_ATTACHMENT_SIZE_BYTES` to 30MB — that would change both routes' 413 messages and the FE ERPHome copy with no product mandate. Compression runs *after* these 413 checks, so it never sees an oversize body. The MIN-500KB-exempt rule is applied inside `compress_bytes` (no route change).

### 9.6 OCR-Attach Entity feature wiring (end to end)

"Attach a file to an ERP entity" is already wired through the two ERPHome routes (`doc_social` attach-to-doc; `file_proxy` standalone inline-field) and the FE (`attachment-upload.tsx` → `doc-side-panel.tsx` → `erpDocumentsApi.uploadAttachment`; `attach-field-input.tsx` → `erpDocumentsApi.uploadFile`). This feature's contribution to that flow:
- **Storage:** compressed bytes land in ERPNext's `File` doctype (no GCS) — the existing `fetch_file`/`fetch_attachment_bytes` proxy and the FE `components/erp/file-preview-modal.tsx` (content-type from proxy header) keep working unchanged because filename+extension are preserved.
- **OCR** stays on the Genie surface only. The ERPHome attach path does NOT import/trigger `file_router`/`library_gemini` (confirmed no OCR on either adapter method). If a future "OCR an ERP attachment" capability is wanted, it would be a *separate* gated feature reading the ERPNext file bytes and feeding `file_router.process` — out of scope here and intentionally not wired.
- **Cancel/preview parity (Genie):** the Genie cancel-purge path (`DELETE /upload/{upload_id}` → `_purge_upload_artifacts`) matches `{stem}.{ext}`/`{stem}_{counter}.{ext}`; because compression keeps the same filename, purge keeps working.

### 9.7 Tests to create (coverage-bearing)

Per `tests-ci` seams: the compressor + classifier live in **non-omitted** modules, so they count toward the 75% backend gate; the adapter (`_attachments.py`) is also non-omitted.
- **`tests/unit/file_processing/test_compression.py`** (pure logic, `@pytest.mark.unit` auto, no DB): image >500KB compresses & shrinks; image <500KB → `skip_small`; native PDF (dense text) → `skip_native`; scanned PDF (image-only) → compressed ≤300KB/page when `compression_pdf_rebuild_enabled=True`; DOCX/XLSX/CSV → skipped; compressed≥original → `fallback_original`/`regression`; corrupt bytes → `fallback_original`/`error` (never raises); **original bytes object is not mutated** (identity/`bytes` immutability + assert input unchanged).
- **`tests/unit/file_processing/test_pdf_classifier.py`**: `is_native_pdf` total-mode vs average-mode boundary cases at the 50-char threshold; mixed-density PDF differs between modes (locks in that the two semantics are preserved).
- **`tests/api/test_erp_attachment_compression.py`** (HTTP via `client`, `respx` mocking ERPNext `POST {BASE_URL}/api/method/upload_file`, seed `test_erp_connection` + `authenticated_user_with_tenant`): POST a >500KB JPEG to `/erp/{doctype}/{name}/attachment` and assert the bytes captured by the respx mock are **smaller than** the uploaded bytes (compression actually applied at the sink); POST a tiny file and assert bytes pass through unchanged; `compress=False` path (drive via the standalone route is `True`, so cover the opt-out by a direct adapter unit test). **Cross-tenant BOLA** test asserting `== 404` strictly (tenant-scoped BLOCKER).
- **`tests/unit/adapters/test_attachments_compress.py`**: call `upload_attachment`/`upload_file_standalone` with `respx`-mocked upstream and assert the multipart `file` part carries compressed bytes when `compress=True`, original bytes when `compress=False`; assert filename+`content_type` tuple unchanged; assert a compressor exception (monkeypatch `compress_bytes` to raise) still yields the normal success envelope with original bytes (fallback contract).
- **`tests/unit/file_processing/test_compression_metrics.py`**: `record_compression` emits the `compression.decision` event with the expected `extra` keys and never logs raw bytes/PII (caplog assertion).
- Mirror the existing fallback idiom from `tests/unit/compliance/brs/test_brs_gcs_upload.py` (monkeypatch to raise → fallback). Async convention: `async def test_*`, no `@pytest.mark.asyncio`.

### 9.8 Non-breakage guarantees

- All five ERPHome callers inherit compression with zero per-caller edits; their existing success/error envelopes, audit calls, and 413/422 guards are untouched.
- `compress: bool = True` default means any caller not updated keeps prior behavior except that bytes get optimized — and the only call that must NOT change (provisioning logo) is explicitly opted out.
- `compress_bytes` is **non-raising** + falls back to original on error/regression → an upload can never fail because of compression.
- Filename + extension + `content_type` preserved → ERPNext serve, FE `erp/file-preview-modal.tsx`, and Genie `chat-utils.buildPreviewUrl`/suffix-derived serve all keep working.
- The two OCR classifier refactors are behavior-preserving delegations (semantics + raise/branch control-flow unchanged); they live in coverage-omitted files so the gate is unaffected.
- No second storage sink added on the ERPHome path (per-surface storage honored).

### 9.x Checklist

- [ ] Create `app/services/file_processing/_pdf_classifier.py` with `avg_chars_per_page()` + `is_native_pdf(text, page_count, mode)` (modes `total`/`average`, `MIN_TEXT_LENGTH=50`).
- [ ] Create `app/services/file_processing/compression.py` with `CompressionResult` dataclass + non-raising `compress_bytes(content, *, filename, content_type)`; keep ≤500 lines.
- [ ] Implement `_classify()` using suffix + content_type + `is_native_pdf(mode="average")`; native PDF/DOCX/XLSX/CSV/TXT → skip.
- [ ] Implement `_compress_image()` on a `BytesIO` copy with RGBA/P→RGB (mirror `image_handler.py:159-161`), `MAX_DIM` downscale, format-preserving re-encode; never mutate input bytes.
- [ ] Implement `_compress_scanned_pdf()` via `fitz` rasterize (200-DPI pattern from `pdf_handler.py:310-326`) + fitz/reportlab rebuild, ≤300KB/page; gate behind `compression_pdf_rebuild_enabled` (default off).
- [ ] Add MIN-500KB exempt guard (`skip_small`) and compressed≥original regression guard (`fallback_original`/`regression`).
- [ ] Ensure `compress_bytes` swallows ALL exceptions → `fallback_original`/`error`; verify it never raises.
- [ ] Create `app/services/file_processing/compression_metrics.py` `record_compression(surface, result, *, filename, content_type, duration_ms)` mirroring `diagnostic_metrics.py`; event `compression.decision`; no PII/bytes/request-scoped ids in `extra`.
- [ ] Add `compression_min_bytes` (500KB), `compression_image_max_dim` (1200), `compression_image_max_kb` (200), `compression_pdf_per_page_kb` (300), `compression_pdf_dpi` (200), `compression_pdf_rebuild_enabled` (False) to `app/config.py` Settings (no `os.getenv`).
- [ ] Add `compress: bool = True` kwarg to `AttachmentsMixin.upload_attachment` (`_attachments.py:77-85`) and `upload_file_standalone` (`_attachments.py:170-176`).
- [ ] Insert compress-COPY hook after the empty-content guard, before the multipart build (before line 108 / 196) in both adapter methods; reassign `file_content = result.data`.
- [ ] Call `record_compression("erphome_attachment"/"erphome_standalone", ...)` at each hook with `duration_ms`.
- [ ] Keep the multipart `files` tuple (filename, bytes, content_type) + the `del`-Content-Type/multipart-boundary logic unchanged; keep each method's `try/except` envelope intact.
- [ ] Update existing `logger.info("Attachment uploaded"/"Standalone file uploaded")` `size` to reflect stored bytes (no duplicate event).
- [ ] Pass `compress=False` at `app/services/erpnext_provisioning/_steps_company.py:335` (branding logo opt-out); keep non-fatal try/except.
- [ ] Refactor `PDFHandler._has_sufficient_text` (`pdf_handler.py:302-308`) to delegate `is_native_pdf(..., mode="total")` (behavior identical).
- [ ] Refactor `library_gemini._extract_pdf_native` (`library_gemini.py:589-618`) to use `is_native_pdf(..., mode="average")`, preserving the `raise FileProcessingException` control-flow.
- [ ] Wire `compress_bytes` + `record_compression("genie_ocr", ...)` into `upload.py:1024-1025` (substitute compressed bytes into `_save_original_file`; OCR at 1044 still reads untouched `temp_path`).
- [ ] Confirm out-of-scope sinks untouched: `import_files/upload.py`, `files.py:upload_logo`, `attachments.py`/`FileStorageService` (no compression hook added).
- [ ] Confirm `_MAX_ATTACHMENT_SIZE_BYTES` stays 10MB at `_shared.py:111` (do NOT raise to 30MB); compression runs after the 413 checks.
- [ ] Create `tests/unit/file_processing/test_compression.py` (image shrink, skip_small, skip_native, scanned-pdf, fallback regression, corrupt→fallback, input-not-mutated).
- [ ] Create `tests/unit/file_processing/test_pdf_classifier.py` (total vs average boundary at 50 chars; mixed-density divergence).
- [ ] Create `tests/unit/adapters/test_attachments_compress.py` (respx: compressed bytes when compress=True, original when compress=False, filename/content_type unchanged, compressor-raise→fallback→success envelope).
- [ ] Create `tests/api/test_erp_attachment_compression.py` (respx ERPNext upload_file: stored bytes < uploaded for >500KB image; tiny file passthrough; cross-tenant BOLA `== 404`).
- [ ] Create `tests/unit/file_processing/test_compression_metrics.py` (event name + extra keys; no bytes/PII in logs via caplog).
- [ ] (Optional) Add `MetricsRegistry._init_compression_metrics()` to `app/core/metrics.py`, wired into `__post_init__` AND `reset_metrics()`; low-cardinality labels only.
- [ ] Verify new non-omitted modules are NOT added to `pyproject.toml [tool.coverage.run].omit`; run `ruff format --check`, `ruff check`, `mypy app/ --ignore-missing-imports`, and the new tests.
- [ ] Run `/preflight`; confirm full gate `pytest tests/ --cov=app --cov-fail-under=75 -q` passes and Alembic heads == 1 (no migration added here).

Section 9 is in `app/services/file_processing/compression.py`, `app/services/file_processing/_pdf_classifier.py`, `app/services/file_processing/compression_metrics.py` (CREATE) and `app/adapters/erpnext/_attachments.py`, `app/services/erpnext_provisioning/_steps_company.py`, `app/services/file_processing/handlers/pdf_handler.py`, `app/services/file_processing/ocr/library_gemini.py`, `app/api/v1/endpoints/upload.py`, `app/config.py` (MODIFY).

---

I have confirmed all the serve/preview seams. I have everything needed to write Section 10.

## 10. Preview / Download & On-Demand Unpack

### 10.1 Premise — preview is byte-transparent, so compression is preview-invisible by construction

Both preview surfaces serve **whatever bytes are stored, at whatever path was written**, with content-type derived from the *filename suffix* (Genie) or the upstream proxy response (ERPHome). Neither surface re-derives anything from the file's *contents*. Therefore, as long as compression honours **Corrected Decisions #3 and #4** — overwrite the single existing write, keep the same filename + extension — preview and download require **no functional change** to keep working. This section's job is to (a) prove that invariant against the two serve paths, (b) specify the optional on-demand DOCX/XLSX rich-preview unpack, (c) specify the additive FE transparency, and (d) enumerate the edge cases (stale optimistic `file_size`, fresh-upload local blob) that arise from compression changing the stored byte count.

This is a **read-mostly** section: most line items are *confirm + test*, not *modify*. The only net-new code is the optional unpack helper and the additive transparency fields.

### 10.2 Genie preview path — confirm byte/suffix transparency (no behavioral change)

**Serve route:** `serve_original_file()` at `app/api/v1/endpoints/upload.py:1933-2004` (GET `/api/v1/upload/sessions/{session_id}/originals/{filename}`, mounted via `router.py:851-856`).

Walk the serve flow against the compression seam:

1. Auth (header JWT or `?token=`) at `upload.py:1945-1960`, UUID session check at `1963-1969`, BOLA owner verify `_register_or_verify_session_owner` at `1972`, filename sanitize at `1975-1979` — **all content-agnostic; untouched by compression.**
2. `storage_path = f"sessions/{session_id}/originals/{filename}"` at `upload.py:1983` — identical to the persist path built in `_save_original_file()` at `upload.py:225-264` (single `upload_bytes` write at `258`). Because compression overwrites the *bytes* argument and **never** the filename (Decision #3 — do not add a 2nd write, the dup-suffixer at `249-254` would rename it), the serve path resolves the exact object the persist path wrote. **No change.**
3. `suffix = Path(filename).suffix.lower()` → `media_type = MIME_TYPE_MAP.get(suffix) or mimetypes.guess_type(...)` at `upload.py:1985-1988` using `MIME_TYPE_MAP` (`1908-1924`). This is **suffix-derived, not byte-sniffed** — the single hard dependency on Decision #4. A recompressed JPEG that stays `*.jpg` serves `image/jpeg`; a rebuilt scanned PDF that stays `*.pdf` serves `application/pdf`. **Confirm in a test that the served `Content-Type` is unchanged after compression.**
4. `storage.get_serve_response(storage_path, filename, media_type)` at `upload.py:1991` → `GCSStorageClient.get_serve_response` (`storage_client.py:378-396`): `exists()` then `download_bytes()`, returns `Response(content=data, media_type=media_type, headers={Content-Disposition: inline, Cache-Control: private, max-age=600})`. It is a **pure byte proxy** — it streams the compressed bytes verbatim, sets `Content-Length` from the actual payload. No signed URL (the v4 helper `_generate_signed_url_sync` at `storage_client.py:335-376` is dead and would 403 without `signBlob`). **No change.**

**Net for Genie preview: zero code modification.** Add regression coverage only (10.x checklist).

> **Cache-Control caveat (call out, do not break):** the serve `Response` sets `Cache-Control: private, max-age=600` (`storage_client.py:394`). Within a session, files are write-once to a unique path (dup-suffixer guarantees uniqueness), so a compressed object can never collide with a previously-cached original at the same URL — the 600s cache is safe. Do **not** introduce in-place re-compression of an existing path (it would also 403 under the objectCreator-only IAM per the storage seam).

### 10.3 ERPHome preview path — confirm proxy transparency (no behavioral change, not in compression scope for storage)

**Serve route:** `proxy_erp_file()` at `app/api/v1/endpoints/erp_documents/file_proxy.py:46-113` (GET `/api/v1/erp/files/file?path=...`).

This surface stores to **ERPNext only** (Decision #5 — no GCS on this path). The compression hook for ERPHome lives in the adapter (`_attachments.py` `upload_attachment`/`upload_file_standalone`) per the `be-erphome` seam; the serve path is downstream of ERPNext and just proxies bytes back:

1. `adapter.fetch_file(path)` at `file_proxy.py:69` returns `{content, content_type, file_name}`.
2. `content_type` comes from the **ERPNext upstream response** (`file_proxy.py:99`), *not* from a local suffix map. Because the adapter compression hook keeps the original filename + a matching `content_type` in the multipart tuple's 3rd element (per `be-erphome` gotcha — recompressed JPEG stays `image/jpeg`/`.jpg`), ERPNext stores and re-serves the correct type. **No change to the proxy.**
3. `Content-Length: str(len(content))` at `file_proxy.py:103` and `Content-Disposition` (RFC 5987) at `108` are computed from the actual bytes — automatically correct for the (smaller) compressed payload. **No change.**
4. FE `components/erp/file-preview-modal.tsx:42-212` routes on the **proxy's `Content-Type` header** (`contentType` state at line 47, `isPdf/isImage/isText` at `99-101`), so it is robust even if a filename changed — but filenames don't change anyway. **No change.**

**Net for ERPHome preview: zero code modification.** Confirm-only.

### 10.4 On-demand temp unpack for DOCX/XLSX rich preview (OPTIONAL — behind a flag, off by default)

**Context / why this is here:** DOCX and XLSX are *native* container formats (Decision #7 — left as-is, never compressed). They preview today only as a **download fallback**: Genie FE `chat-utils.ts` `PREVIEWABLE_EXTENSIONS` is `png/jpg/jpeg/gif/bmp/webp/tiff/pdf` (no office types), and ERPHome `file-preview-modal.tsx` falls through to "Preview not supported → Download" for non-PDF/image/text (`erp/file-preview-modal.tsx:189-205`). "On-demand unpack" means: when a user opens a DOCX/XLSX, *optionally* convert the stored container to a previewable artifact (HTML/text/first-sheet) **on the read path, in a temp dir, transiently** — never persisting a second copy, never touching the stored original.

**This is explicitly OPTIONAL and ships OFF.** It is not required for the core compression feature and must not block it. Gate it on a new `Settings` field (per the `be-logging` gotcha: knobs live in `app/config.py`, never `os.getenv`):

- **CREATE** flag in `app/config.py` (`Settings`): `rich_office_preview_enabled: bool = False` (subsystem-named, not `feature_*` — it is an infra/operational toggle per CLAUDE.md feature-flag rules, *not* a per-tenant plan feature).

**CREATE** `app/services/file_processing/office_preview.py` (new, non-omitted-from-coverage module so it counts toward the 75% gate — the `tests-ci` seam warns OCR/`upload.py` files are coverage-exempt):

```python
# app/services/file_processing/office_preview.py
logger = logging.getLogger(__name__)

class OfficePreviewResult(NamedTuple):
    media_type: str          # "text/html" | "text/plain"
    content: bytes
    rendered_from: str       # "docx" | "xlsx" — for telemetry

def render_docx_preview(data: bytes) -> OfficePreviewResult:
    """python-docx (already in requirements) -> minimal sanitized HTML/text.
    Reads from BytesIO(data); writes NOTHING to disk. Never mutates `data`."""

def render_xlsx_preview(data: bytes, *, max_rows: int = 200, max_cols: int = 50) -> OfficePreviewResult:
    """openpyxl/pandas (already in requirements) -> first-sheet bounded table HTML.
    Bounds rows/cols to avoid unbounded render of a huge sheet."""
```

Implementation notes:
- Operate on `BytesIO(data)` — **no `NamedTemporaryFile`, no temp_path** (avoids the `image_handler.py:162` in-place-overwrite class of bug entirely; the read path has no OCR temp to corrupt). If a lib genuinely needs a path (some openpyxl flows), write to a `tempfile.TemporaryDirectory()` that is `finally`-cleaned in the same call — never under `sessions/.../originals` and never via `storage.upload_bytes`.
- DOCX→HTML must be **sanitized** (escape/strip, allowlist tags) — the bytes originate from user upload; do not emit raw HTML the FE would render unsanitized. (FE renders previews via blob URL in an iframe already, but defense-in-depth.)
- Bounded: cap XLSX rows/cols (signature above) and DOCX output size; raise a domain exception (`FileProcessingException` family) on oversize/parse failure so the caller can fall back to download — **never** raise an unhandled error.

**MODIFY (guarded) the serve endpoints to call it only when the flag is on AND a `?render=html` query param is present:**
- Genie: extend `serve_original_file()` (`upload.py:1933`) — when `settings.rich_office_preview_enabled and render == "html" and suffix in {".docx",".xlsx"}`, fetch bytes via `storage.download_bytes(storage_path)`, run the renderer, return the HTML/`text` `Response`. Default path (no `render`) is **unchanged** — still the raw byte proxy. This keeps the existing download-fallback behavior intact when the flag is off.
- ERPHome: same shape in `proxy_erp_file()` (`file_proxy.py:62`) — guarded `render` branch over `data` fetched at `69`, else the existing `StreamingResponse`.

**If on-demand unpack is descoped:** do nothing on the serve paths — the current download-fallback (Genie: non-previewable ext → chip with download; ERPHome: `erp/file-preview-modal.tsx:189-205`) already works and is the acceptable baseline. Recommended default: **ship the flag wired but `False`, with the renderer + unit tests present**, so enabling later is a config flip, not a code change.

### 10.5 Frontend transparency — additive only, no change required for function

Per the `fe` seam, FE limits are already correct (30MB/32MB/5/10) and preview already keys off filename+extension, so **nothing must change for preview/download to keep working**. Transparency ("this file was compressed from X to Y") is purely additive observability and should land **only if** the BE returns the sizes:

**Genie (chat) — additive:**
- **MODIFY** `src/types/chat.ts` `ExtractionMetadata` (`442-464`): add optional `original_size?: number; stored_size?: number; was_compressed?: boolean`. Optional ⇒ no breakage to existing responses.
- **MODIFY** `src/components/chat/file-preview-modal.tsx` header (currently shows only `formatBytes(attachment.fileSize)` at `164-167`): when `was_compressed`, render `formatBytes(stored_size)` with a muted `compressed from {formatBytes(original_size)}` subline. i18n keys under `chat:file_preview` (`chat.json:134`) — **no hardcoded English** (CLAUDE.md i18n blocker).

**ERPHome — additive, only if compression-transparency is wanted there:**
- The adapter response would need to carry `original_size`/`stored_size` (the `be-erphome` seam already extends the adapter's `logger.info('Attachment uploaded', extra={...})` with these). Surface in `doc-side-panel.tsx` near `formatBytes(att.file_size)` (`212-214`). Lower priority than Genie; defer unless explicitly requested.

**Do NOT change** the FE size constants down to 10MB (the prompt's premise is already satisfied — `constants.ts:1893-1896` is 30MB). The MIN-500KB-exempt rule (Decision #8) is BE-only and has **no FE representation**; a user-facing hint, if any, belongs in the `chat-input.tsx` limit badge (`325-346`) and must be i18n'd, but is not required.

### 10.6 Edge cases (explicit handling)

1. **Stale optimistic `file_size` (Genie):** the FE shows the *pre-upload* local file size optimistically; after the BE compresses, the *stored* size is smaller. The preview header (`file-preview-modal.tsx:164`) shows `attachment.fileSize`. **Resolution:** once `was_compressed`/`stored_size` exist on `ExtractionMetadata`, prefer `stored_size` for the displayed bytes and keep `original_size` for the "compressed from" subline. If transparency is descoped, the optimistic size is a cosmetic over-statement only (it will reconcile to the response's `extraction.metadata.file_size` on the next render) — **acceptable, no breakage**. Document the cosmetic gap.
2. **Fresh-upload local blob until reload (Genie):** immediately post-upload the FE may render from a local `URL.createObjectURL` blob (the *original* bytes the browser holds) rather than the proxied *compressed* bytes. The two are visually identical (image/PDF), differing only in byte count. On reload, preview switches to the proxied compressed object via `useAuthedFileUrl`. **Resolution:** none required — visually equivalent; the only observable difference is the size number, covered by edge case (1). Note it so a reviewer doesn't file it as a bug.
3. **Compression fell back to original (Decision #9):** when compressed ≥ original or compression errored, the **original** bytes were stored (logged at WARNING with `fallback_reason` per `be-logging`). The serve path is identical (same filename/path), so preview works regardless. `was_compressed` must be `false` in that case so the FE shows no misleading "compressed" badge — the BE sets it from the *actual* persisted outcome, not the *intent*.
4. **Office file opened with rich preview flag OFF:** `serve_original_file`/`proxy_erp_file` return the raw container bytes; FE falls through to the download button (Genie chip / `erp/file-preview-modal.tsx:189-205`). **Confirm the download still yields a valid, openable DOCX/XLSX** (it must — these are never compressed, stored byte-identical).
5. **Recompressed image preview fidelity:** a recompressed JPEG (≤200KB, ≤1200px) is lossy vs the original. Preview shows the *stored* (compressed) image — correct, since storage holds the compressed copy. The OCR result was computed on the *original* temp_path (`upload.py:1044`) before unlink, so extraction quality is unaffected. Confirm the compressed image is still a valid decodable JPEG/PNG (round-trip test).
6. **Download = same path, no special-casing:** Genie has no separate download route (preview `Response` is `inline`; browser/FE handles save). ERPHome download is `?download=true` on the same `proxy_erp_file` (`file_proxy.py:64`, `106`) → `Content-Disposition: attachment`. Both stream the stored (compressed) bytes — **no download-specific change needed.**

### 10.7 Tests (placement per `tests-ci` seam — keep new logic out of coverage-omitted files)

- **Genie serve regression** → `tests/api/` (mirror `test_upload_cancel.py`): upload an image, monkeypatch `file_router.process`, assert `download_bytes("sessions/{sid}/originals/{filename}")` returns the *compressed* bytes AND that GET `/sessions/{sid}/originals/{filename}` returns `200` with the **suffix-derived `Content-Type` unchanged** and `Content-Length == len(compressed)`.
- **Suffix/content-type invariance** → assert a recompressed `*.jpg` still serves `image/jpeg` and a rebuilt `*.pdf` still serves `application/pdf` (guards Decision #4).
- **Office preview renderer units** → `tests/unit/file_processing/test_office_preview.py`: `render_docx_preview`/`render_xlsx_preview` over `BytesIO` fixtures (a tiny generated docx via python-docx, a tiny xlsx via openpyxl); assert media_type, bounded rows/cols, HTML sanitization, and graceful domain-exception on a corrupt blob. (LocalStorageClient-backed, no live GCS — conftest pins it.)
- **Guarded serve branch** → with `rich_office_preview_enabled=True` + `?render=html`, assert HTML body; with flag `False`, assert raw bytes (download fallback) — proving the flag truly gates and the default path is untouched.
- **Fallback-to-original preview** → store original (simulate compressed ≥ original), assert serve still `200` and `was_compressed=false` surfaced in the upload response.
- **FE** → extend `tests/unit/lib/api/upload.test.ts` for the new optional `ExtractionMetadata` fields passing through; place any pure size-display logic in a non-coverage-excluded `lib/utils` helper (component dirs are coverage-excluded per `fe`/`tests-ci`).

### 10.x Checklist

- [ ] Re-grep `serve_original_file` in `upload.py` to reconfirm line numbers before any edit (file is ~2004 lines; numbers drift).
- [ ] Confirm Genie serve path `storage_path` (`upload.py:1983`) byte-for-byte matches the persist path in `_save_original_file` (`upload.py:225-264`, write at `258`) — same filename, same extension, single write.
- [ ] Confirm `media_type` is suffix-derived (`MIME_TYPE_MAP` `upload.py:1908-1924`, resolved at `1985-1988`) and add a test asserting it is unchanged after compression.
- [ ] Confirm `GCSStorageClient.get_serve_response` (`storage_client.py:378-396`) is a pure byte proxy (no signed URL; dead `_generate_signed_url_sync` at `335-376` stays unused).
- [ ] Confirm the `Cache-Control: private, max-age=600` header (`storage_client.py:394`) is safe given write-once unique paths; verify no in-place re-compression of an existing object is introduced (would 403 under objectCreator-only IAM).
- [ ] Confirm ERPHome `proxy_erp_file` (`file_proxy.py:46-113`) sets `content_type` from the ERPNext upstream (`:99`) and `Content-Length` from actual bytes (`:103`) — no change needed.
- [ ] Confirm ERPHome FE `erp/file-preview-modal.tsx` routes on proxy `Content-Type` (`:47,99-101`), robust to byte-size change.
- [ ] DECIDE (product): ship on-demand DOCX/XLSX rich preview, or keep download-fallback baseline. Default recommendation: wire flag OFF + renderer + tests present.
- [ ] CREATE `Settings.rich_office_preview_enabled: bool = False` in `app/config.py` (subsystem-named infra toggle, NOT `feature_*`, NOT `os.getenv`).
- [ ] CREATE `app/services/file_processing/office_preview.py` with `render_docx_preview`/`render_xlsx_preview` (BytesIO-only, no temp_path mutation, sanitized HTML, bounded rows/cols, domain-exception on failure). Keep it OUT of the `pyproject.toml [tool.coverage.run].omit` list.
- [ ] MODIFY (guarded) `serve_original_file` (`upload.py:1933`) to branch to office_preview only when `rich_office_preview_enabled AND ?render=html AND suffix in {.docx,.xlsx}`; default path unchanged.
- [ ] MODIFY (guarded) `proxy_erp_file` (`file_proxy.py:62`) with the same flag-gated `render` branch; else existing `StreamingResponse`.
- [ ] Verify office_preview never calls `storage.upload_bytes` and never writes under `sessions/.../originals` (read-path only; no second persisted copy).
- [ ] MODIFY `src/types/chat.ts` `ExtractionMetadata` (`442-464`): add optional `original_size?`, `stored_size?`, `was_compressed?` (additive, no breakage).
- [ ] MODIFY `src/components/chat/file-preview-modal.tsx` header (`164-167`) to show `stored_size` + "compressed from {original_size}" subline when `was_compressed`; i18n under `chat:file_preview` (`chat.json:134`), no hardcoded English.
- [ ] (Optional/ERPHome) surface `original_size`/`stored_size` near `doc-side-panel.tsx` `formatBytes(att.file_size)` (`212-214`) only if BE attachment response carries them.
- [ ] Edge: prefer `stored_size` over optimistic `attachment.fileSize` in the Genie preview header once the field exists; document the cosmetic over-statement when transparency is descoped.
- [ ] Edge: document fresh-upload local-blob vs proxied-compressed visual equivalence (size-only delta); no code change.
- [ ] Edge: ensure BE sets `was_compressed=false` when compression fell back to original (Decision #9) so no misleading badge.
- [ ] Edge: confirm flag-OFF office download still yields a valid openable DOCX/XLSX (byte-identical, never compressed).
- [ ] Edge: round-trip test that a recompressed image is still a decodable JPEG/PNG and previews correctly.
- [ ] Edge: confirm ERPHome `?download=true` (`file_proxy.py:64,106`) streams compressed bytes with `Content-Disposition: attachment` — no download-specific change.
- [ ] TEST (`tests/api/`, mirror `test_upload_cancel.py`): Genie serve returns compressed bytes + unchanged suffix Content-Type + correct Content-Length.
- [ ] TEST (`tests/unit/file_processing/test_office_preview.py`): docx/xlsx renderers over BytesIO fixtures — media_type, bounds, sanitization, corrupt-blob domain-exception.
- [ ] TEST: guarded serve branch — `render=html` + flag ON returns HTML; flag OFF returns raw bytes.
- [ ] TEST: fallback-to-original still serves `200` and surfaces `was_compressed=false`.
- [ ] TEST (FE): extend `tests/unit/lib/api/upload.test.ts` for the optional `ExtractionMetadata` fields; put size-display logic in a non-excluded `lib/utils` helper to meet ≥85% new-code coverage.
- [ ] Run BE gate (`ruff` + `mypy app/ --ignore-missing-imports` + `pytest ... --cov-fail-under=75`) and FE gate (`format` + `lint` + `type-check` + `test:coverage`); keep Alembic single-head (no migration in this section).

---

I have everything I need. Writing Section 11 now.

## 11. Observability: Logging & Metrics (incl. every GCS op)

This section specifies the **complete telemetry surface** for the auto-compression feature: structured logs and metrics for every eligibility decision, per-format compression, validation, fallback, **every GCS/Local storage op** (upload + serve), and ERPNext attachment upload. It strictly mirrors the two house idioms confirmed in the seams — log-based events à la `app/services/tally_sync/diagnostic_metrics.py` (primary), and an in-process registry family à la `MetricsRegistry._init_brs_metrics` (secondary, for the Prometheus `/metrics` scrape) — and it actively **removes the existing CLAUDE.md violations** in `upload.py` (`_save_original_file` f-string logs at `upload.py:259`/`:263`, and the `"="*60` banner lines) rather than extending them.

### 11.1 House idioms this section binds to (do not deviate)

- **Logging**: module-level `logger = logging.getLogger(__name__)`. First arg is a **stable event name string**, never an f-string (`CLAUDE.md` → "never f-strings in log messages"). All per-call data goes in `extra={}`. `app/core/logging.py:JSONFormatter.format` (lines 57-90) auto-injects `get_log_context()` (request_id, tenant_id, user_id, **session_id**, layer, agent) and runs `redact_dict()` (line 88) before emit — so we **never** put request_id/tenant_id/user_id/session_id in `extra`, and we never log file bytes, tokens, GSTIN, or PII.
- **Log-based metrics (primary)**: a new thin-wrapper module, exactly the shape of `diagnostic_metrics.py` (`record_run`/`record_pipeline_filter` etc.). No registry wiring, no test-reset, lowest risk — this is the codebase's forward-looking idiom ("Prometheus is NOT used in this codebase", `diagnostic_metrics.py:10`). Cloud Monitoring derives metrics from `event name + extra`.
- **Registry metrics (secondary)**: a `MetricsRegistry._init_compression_metrics()` mirroring `_init_brs_metrics` (`metrics.py:864-909`), wired into `__post_init__` (`metrics.py:490-506`) **and** `reset_metrics()` (`metrics.py:980-993`) — the second wiring is mandatory or pytest leaks counter state across tests. Auto-exported at `GET /api/v1/metrics` (`observability.py:327-343` → `metrics.export()`).
- **Combined emit helper**: each public telemetry function does `counter.inc(...)` + `histogram.observe(...)` + `logger.info/warning("<event>", extra={...})` in one call, mirroring `metrics.log_form_generated` (`metrics.py:281-294`) and `feature_service.py:328-340`. Call sites stay one-liners.
- **`No double logging`**: global handlers in `app/main.py` log on raise. The compression fallback path is **recoverable, never raises** — `_save_original_file` is non-blocking (returns `None` on failure). So fallback is logged at `WARNING` (with `fallback_reason`), not `ERROR`, and no `logger.error(...)` precedes any `raise`.

### 11.2 Event taxonomy (stable names — these are the metric keys)

| Event name | Level | Emitted when | Carries (in `extra`) |
|---|---|---|---|
| `compression.eligibility` | INFO | After classification, before compress | `filename, content_type, surface, eligible, kind, skip_reason, original_size_bytes` |
| `compression.applied` | INFO | A format was compressed (success) | `filename, kind, strategy, original_size_bytes, compressed_size_bytes, saved_bytes, saved_pct, duration_ms` |
| `compression.skipped` | INFO | Below 500KB floor / native / unsupported | `filename, kind, skip_reason, original_size_bytes` |
| `compression.fallback` | WARNING | compress failed OR compressed ≥ original | `filename, kind, strategy, fallback_reason, original_size_bytes, compressed_size_bytes, duration_ms` |
| `storage.upload` | INFO (ok) / WARNING (error) | Every `upload_bytes` for an original | `storage_backend, bucket, storage_path, bytes, content_type, status, duration_ms` |
| `storage.serve` | INFO (ok) / WARNING (miss/error) | Every preview serve (`get_serve_response`) | `storage_backend, bucket, storage_path, bytes, content_type, status, duration_ms` |
| `erpnext.attachment.upload` | INFO (ok) / WARNING (error) | Every ERPHome adapter upload | `doctype?, surface, original_size_bytes, compressed_size_bytes, saved_pct, compressed, status, duration_ms` |

`surface` ∈ `{"genie_ocr", "erphome_attach", "erphome_standalone"}`. `kind` ∈ `{"image", "scanned_pdf", "native_pdf", "docx", "xlsx", "csv", "text", "other"}`. `strategy` ∈ `{"image_jpeg", "image_webp_to_jpeg", "pdf_raster_rebuild", "none"}`. `skip_reason` ∈ `{"below_min_floor", "native_pdf", "unsupported_format", "no_session"}`. `fallback_reason` ∈ `{"compress_error", "not_smaller", "validation_failed", "timeout"}`. `status` ∈ `{"ok", "error", "not_found"}`. **All label values are bounded low-cardinality enums** — `tenant_id`/`session_id`/`filename` are NEVER used as registry labels (high cardinality); they live only in the log `extra` (and tenant/session come free from `get_log_context`).

### 11.3 Files to CREATE

**1. `/Users/bhags/Desktop/erpsense_all/erpsense-backend/app/services/file_processing/compression_metrics.py`** (NEW — primary telemetry module; the one place call sites import). This is the `diagnostic_metrics.py` analog. **Critical placement note from the tests-ci seam**: `pyproject.toml [tool.coverage.run].omit` excludes `upload.py` and the entire OCR handler/engine set (`library_gemini.py`, `pdf_handler.py`, `image_handler.py`). This new module must **NOT** be added to that omit list so its lines count toward the 75% gate. Keep it ≤500 lines (modularity BLOCKER for new files).

```python
"""Compression + storage observability — log-based metrics (GCP-native)
mirroring app/services/tally_sync/diagnostic_metrics.py, plus a thin
bridge to the in-process MetricsRegistry for the /metrics scrape.

Every emit is `metrics.<counter>.inc()/<hist>.observe()` + a structured
`logger.<level>("<event>", extra={...})`. Request-scoped ids
(tenant_id, session_id, request_id) are auto-injected by JSONFormatter
via get_log_context() — never passed in extra. No file bytes / PII.
"""
from __future__ import annotations
import logging
from app.core.metrics import metrics

logger = logging.getLogger(__name__)

def _pct(original: int, final: int) -> float:
    return round((1 - final / original) * 100, 2) if original > 0 else 0.0

def record_eligibility(*, filename: str, content_type: str | None, surface: str,
                       eligible: bool, kind: str, original_size_bytes: int,
                       skip_reason: str | None = None) -> None:
    logger.info("compression.eligibility", extra={
        "filename": filename, "content_type": content_type, "surface": surface,
        "eligible": eligible, "kind": kind,
        "original_size_bytes": int(original_size_bytes),
        "skip_reason": skip_reason})

def record_applied(*, filename: str, kind: str, strategy: str, surface: str,
                   original_size_bytes: int, compressed_size_bytes: int,
                   duration_ms: float) -> None:
    saved = max(original_size_bytes - compressed_size_bytes, 0)
    pct = _pct(original_size_bytes, compressed_size_bytes)
    metrics.compression_decisions_total.inc(strategy=strategy, decision="compressed")
    metrics.compression_bytes_saved.observe(float(saved), strategy=strategy)
    metrics.compression_ratio_pct.observe(pct, strategy=strategy)
    metrics.compression_duration_seconds.observe(duration_ms / 1000.0, strategy=strategy)
    logger.info("compression.applied", extra={
        "filename": filename, "kind": kind, "strategy": strategy, "surface": surface,
        "original_size_bytes": int(original_size_bytes),
        "compressed_size_bytes": int(compressed_size_bytes),
        "saved_bytes": int(saved), "saved_pct": pct,
        "duration_ms": round(duration_ms, 2)})

def record_skipped(*, filename: str, kind: str, skip_reason: str, surface: str,
                   original_size_bytes: int) -> None:
    metrics.compression_decisions_total.inc(strategy="none", decision="skipped")
    logger.info("compression.skipped", extra={
        "filename": filename, "kind": kind, "skip_reason": skip_reason,
        "surface": surface, "original_size_bytes": int(original_size_bytes)})

def record_fallback(*, filename: str, kind: str, strategy: str, surface: str,
                    fallback_reason: str, original_size_bytes: int,
                    compressed_size_bytes: int | None = None,
                    duration_ms: float | None = None) -> None:
    metrics.compression_fallback_total.inc(reason=fallback_reason)
    metrics.compression_decisions_total.inc(strategy=strategy, decision="fallback")
    extra: dict[str, object] = {
        "filename": filename, "kind": kind, "strategy": strategy, "surface": surface,
        "fallback_reason": fallback_reason,
        "original_size_bytes": int(original_size_bytes)}
    if compressed_size_bytes is not None:
        extra["compressed_size_bytes"] = int(compressed_size_bytes)
    if duration_ms is not None:
        extra["duration_ms"] = round(duration_ms, 2)
    logger.warning("compression.fallback", extra=extra)  # recoverable → WARNING

def record_storage_upload(*, storage_backend: str, bucket: str | None, storage_path: str,
                          bytes_written: int, content_type: str, status: str,
                          duration_ms: float, error: str | None = None) -> None:
    metrics.storage_op_total.inc(op="upload", backend=storage_backend, status=status)
    metrics.storage_op_duration_seconds.observe(duration_ms / 1000.0, op="upload", backend=storage_backend)
    extra = {"storage_backend": storage_backend, "bucket": bucket,
             "storage_path": storage_path, "bytes": int(bytes_written),
             "content_type": content_type, "status": status,
             "duration_ms": round(duration_ms, 2)}
    if error:
        extra["error"] = error
    (logger.info if status == "ok" else logger.warning)("storage.upload", extra=extra)

def record_storage_serve(*, storage_backend: str, bucket: str | None, storage_path: str,
                         bytes_read: int | None, content_type: str | None, status: str,
                         duration_ms: float, error: str | None = None) -> None:
    metrics.storage_op_total.inc(op="serve", backend=storage_backend, status=status)
    metrics.storage_op_duration_seconds.observe(duration_ms / 1000.0, op="serve", backend=storage_backend)
    extra = {"storage_backend": storage_backend, "bucket": bucket,
             "storage_path": storage_path, "bytes": bytes_read,
             "content_type": content_type, "status": status,
             "duration_ms": round(duration_ms, 2)}
    if error:
        extra["error"] = error
    (logger.info if status == "ok" else logger.warning)("storage.serve", extra=extra)

def record_erpnext_attachment(*, surface: str, doctype: str | None,
                              original_size_bytes: int, compressed_size_bytes: int,
                              compressed: bool, status: str, duration_ms: float,
                              error: str | None = None) -> None:
    metrics.erpnext_attachment_total.inc(status=status, compressed=str(compressed).lower())
    metrics.erpnext_attachment_duration_seconds.observe(duration_ms / 1000.0, status=status)
    extra = {"surface": surface, "doctype": doctype,
             "original_size_bytes": int(original_size_bytes),
             "compressed_size_bytes": int(compressed_size_bytes),
             "saved_pct": _pct(original_size_bytes, compressed_size_bytes),
             "compressed": compressed, "status": status,
             "duration_ms": round(duration_ms, 2)}
    if error:
        extra["error"] = error
    (logger.info if status == "ok" else logger.warning)("erpnext.attachment.upload", extra=extra)
```

**Backend-scope note for the platform team (out of backend repo):** Cloud Monitoring log-based-metric definitions + alert policies for the event names above belong in `erpsense-platform/terraform/modules/monitoring/` (same boundary stated by `diagnostic_metrics.py:31-33`). Backend only emits the events; this section's deliverable stops at emit + the in-process registry.

### 11.4 Files to MODIFY

**1. `/Users/bhags/Desktop/erpsense_all/erpsense-backend/app/core/metrics.py`** — add `_init_compression_metrics()` mirroring `_init_brs_metrics` (`metrics.py:864`), and wire it in **both** `__post_init__` (after line 506) **and** `reset_metrics()` (after line 993).

```python
def _init_compression_metrics(self) -> None:
    """Auto-compression + storage-op metrics (Plan §11)."""
    self.compression_decisions_total = Counter(
        "erpsense_compression_decisions_total",
        "Compression decisions by strategy and outcome",
        ["strategy", "decision"],  # decision: compressed | skipped | fallback
    )
    self.compression_fallback_total = Counter(
        "erpsense_compression_fallback_total",
        "Compression fallbacks to original bytes",
        ["reason"],  # compress_error | not_smaller | validation_failed | timeout
    )
    self.compression_bytes_saved = Histogram(
        "erpsense_compression_bytes_saved",
        "Bytes saved per compressed file",
        ["strategy"],
        buckets=(0, 50_000, 100_000, 250_000, 500_000, 1_000_000, 5_000_000, 30_000_000),
    )
    self.compression_ratio_pct = Histogram(
        "erpsense_compression_ratio_pct",
        "Percent size reduction per compressed file",
        ["strategy"],
        buckets=(0, 10, 25, 50, 70, 85, 95, 99),
    )
    self.compression_duration_seconds = Histogram(
        "erpsense_compression_duration_seconds",
        "Per-file compression CPU time",
        ["strategy"],
        buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
    )
    self.storage_op_total = Counter(
        "erpsense_storage_op_total",
        "Storage ops (upload/serve) by backend and status",
        ["op", "backend", "status"],  # op: upload|serve  backend: gcs|local  status: ok|error|not_found
    )
    self.storage_op_duration_seconds = Histogram(
        "erpsense_storage_op_duration_seconds",
        "Storage op latency",
        ["op", "backend"],
        buckets=(0.005, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
    )
    self.erpnext_attachment_total = Counter(
        "erpsense_erpnext_attachment_total",
        "ERPHome attachment uploads by status and compressed flag",
        ["status", "compressed"],
    )
    self.erpnext_attachment_duration_seconds = Histogram(
        "erpsense_erpnext_attachment_duration_seconds",
        "ERPHome attachment upload latency",
        ["status"],
        buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0),
    )
```

Wiring (exact): add `self._init_compression_metrics()` as a new line after `metrics.py:506` (inside `__post_init__`) and after `metrics.py:993` (inside `reset_metrics`). **The `reset_metrics` addition is non-optional** — `tests-ci` gotcha: missing it leaks counters across xdist workers.

**2. `/Users/bhags/Desktop/erpsense_all/erpsense-backend/app/api/v1/endpoints/upload.py`** — at `_save_original_file` (`upload.py:225-264`), this is the single Genie persist seam. The compressed bytes are substituted upstream at the `_save_original_file` call (`upload.py:1024-1025`) per Decisions 1-4/9; here we **replace the two f-string logs and instrument the single `upload_bytes` write** (`upload.py:258`) with timing + the structured `storage.upload` event. The compression decision events (`compression.eligibility|applied|skipped|fallback`) are emitted from the compression module invoked just before line 1024 — not inside `_save_original_file` (one axis of change per file; storage persist ≠ compression decision).

Replace `upload.py:257-264` (drop the f-strings at 259/263):
```python
content_type = mimetypes.guess_type(safe_name)[0] or "application/octet-stream"
backend = "gcs" if type(storage).__name__ == "GCSStorageClient" else "local"
_t0 = time.perf_counter()
try:
    await storage.upload_bytes(storage_path, content, content_type=content_type)
except Exception as e:
    record_storage_upload(storage_backend=backend, bucket=getattr(storage, "_bucket_name", None),
                          storage_path=storage_path, bytes_written=len(content),
                          content_type=content_type, status="error",
                          duration_ms=(time.perf_counter() - _t0) * 1000, error=str(e))
    logger.warning("upload.original_save_failed", extra={"filename": safe_name})  # non-blocking
    return None
record_storage_upload(storage_backend=backend, bucket=getattr(storage, "_bucket_name", None),
                      storage_path=storage_path, bytes_written=len(content),
                      content_type=content_type, status="ok",
                      duration_ms=(time.perf_counter() - _t0) * 1000)
return storage_path
```
Also: the `"="*60` banner logs (`upload.py:259, 323-372` per the be-logging seam) are CLAUDE.md violations — **do not extend them**; where touched in the compression path, replace with the structured events above. Keep the existing `except Exception` non-blocking contract (`upload.py:262-264`) intact.

**3. `/Users/bhags/Desktop/erpsense_all/erpsense-backend/app/api/v1/endpoints/upload.py`** — `serve_original_file` (`upload.py:1933-1995` per be-storage; `serve_original_file` end at 2004). Wrap the `storage.get_serve_response(...)` call with timing and `record_storage_serve(... status="ok"/"not_found"/"error")`, so every **preview** op is observable (the storage layer's existing `%`-style DEBUG logs at `storage_client.py:228, 252, 285` stay as-is — we add structured telemetry at the **caller**, keeping the storage layer transport-only, per the be-logging gotcha). `bytes_read`/`content_type` come from the serve `Response` (`Content-Length`/`media_type`).

**4. `/Users/bhags/Desktop/erpsense_all/erpsense-backend/app/adapters/erpnext/_attachments.py`** — `upload_attachment` (lines 77-168) and `upload_file_standalone` (170-233). The existing `logger.info` at `_attachments.py:147` and `:222` already use `extra={}` (compliant). Extend each to call `record_erpnext_attachment(...)` after the POST, capturing `duration_ms`, `original_size_bytes`/`compressed_size_bytes` (the compression hook adds these locally before the multipart build at `:108`/`:196`), `compressed: bool`, and `status` derived from the response envelope. The provisioning logo path (`_steps_company.py:335`, `compress=False`) records `compressed=False`. A compression exception inside the method must be caught and recorded as `compression.fallback` (reason `compress_error`) then proceed with original bytes — never bubble out of the adapter's `try/except` success envelope (be-erphome gotcha).

### 11.5 Correlation, tenant/session fields, data safety

- **Correlation comes free.** `get_log_context()` (`request_context.py:188-222`) injects `request_id`, `tenant_id`, `user_id`, `session_id`, `layer`, `agent` into every line via `JSONFormatter.format` (`logging.py:68`). The Genie upload flow already sets `session_id_ctx`; the auth dep sets `tenant_id`/`user_id`. **We add NONE of these to `extra`** (be-logging gotcha: duplicating shadows the canonical value).
- **Registry labels are low-cardinality only** (`strategy, decision, reason, op, backend, status, compressed`). `tenant_id` is **never** a Prometheus label (high cardinality) — it rides only in the log `extra` (auto-injected) for forensic per-upload correlation.
- **No PII / no bytes.** Only filename, sizes, ratios, durations, backend, path are logged. `redact_dict()` runs in `JSONFormatter.format:88` as a backstop, but we don't rely on it. The FE proxy (`[...path]/route.ts` `VERBOSE_LOG_PATHS`) deliberately does not log upload bodies — the spec's "logging for all ops" is satisfied **backend-side** here, never by dumping bytes at the FE.
- **Optional new contextvar — NOT needed.** No new `contextvar` is introduced (so no `_ALL_CONTEXT_VARS` append / set-reset-in-finally obligation). If a dedicated `layer_ctx="adapter"` is desired around the compression entry point it should be set/reset by the existing entry-point code, not a new var.

### 11.6 Log levels & sampling

- **INFO** = business events: `compression.eligibility`, `compression.applied`, `compression.skipped`, `storage.upload (ok)`, `storage.serve (ok)`, `erpnext.attachment.upload (ok)`.
- **WARNING** = recoverable: `compression.fallback` (always — compressed≥original or compress error → stored original), `storage.upload (error)` (non-blocking — OCR proceeds), `storage.serve (not_found/error)`, `erpnext.attachment.upload (error)`.
- **ERROR**: none emitted here. Request-failing errors are raised and logged once by `app/main.py` global handlers (No-double-logging). The compression path never raises.
- **Sampling**: emit **1 line per file per stage** (uploads are low-volume, ≤5 files/session, synchronous inline — no high-frequency loop). The only loops are ERPHome multi-attachment (`employee_expense_service.py:617/651`, `employee_travel_service.py:750`) which iterate a handful of files — one `erpnext.attachment.upload` per file is correct, no sampling. Zero-count lines are suppressed for `compression.skipped`-style aggregates the way `diagnostic_metrics.record_pipeline_filter` early-returns (`diagnostic_metrics.py:146-147`) only where a zero would be pure noise; eligibility/applied/fallback always emit (they are the decision audit trail).

### 11.7 How existing behavior stays unbroken

- **`upload_bytes`/`get_serve_response` signatures untouched** (`storage_client.py:41, 378-396`) — telemetry is added at callers (`upload.py`), not in the storage layer; the layer keeps its `%`-style DEBUG logs. No second write is added at `_save_original_file` (the dup-filename suffixer at `:249-254` is preserved; we only instrument the existing single write at `:258`).
- **Non-blocking contract preserved**: `_save_original_file` still returns `None` on any failure (`upload.py:262-264`); a `record_storage_upload(status="error")` is purely additive and does not raise, so OCR (`upload.py:1044`) still runs.
- **Registry is additive + reset-safe**: `_init_compression_metrics` is initialized in `__post_init__` and `reset_metrics`, so the existing `/api/v1/metrics` export (`observability.py:327-343`) gains new series with zero impact on existing families; pytest stays leak-free across `-n 4 --dist=loadfile`.
- **ERPHome adapter envelope unchanged**: telemetry wraps the existing POST; the `{success, error}` envelope (be-erphome) and the deliberate Content-Type/Host header handling (`_attachments.py:135, 210`) are not modified.
- **Coverage gate**: telemetry call sites in `upload.py`/`_attachments.py` live in coverage-omitted files, but the **measured logic** (`compression_metrics.py` + `_init_compression_metrics` in the non-omitted `metrics.py`) is exercised by unit tests (§ test plan), keeping new code ≥85% and the merged dataset ≥75% (`deploy.yml` `--cov-fail-under=75`).

### 11.8 Checklist

- [ ] Create `app/services/file_processing/compression_metrics.py` with `record_eligibility`, `record_applied`, `record_skipped`, `record_fallback`, `record_storage_upload`, `record_storage_serve`, `record_erpnext_attachment` (module-level `logger = logging.getLogger(__name__)`, event-name-first, all data in `extra`, no f-strings, ≤500 lines).
- [ ] Verify `compression_metrics.py` is NOT added to `pyproject.toml [tool.coverage.run].omit` (must count toward 75%).
- [ ] Add `MetricsRegistry._init_compression_metrics()` to `app/core/metrics.py` with the 9 metrics in §11.4 (counters: `compression_decisions_total{strategy,decision}`, `compression_fallback_total{reason}`, `storage_op_total{op,backend,status}`, `erpnext_attachment_total{status,compressed}`; histograms: `compression_bytes_saved`, `compression_ratio_pct`, `compression_duration_seconds`, `storage_op_duration_seconds`, `erpnext_attachment_duration_seconds`).
- [ ] Wire `self._init_compression_metrics()` into `MetricsRegistry.__post_init__` (after `metrics.py:506`).
- [ ] Wire `metrics._init_compression_metrics()` into `reset_metrics()` (after `metrics.py:993`) — mandatory for test isolation.
- [ ] Confirm no `tenant_id`/`session_id`/`filename` used as registry labels (low-cardinality enums only).
- [ ] In `upload.py:_save_original_file` replace the f-string log at `:259` with `record_storage_upload(... status="ok")` (timed around the `upload_bytes` at `:258`).
- [ ] In `upload.py:_save_original_file` replace the f-string warning at `:263` with structured `upload.original_save_failed` + `record_storage_upload(status="error")`, preserving the non-blocking `return None` contract.
- [ ] Derive `storage_backend` via `type(storage).__name__` and `bucket` via `getattr(storage, "_bucket_name", None)`; add `import time` if absent.
- [ ] Instrument `upload.py:serve_original_file` (1933-2004) with `record_storage_serve(status=ok|not_found|error)` around `storage.get_serve_response`.
- [ ] Emit `compression.eligibility` / `applied` / `skipped` / `fallback` from the compression module at the Genie call site (just before `upload.py:1024`), with `surface="genie_ocr"`.
- [ ] Do NOT extend the `"="*60` banner logs (`upload.py:259, 323-372`); leave/replace, never add new banner/f-string logs.
- [ ] Instrument `_attachments.py:upload_attachment` (147) and `upload_file_standalone` (222) with `record_erpnext_attachment(...)` (timed, `compressed` flag, `surface`, `status` from envelope).
- [ ] Ensure compression exceptions inside the adapter are caught → `record_fallback(reason="compress_error")` → original bytes → normal success envelope (never bubble).
- [ ] Provisioning logo push (`_steps_company.py:335`, `compress=False`) records `erpnext.attachment.upload` with `compressed=False`.
- [ ] Confirm NO request-scoped ids (request_id/tenant_id/user_id/session_id/layer/agent) are passed in any `extra={}` (auto-injected by `JSONFormatter`).
- [ ] Confirm logs contain only filename/sizes/ratios/durations/backend/path/strategy — no file bytes, tokens, GSTIN, bank details, or PII.
- [ ] Set levels: INFO for ok/decision events, WARNING for fallback + storage/attachment errors, ERROR for none (no double-logging before raises).
- [ ] Unit-test `compression_metrics.py` helpers (assert counter `.inc`/histogram `.observe` called with correct labels and that `extra` excludes PII) under `tests/unit/file_processing/` or `tests/unit/core/`.
- [ ] Unit-test `_init_compression_metrics` registration + `reset_metrics()` reset (assert series exist in `metrics.export()` and reset to zero) under `tests/unit/core/`.
- [ ] Add a note in the plan that Cloud Monitoring log-based-metric + alert definitions for these event names are platform-team scope (`erpsense-platform/terraform/modules/monitoring/`), out of backend repo.
- [ ] Run `ruff format --check && ruff check && mypy app/ --ignore-missing-imports` and the metrics/compression unit tests green before declaring §11 done.

---

Now I have enough verified detail. Writing Section 12.

## 12. Security Review & Hardening (repeated gates)

This feature adds CPU/memory-bound transformation of *attacker-controlled file bytes* on two surfaces — Genie OCR (`upload.py:upload_file()` → compress a copy of `content` → `_save_original_file()` → `storage.upload_bytes`) and ERPHome attach (`_attachments.py:upload_attachment` / `upload_file_standalone`). Compression code parses, decodes, rasterizes and rebuilds untrusted PDFs/images, so the threat surface is materially larger than "store bytes". This section enumerates the threats, the concrete mitigations (file:symbol, files to CREATE/MODIFY, signatures), and a repeatable checklist to run **every implementation pass**. The overriding non-functional invariant from CORRECTED DECISION #9 is load-bearing for security too: **no compression code path may raise into the request** — a hostile file must degrade to "store original, log, continue", never to a 500 or a hang.

### 12.1 New attack surface introduced by this feature

| Threat | Vector | Sink |
|---|---|---|
| Image decompression bomb | Tiny PNG/GIF that decodes to gigapixels | `Pillow Image.open(BytesIO(content))` in the new compressor |
| PDF bomb / nested-object / deeply-nested xref | Crafted PDF that explodes on `fitz.open` / per-page `get_pixmap` | scanned-PDF rasterize path (mirrors `pdf_handler.py:_pdf_to_images` 310-326) |
| Encrypted / password / malformed PDF | `fitz.open` raises or hangs | scanned-PDF classify + rasterize |
| Content-type / extension spoofing | `.jpg` that is actually a PDF/HTML/SVG/polyglot | classification by suffix in compressor + multipart 3rd-tuple in `_attachments.py:108/196` |
| Path traversal in storage key | `../` or null bytes in `original_filename` reaching `storage_path` | `_save_original_file()` `storage_path = f"sessions/{sid}/originals/{safe_name}"` (upload.py:247) |
| Cross-tenant storage read/write | Forged `session_id` / filename on serve/persist | `serve_original_file()` (upload.py:1927-2004) + storage key composition |
| CPU/memory DoS | Many large images compressed "in parallel" inside one request | `asyncio.gather` + `asyncio.to_thread` compression in `upload_file()` |
| Secrets/PII in logs | New compression/GCS structured logs | new `compression_metrics.py` + `logger.info("upload.original_saved", …)` |
| GCS privilege escalation / overwrite | Re-uploading compressed bytes to an existing key | `GCSStorageClient.upload_bytes` (objectCreator only — no overwrite/delete) |

### 12.2 Decompression / image bombs (Pillow) — hard caps

CREATE `app/services/file_processing/compression.py` (NOT in `pyproject.toml [tool.coverage.run].omit`, so it counts toward the 75% gate; `upload.py`/`image_handler.py`/`pdf_handler.py` are omitted and must not host the logic). Set the Pillow guard **at module import**, before any `Image.open`:

```python
from PIL import Image
# Hard cap BEFORE any decode. None == disabled (vulnerable); a low int raises DecompressionBombError.
Image.MAX_IMAGE_PIXELS = settings.compression_max_image_pixels  # e.g. 24_000_000 (~ 24MP)
```

Mitigations, all in `compress_image(content: bytes, content_type: str) -> CompressionResult`:
- Open from a `BytesIO(content)` **copy**, never `temp_path` (the `image_handler.py:_resize_if_needed` `img.save(file_path)` overwrite hazard at line 162 means OCR's temp must stay pristine — DECISION #1).
- Wrap decode in `try/except (Image.DecompressionBombError, Image.DecompressionBombWarning, OSError, Exception)` → on any failure return `CompressionResult(bytes=content, compressed=False, fallback_reason="decode_failed")`. Treat `DecompressionBombWarning` as an error too (set `warnings.simplefilter("error", Image.DecompressionBombWarning)` locally or check `img.size` product against the cap explicitly).
- Independent dimension cap (`width*height` and each axis) re-checked **after** `Image.open` lazy header parse via `img.size` — do not rely solely on `MAX_IMAGE_PIXELS` because animated/multiframe formats can bypass per-frame.
- Replicate `image_handler.py:159-161` RGBA/P→RGB conversion before JPEG encode (compatibility + strips alpha-channel bombs).
- Strip EXIF/metadata on re-encode (`img.save(out, format=..., exif=b"")`) — removes GPS/PII and oversized APPn segments.
- Add knobs to `app/config.py` Settings (NOT `os.getenv` — R03): `compression_enabled: bool = True`, `compression_max_image_pixels: int`, `compression_image_target_kb: int = 200`, `compression_image_max_dim: int = 1200`, `compression_scanned_pdf_target_kb_per_page: int = 300`, `compression_min_exempt_bytes: int = 500*1024`, `compression_pdf_rebuild_enabled: bool = False` (DECISION #7 — off by default), `compression_pdf_render_dpi: int = 150`.

### 12.3 Malicious / encrypted / bomb PDFs

In `compress_scanned_pdf(content: bytes) -> CompressionResult` (only reached when classified **scanned** — see 12.4):
- `fitz.open(stream=content, filetype="pdf")` on a **copy**, inside `to_thread`, inside the existing `asyncio.wait_for` budget. Reject `doc.is_encrypted` / `doc.needs_pass` → fallback original.
- Re-enforce page cap independent of upload-time `MAX_PAGES_PER_PDF=10` (upload.py:84): `if doc.page_count > settings... : fallback` — the upload page-count check (upload.py:954-1019) runs before compression so this is defence-in-depth, not the primary gate.
- Per-page `get_pixmap` capped DPI (`compression_pdf_render_dpi` 150, lower than OCR's 200 in `pdf_handler.py:311` — output quality, not OCR input) and a running output-size budget; abort to fallback if cumulative bytes exceed `page_count * target_kb_per_page`.
- The rebuild path (DECISION #7, `pikepdf`/`img2pdf` NOT available — confirmed `requirements.txt` has only `PyMuPDF>=1.23.0`, `Pillow>=10.0.0`, `reportlab>=4.1`) is **behind `compression_pdf_rebuild_enabled=False`**. Keep it flag-gated so the riskiest code path ships dark and is enabled only after review.
- Hard wall-clock guard: the whole compression block runs under `asyncio.wait_for(..., settings.compression_timeout_s)` (separate, smaller than `_OCR_PROCESSING_TIMEOUT=300`) so a slow bomb cannot consume the full 300s OCR budget.

### 12.4 Content-type / extension spoofing — sniff before transform

The decision "compress only images + scanned PDFs, leave native PDF/DOCX/XLSX/CSV as-is" must be driven by **real bytes**, not the client-supplied suffix/`content_type`. Add a magic-byte sniff:
- `python-magic` is NOT in `requirements.txt` today. Either ADD it (preferred — `python-magic` + libmagic, note libmagic must be in the backend Docker image) or implement a minimal in-repo signature check (PNG `\x89PNG`, JPEG `\xff\xd8\xff`, GIF `GIF8`, PDF `%PDF`, etc.) inside `compression.py`. Document the choice; if adding the dep, update `requirements.txt` and the Dockerfile.
- Classification flow in `compression.py:classify(content, declared_suffix) -> Literal["image","scanned_pdf","native","skip"]`:
  1. sniff true type from bytes.
  2. if sniffed type ≠ declared suffix family → log `compression.decision` with `decision="skip_type_mismatch"` and store original (do NOT silently "fix" the extension — DECISION #4 requires same filename+extension; a mismatch is a security signal, fall back).
  3. PDF: reuse the shared classifier helper (see 12.5) to split native vs scanned; only scanned → compress.
  4. Never attempt to compress `.svg`/`.html`/unknown → `skip`.
- This also blocks polyglots: a file that sniffs as PDF but is named `.jpg` is stored as-is, never fed to the image encoder, and OCR (which reads the untouched `temp_path`) is unaffected.

### 12.5 Reuse the OCR classifier without changing OCR behavior

Per the OCR seam, two classifiers with **different semantics** must not be collapsed: `PDFHandler._has_sufficient_text` (pdf_handler.py:302-308, TOTAL `len(text) >= 50*page_count`, branches) and `LibraryGeminiProcessor._extract_pdf_native` (library_gemini.py:589-618, AVERAGE `avg < 50`, raises). CREATE `app/services/file_processing/handlers/_pdf_classifier.py` exposing `is_native_pdf(text, page_count, *, mode)` parameterized by mode/threshold; delegate from both existing sites preserving each one's semantics + raise-vs-branch control flow; the compressor calls the SAME helper so "scanned" means exactly what OCR thinks it means. Security relevance: keeps the compress/skip decision in lockstep with OCR so an attacker cannot craft a PDF that OCR treats as native (skipping Gemini) while compression rebuilds it (or vice versa), avoiding divergent-parse confusion.

### 12.6 Path traversal & tenant isolation of storage keys

- `_save_original_file()` already sanitizes (upload.py:241-244): `replace("/","_").replace("\\","_").replace("\x00","")` + `lstrip(".")`. **Keep and re-verify** this on every pass — compression must NOT introduce a second filename derivation. Because compression preserves the exact `safe_name` (DECISION #4), no new traversal vector is added. Do NOT add a 2nd `upload_bytes` write (the dup-suffixer at 249-254 would create `{name}_1` and the cancel-purge in `_purge_upload_artifacts` matches `{stem}.{ext}`/`{stem}_{counter}.{ext}`, so a divergent key would orphan the file).
- `LocalStorageClient._resolve` (storage_client.py:110-114) enforces `resolved.startswith(base_dir.resolve())` → `FileValidationException`. Add/keep a unit test asserting traversal denial still fires for the compressed-bytes path.
- Tenant isolation: `serve_original_file()` (upload.py:1927-2004) verifies UUID `session_id` (1963), session-owner (1972), filename sanitize (1975), auth via header-or-`?token`. Compression does not touch this path; the **mandatory regression** is that a cross-tenant GET of `sessions/{other_sid}/originals/{file}` still returns 404 (BOLA test, strict `== 404`, mirror `test_file_attachments.py:420-450`).
- ERPHome side has NO GCS sink (DECISION #5) — do NOT add one. The compressed bytes go only into the multipart tuple to ERPNext; storage-key traversal is N/A there, but the multipart **filename** must remain the sanitized original (ERPNext derives its File name + serve content-type from it).

### 12.7 DoS / resource exhaustion

- Size caps already gate before compression: `MAX_FILE_SIZE=30MB` (upload.py:81), `MAX_TOTAL_SIZE=32MB` (82), `MAX_FILES_PER_SESSION=5` (83), `MAX_PAGES_PER_PDF=10` (84). Compression runs **after** all of these (after the page-count block at 1019, before persist at 1024). ERPHome cap is genuinely `_MAX_ATTACHMENT_SIZE_BYTES=10MB` (_shared.py:111) — do NOT raise it (DECISION #8 30MB applies to Genie only).
- MIN-500KB exempt (DECISION #8): files `< compression_min_exempt_bytes` skip compression entirely → less CPU, fewer bombs reach the decoder.
- "Parallel" = `asyncio.gather` of `asyncio.to_thread(compress_one, ...)` **within one request** (DECISION #6, sync inline, no worker). Bound it: cap concurrency to a small constant (the per-session 5-file limit already bounds fan-out) and never spawn unawaited fire-and-forget tasks (the xdist `--dist=loadfile` loop-leak gotcha). The whole block under `asyncio.wait_for` (12.3) so CPU is time-bounded.
- `to_thread` keeps the event loop responsive; `MAX_IMAGE_PIXELS` + DPI cap bound per-op memory.

### 12.8 GCS IAM least-privilege, secrets, DEV correctness

- Cloud Run SA has `objectCreator + objectViewer` ONLY (phase2/main.tf 35-38) — no overwrite, no delete, no `signBlob`. The compressed write must go to a **fresh key** (the existing single write at storage_client.py:258 via the dup-suffixer); never overwrite an existing object or it 403s in prod. Do NOT switch preview to the dead `_generate_signed_url_sync` (storage_client.py:335-376) — it needs ungranted `iam.serviceAccounts.signBlob`; keep the byte-proxy (`get_serve_response`).
- Secrets: ADC only (no key file). No new IAM needed (`upload_bytes`/`download_bytes`/`exists` covered). Do NOT add new GCS scopes for this feature.
- DEV-correctness assertion (DECISION #10): tests force LocalStorageClient (conftest 209-213). Add a **dev smoke** step (manual/CI smoke job, not unit) that asserts `type(get_storage_client()).__name__ == "GCSStorageClient"` when a bucket is configured — guards the silent-degrade-to-local footgun (storage_client.py:412-417 only WARNs).

### 12.9 No secrets / PII in logs

- New logs use the house idiom (event-name first arg, data in `extra=`, NO f-strings — CLAUDE.md hard rule). REPLACE the existing f-string at upload.py:259 (`[UPLOAD] Saved original file: …`) and the `'='*60` banners (upload.py:323-372) with `logger.info("upload.original_saved", extra={...})`. CREATE `app/services/file_processing/compression_metrics.py` mirroring `tally_sync/diagnostic_metrics.py`: events `compression.decision`, `compression.fallback`, `gcs.upload`.
- Allowed `extra` fields ONLY: `filename`, `content_type`, `original_size_bytes`, `compressed_size_bytes`, `saved_bytes`, `saved_pct`, `strategy`, `decision`, `fallback_reason`, `storage_path`, `storage_backend`, `duration_ms`. NEVER log file bytes, base64, tokens, GSTIN, or PII. `request_id`/`tenant_id`/`user_id`/`session_id`/`layer` are auto-injected by `JSONFormatter` (logging.py:68) — do NOT duplicate. `redact_dict()` runs at logging.py:88 but is a backstop, not a license.
- "No double logging": on the fallback path log at WARNING with `fallback_reason` (recoverable). Do NOT `logger.error` before any raise — but there should be NO raise here anyway (12.10).

### 12.10 Fail-safe contract (the security keystone)

Every compression entry point (`compress_image`, `compress_scanned_pdf`, the gather wrapper, the ERPHome adapter hook) wraps its body in `try/except Exception` and returns the **original bytes** on ANY error or when `compressed_size >= original_size` (DECISION #9). Specifically:
- `_save_original_file()` is explicitly non-blocking (returns `None`, OCR proceeds) — preserve that contract; a compression failure must never block OCR or the request.
- `_attachments.py` methods already return `{"success": False, "error": {...}}` envelopes — a compression exception inside must be caught and fall back to original bytes, still returning the normal success envelope (never bubble unhandled).
- Provisioning logo (`_steps_company.py:335`) passes `compress=False` (opt-out kwarg on both adapter methods) so the already-optimized branding asset is not re-decoded — closes a needless decode of admin-controlled bytes and a behavior-change risk.

### 12.11 Files to CREATE / MODIFY (security-relevant)

CREATE: `app/services/file_processing/compression.py` (caps, sniff, fail-safe), `app/services/file_processing/compression_metrics.py` (safe structured events), `app/services/file_processing/handlers/_pdf_classifier.py` (shared classifier). MODIFY: `app/config.py` (Settings knobs incl. `MAX_IMAGE_PIXELS` value, flags off-by-default), `app/api/v1/endpoints/upload.py` (wire compressed bytes into `_save_original_file` call at 1024-1025; replace f-string log at 259), `app/adapters/erpnext/_attachments.py` (hook before multipart at 108/196 + `compress=False` kwarg), `app/services/erpnext_provisioning/_steps_company.py:335` (pass `compress=False`), `requirements.txt`/Dockerfile (if adding `python-magic`/libmagic). Tests: `tests/unit/file_processing/test_compression.py` (bombs, spoofing, fallback), `tests/unit/core/test_storage_client.py` (traversal still denied), `tests/api/test_upload_*` (persist-compressed + BOLA 404), `tests/api/v1/` (ERPHome respx + fallback).

### 12.12 Checklist
- [ ] `Image.MAX_IMAGE_PIXELS` set to a finite cap at import in `compression.py`; `DecompressionBombWarning` treated as error.
- [ ] Image decode/encode operates on `BytesIO(content)` copy; `temp_path` never touched by compression (OCR reads pristine original).
- [ ] Independent width/height/megapixel re-check after `Image.open`; RGBA/P→RGB conversion mirrored from `image_handler.py:159-161`; EXIF/metadata stripped on re-encode.
- [ ] `fitz.open(stream=copy)` rejects `is_encrypted`/`needs_pass`; per-page DPI capped; cumulative output-size budget aborts to fallback.
- [ ] PDF rebuild path gated behind `compression_pdf_rebuild_enabled=False` (off by default); no `pikepdf`/`img2pdf` introduced.
- [ ] Whole compression block under `asyncio.wait_for(settings.compression_timeout_s)` smaller than `_OCR_PROCESSING_TIMEOUT=300`.
- [ ] Magic-byte sniff before transform; type/suffix mismatch → store original + `decision="skip_type_mismatch"` (no silent rename); `.svg`/`.html`/unknown → skip.
- [ ] If `python-magic` added: `requirements.txt` + libmagic in Docker image updated; else in-repo signature check documented.
- [ ] Shared `_pdf_classifier.is_native_pdf` delegated from both `PDFHandler._has_sufficient_text` and `library_gemini._extract_pdf_native` preserving each semantics + raise-vs-branch; compressor uses the same helper.
- [ ] Filename sanitization at `_save_original_file` (upload.py:241-244) intact; compression preserves exact `safe_name`; same filename+extension (serve content-type is suffix-derived).
- [ ] Exactly ONE `upload_bytes` write (storage_client.py:258); no 2nd write; dup-suffixer + cancel-purge (`_purge_upload_artifacts`) still match the key.
- [ ] `LocalStorageClient._resolve` traversal denial unit-tested for the compressed-bytes path.
- [ ] Cross-tenant `serve_original_file` GET returns strict `== 404` (BOLA regression test).
- [ ] No GCS sink added to the ERPHome path (DECISION #5); ERPHome cap stays 10MB (`_shared.py:111`).
- [ ] Files `< compression_min_exempt_bytes` (500KB) skip compression.
- [ ] "Parallel" = `asyncio.gather` + `asyncio.to_thread` within the request; bounded concurrency; no unawaited fire-and-forget tasks.
- [ ] Compressed write targets a fresh key only (objectCreator can't overwrite/delete); preview stays byte-proxy (`get_serve_response`), signed-URL helper untouched.
- [ ] No new GCS IAM scopes; ADC only; no key file added.
- [ ] DEV smoke asserts `GCSStorageClient` selected when bucket configured (guards silent local degrade).
- [ ] All new logs: event-name first arg + `extra=`, NO f-strings; upload.py:259 f-string + `'='*60` banners replaced.
- [ ] `extra` carries only filename/size/pct/strategy/decision/reason/backend/duration; no bytes/tokens/PII; request/tenant/session ids NOT duplicated.
- [ ] Fallback (error OR compressed≥original) returns original bytes, no retry, logged WARNING with `fallback_reason`; never raises into request.
- [ ] `_save_original_file` non-blocking contract preserved; OCR proceeds on compression failure.
- [ ] ERPHome adapter compression exception caught → original bytes + normal success envelope; provisioning logo passes `compress=False`.
- [ ] No `logger.error` before any raise (no double logging); global handlers in `main.py` own request-failure logging.
- [ ] Config knobs in `app/config.py` Settings (not `os.getenv`); `compression_pdf_rebuild_enabled` defaults False.
- [ ] New security-bearing code lives OUTSIDE `pyproject.toml [tool.coverage.run].omit` (not in `upload.py`/`image_handler.py`/`pdf_handler.py`/`library_gemini.py`) so it counts toward the 75% gate.
- [ ] `mypy app/ --ignore-missing-imports` + `ruff` clean for all new modules each pass.
- [ ] Run `/security-review` on the diff every pass; re-run this checklist top-to-bottom before each commit.

---

I have all the anchors confirmed. Writing the section now.

## 13. Edge Cases & Robustness Catalog

This catalog enumerates every input/environment scenario the auto-compression + persistence feature must survive, the exact expected behavior, and the precise code that enforces it. Two surfaces are covered: **Genie OCR** (`app/api/v1/endpoints/upload.py` → `_save_original_file` → `get_storage_client().upload_bytes`) and **ERPHome attach** (`app/adapters/erpnext/_attachments.py` → ERPNext `/api/method/upload_file`). The overriding invariant across all rows is **CORRECTED DECISION #9**: *compression is best-effort and may never raise into, block, or alter the OCR/attach path.* On any compression anomaly, the **original bytes** are persisted/sent and a structured WARNING is logged.

### 13.1 Files to CREATE and MODIFY for robustness

**CREATE**

- `/Users/bhags/Desktop/erpsense_all/erpsense-backend/app/services/file_processing/compression.py` — the only place compression logic lives. Pure, side-effect-free, coverage-bearing (NOT in `pyproject.toml [tool.coverage.run].omit`). All edge-case branches below resolve here. Proposed surface:
  ```python
  # Strategy result — always returns; NEVER raises to the caller.
  @dataclass(frozen=True)
  class CompressionResult:
      data: bytes                 # compressed OR original (fallback) bytes
      original_size: int
      stored_size: int
      strategy: str               # "image" | "scanned_pdf" | "none"
      decision: str               # "compressed" | "skipped_native" | "skipped_min_size"
                                  #  | "skipped_too_small_px" | "fallback_regression"
                                  #  | "fallback_error" | "skipped_unsupported"
      fallback_reason: str | None

  # Tunables read from app/config.py Settings (NOT os.getenv — R03):
  #   compression_enabled (default True), compression_min_exempt_bytes (500*1024),
  #   compression_image_target_bytes (200*1024), compression_image_max_px (1200),
  #   compression_scanned_pdf_target_bytes_per_page (300*1024),
  #   compression_pdf_rebuild_enabled (default False, guarded — DECISION #7).

  def compress_content(
      content: bytes, filename: str, content_type: str | None
  ) -> CompressionResult:
      """Decide + compress on a COPY of `content`. Total try/except wrapper:
      any exception -> CompressionResult(data=content, decision='fallback_error')."""
  ```
- `/Users/bhags/Desktop/erpsense_all/erpsense-backend/app/services/file_processing/compression_metrics.py` — log-based metric helpers mirroring `app/services/tally_sync/diagnostic_metrics.py` (`record_compression_decision`, `record_compression_fallback`, `record_gcs_upload`). Event names `compression.decision`, `compression.fallback`, `gcs.upload`.
- `/Users/bhags/Desktop/erpsense_all/erpsense-backend/app/services/file_processing/_pdf_classifier.py` — shared `is_native_pdf(text, page_count, *, mode)` so the compression "scanned vs native" decision stays in lock-step with OCR's classifier (`PDFHandler._has_sufficient_text` at `pdf_handler.py:302-308` uses TOTAL `len(text) >= 50*page_count`; `library_gemini._extract_pdf_native` uses PER-PAGE AVERAGE). The helper is parameterized on `mode` so neither existing call site's semantics change.
- Tests: `tests/unit/file_processing/test_compression.py` (every row below as a unit case), `tests/api/test_upload_compression.py` (Genie persist-compressed, mirrors `tests/api/test_upload_cancel.py`), `tests/api/v1/test_attachment_compression.py` (ERPHome, respx).

**MODIFY**

- `/Users/bhags/Desktop/erpsense_all/erpsense-backend/app/api/v1/endpoints/upload.py`
  - `_save_original_file()` (`225-264`): change signature to accept pre-compressed bytes, replace the f-string log at line `259` with structured `logger.info("upload.original_saved", extra={...})` (no double logging, no banner — current style violates CLAUDE.md). Keep the **single** `upload_bytes` write at line `258` and the dup-filename suffixer at `249-254` (DECISION #3 — do not add a 2nd write).
  - `upload_file()` call site at `1024-1025`: compute `compress_content(content, original_filename, file.content_type)` **after** the read at `864` and **after** PDF page-count validation ends (`1019`), then pass `result.data` into `_save_original_file`. OCR at `1044-1047` still reads the untouched `temp_path` (DECISION #1/#2).
- `/Users/bhags/Desktop/erpsense_all/erpsense-backend/app/adapters/erpnext/_attachments.py`
  - `upload_attachment` (`77-168`) and `upload_file_standalone` (`170-233`): add `compress: bool = True`; when true, run `compress_content` on a COPY of `file_content` immediately **before** the multipart build (`files = {...}` at lines `108`/`196`); extend the existing `extra={...}` at `153`/`222` with `original_size`/`stored_size`/`compressed`.
- `/Users/bhags/Desktop/erpsense_all/erpsense-backend/app/services/erpnext_provisioning/_steps_company.py` line `335`: pass `compress=False` (logo already optimized by `BrandingService` — the one in-adapter caller that must opt out).
- `/Users/bhags/Desktop/erpsense_all/erpsense-backend/app/config.py`: add the `compression_*` Settings fields above.

### 13.2 Size-boundary edge cases (Genie + ERPHome)

| Scenario | Expected behavior | Enforcement point |
|---|---|---|
| **0-byte file** | Genie: `content = b""` (`upload.py:864`); `_save_original_file` early-returns `None` because the upstream supported-extension/empty checks short-circuit before persist. ERPHome: both adapter methods already guard `if not file_content` (`_attachments.py:102-106`) → returns `{"success": False, validation}` BEFORE compression. `compress_content(b"", ...)` is never reached on ERPHome; on Genie it must return `decision="skipped_min_size"`, `data=b""`. | `compress.py` guard `if not content: return as-is`; adapter `102-106` |
| **< 500 KB (MIN exempt)** | `decision="skipped_min_size"`, store/send original unchanged. DECISION #8: the 500 KB floor is **net-new**, lives ONLY in `compress_content` keyed off `compression_min_exempt_bytes`; it is NOT a rejection and has no FE representation. | `compress.py`: `if len(content) < settings.compression_min_exempt_bytes` |
| **Exactly 500 KB** | Boundary: `< floor` is exempt, so exactly 500 KB is **eligible** for compression (then routed by type). Unit test asserts the off-by-one explicitly. | `compress.py` |
| **Exactly 30 MB (Genie MAX)** | Accepted (`<=` MAX_FILE_SIZE at `upload.py:81`), compressed if image/scanned-PDF. FE already at 30 MB (`constants.ts:1893`) — DECISION #8 reconciliation is a **no-op** on FE; verify equality only. | `upload.py` size checks `760`/`919-946` |
| **> 30 MB (Genie)** | Rejected BEFORE read (`upload.py:760` Content-Length pre-check) or post-read (`919-946`) → 413. Compression never runs (short-circuit precedes line `1024`). | `upload.py:760, 919-946` |
| **Exactly 10 MB / > 10 MB (ERPHome)** | ERPHome cap is genuinely **10 MB** (`_shared.py:111 _MAX_ATTACHMENT_SIZE_BYTES`), NOT 30 MB. `doc_social.py:320` / `file_proxy.py:172` read `_MAX+1` then 413. Compression runs only for the in-range path. **Do NOT raise this to 30 MB** — that is the Genie surface only. | `_shared.py:111`; `doc_social.py:287-320`; `file_proxy.py:172` |
| **> 32 MB total session (Genie)** | Rejected at cumulative check (`upload.py:869-916`) → 413. Independent of per-file compression. | `upload.py:869-916` |

### 13.3 Image content edge cases (strategy `"image"`, target ≤200 KB / ≤1200 px)

All operate on a `BytesIO(content)` COPY — **never** `temp_path** (the `ImageHandler._resize_if_needed` in-place `img.save(file_path)` at `image_handler.py:162` is the canonical footgun this avoids).

| Scenario | Expected behavior |
|---|---|
| **Already-compressed / small image** | If already ≤ target after a single re-encode attempt OR re-encode ≥ original → `decision="fallback_regression"`, store original (DECISION #9). Never upscale, never multi-pass thrash. |
| **Image smaller than 1200 px** | No resize; attempt quality re-encode only. If that doesn't beat original, store original (`fallback_regression`). |
| **Huge dimensions (> 1200 px, e.g. 8000×6000)** | Downscale to longest-edge ≤ 1200 px (LANCZOS), JPEG/WebP encode toward ≤200 KB. Keep **same filename+extension** (DECISION #4 — serve content-type is suffix-derived at `upload.py` `MIME_TYPE_MAP`; FE `chat-utils.EXT_TO_MIME` mirrors). |
| **RGBA / palette (P) / LA mode** | Replicate `image_handler.py:159-161`: convert to RGB before JPEG encode on the COPY. If original is PNG with alpha that must be preserved, prefer WebP-with-alpha or keep PNG; if conversion would lose alpha AND output isn't smaller, fall back to original. |
| **CMYK image** | `img.convert("RGB")` on the copy before encode; never write CMYK JPEG. On conversion failure → `fallback_error`. |
| **Animated GIF / multi-frame** | Detect `getattr(img, "is_animated", False) or n_frames>1` → `decision="skipped_unsupported"`, store original (single-frame re-encode would destroy animation). |
| **Transparent PNG used as logo (provisioning)** | Never reaches compression: `_steps_company.py:335` passes `compress=False`. |
| **Corrupt / truncated image** | `Image.open` / `.load()` raises → caught by the module-level try → `decision="fallback_error"`, `fallback_reason="decode_failed"`, store original. WARNING logged. OCR still runs on the untouched `temp_path`. |
| **Image bomb (decompression bomb)** | `Pillow` raises `Image.DecompressionBombError` → same `fallback_error` path; original stored. Optionally set a `MAX_IMAGE_PIXELS` guard in `compress.py`, but failure-as-fallback already covers it. |
| **EXIF-rotated photo** | `ImageOps.exif_transpose` on the copy before resize so the stored preview matches OCR orientation; if EXIF parsing fails, proceed without transpose (non-fatal). |

### 13.4 PDF content edge cases (strategy `"scanned_pdf"`, target ≤300 KB/page; native PDF skipped)

PDF rebuild is **OPTIONAL and OFF by default** (`compression_pdf_rebuild_enabled=False`, DECISION #7). When off, every PDF row below resolves to `decision="skipped_native"` / store original. When on (guarded), rebuild uses **PyMuPDF (fitz) + Pillow + reportlab** only — `pikepdf`/`img2pdf` are NOT in requirements.

| Scenario | Expected behavior |
|---|---|
| **Native (text) PDF** | `is_native_pdf(...)` true → `decision="skipped_native"`, store original. Container rebuild does not apply (DECISION #7). |
| **Scanned PDF, rebuild flag OFF** | `decision="skipped_native"` (effectively), store original. No raster work performed. |
| **Scanned PDF, rebuild flag ON** | Open via fitz on a COPY (BytesIO/temp copy, never `temp_path`), rasterize at the OCR-consistent 200 DPI (`pdf_handler._pdf_to_images:310-326`), re-encode pages, rebuild. If rebuilt ≥ original/page → `fallback_regression`, store original. Same `.pdf` extension preserved. |
| **Password-protected / encrypted PDF** | fitz `needs_pass`/auth raises → `decision="fallback_error"`, `fallback_reason="encrypted_pdf"`, store original. OCR's own page-count validation (`upload.py:954-1019`) already handles its side independently. |
| **Corrupt / truncated PDF** | fitz open raises → `fallback_error`, store original. (Genie page-count block at `954-1019` may itself 400 first; if it does, compression is never reached — fine.) |
| **PDF > MAX_PAGES_PER_PDF (Genie 10)** | Rejected at `upload.py:954-1019` (400) before compression. Compression never sees it. |
| **Mixed native+scanned pages** | Classifier semantics differ by site (TOTAL vs AVERAGE) — `_pdf_classifier` preserves each. For the compression decision specifically, treat "any non-trivial extractable text" conservatively as native → skip, to avoid degrading a readable PDF. |

### 13.5 Filename / metadata edge cases

| Scenario | Expected behavior |
|---|---|
| **Unknown / no extension** | `mimetypes.guess_type` returns `None` (`upload.py:257`) → `application/octet-stream`. `compress_content` → `decision="skipped_unsupported"`, store original. Extension preserved verbatim. |
| **Non-ASCII filename (e.g. `请款单.png`)** | Persisted as-is after sanitize (`_save_original_file:241-244` strips only `/ \ \x00` and leading dots). Preview already `encodeURIComponent`s (`chat-utils.buildPreviewUrl`). Compression must NOT alter the name. |
| **Very long filename** | No truncation introduced by compression (don't change behavior). If GCS object-name limits bite, that is pre-existing and surfaces as a storage error → `fallback`/`None` return, non-blocking. |
| **Duplicate filename in session** | Suffixer at `_save_original_file:249-254` renames to `{stem}_{counter}{suffix}` — unchanged. Because compression keeps the same extension, the `DELETE /upload/{upload_id}` purge (`cancel_upload` + `_purge_upload_artifacts`, which matches `{stem}.{ext}` / `{stem}_{counter}.{ext}`) keeps working. |
| **Extension/content mismatch (e.g. `.png` that is actually a PDF)** | Classification is content-sniffed in `compress.py` (open with Pillow → if fails, try fitz), not extension-trusted. If neither opens → `skipped_unsupported`/`fallback_error`, store original. Extension still preserved (preview content-type stays suffix-derived — DECISION #4). |

### 13.6 Storage / environment edge cases (Genie → GCS)

`get_storage_client()` (`storage_client.py:399-418`) selects GCS in DEV/prod when a bucket is configured (DECISION #10), Local otherwise. All these resolve through `_save_original_file`'s existing try/except (`262-264`) which returns `None` and **does not** block OCR.

| Scenario | Expected behavior |
|---|---|
| **GCS down / network timeout** | `upload_bytes` raises → caught at `_save_original_file:262`, `logger.warning("upload.original_save_failed", extra={...})`, return `None`. OCR completes; `UploadResponse` still returned. Preview later 404s gracefully (FE `file-preview-modal` already handles unavailable). |
| **GCS permission-denied (objectCreator-only)** | IAM grants create-only (no overwrite/delete — `phase2/main.tf:35-38`). Re-upload to an existing path 403s; the suffixer's fresh path avoids this. A genuine 403 → same non-blocking `None` path + WARNING. |
| **Misconfigured GCS silently degrades to Local** | `get_storage_client` logs a WARNING and returns `LocalStorageClient` when no bucket set (`412-417`). Dev smoke must **assert the GCS client type**, not just "upload worked" (LocalStorageClient never fails). |
| **`record_gcs_upload` telemetry** | Emit `gcs.upload` with `{backend: type(storage).__name__, status, storage_path, stored_size, duration_ms}` at the `upload.py` caller seam — NOT inside `storage_client` (keep storage transport-only). |
| **compressed ≥ original** | `decision="fallback_regression"`; persist `content` (original). `metrics.record_compression_fallback(reason="regression")`. No retry. |

### 13.7 ERPHome attach edge cases (→ ERPNext only, NO GCS)

| Scenario | Expected behavior |
|---|---|
| **ERPNext upload failure (non-2xx / exception)** | Compression is independent and already done (or fell back) before the POST. `upload_attachment`/`upload_file_standalone` keep their existing try/except returning `{"success": False, error}` (`_attachments.py:157-168`, `170-233`). A compression exception must be caught INSIDE `compress_content` so it never bubbles into the multipart build. |
| **Compressed JPEG must stay `image/jpeg`** | After compression, pass the (possibly updated) `content_type` as the 3rd element of the multipart tuple (`files = {"file": (file_name, data, content_type)}`), NEVER as a request header (the methods deliberately strip Content-Type so httpx sets the boundary — `_attachments.py:130-135`, `196`). Keep filename+extension matching the content_type. |
| **Provisioning logo (in-adapter, out-of-scope)** | `compress=False` at `_steps_company.py:335` — already-optimized branding logo is not re-touched. Failures here are already swallowed/non-fatal. |
| **employee_expense / employee_travel loop reuse** | Both iterate service tuples and call `upload_attachment` (`employee_expense_service.py:617,651`; `employee_travel_service.py:750`). Adapter-level hook covers them with zero per-caller edits; compress a COPY so list-held byte references aren't mutated. |
| **Out-of-scope sinks untouched** | `import_files/upload.py`, `files.py:upload_logo`, `attachments.py` (FileStorageService) do NOT call the adapter → no compression hook, no behavior change. |

### 13.8 Concurrency, partiality, and async edge cases

| Scenario | Expected behavior |
|---|---|
| **Concurrent uploads, same session** | Each request has its own `content`, `temp_path`, and registry entry (`upload.py:851-855`). Compression is per-request CPU work via `asyncio.to_thread` (DECISION #6 — synchronous inline, no worker). Duplicate filenames across concurrent requests resolve via the suffixer; a rare race on `exists()` is tolerable (one gets `_1`). |
| **Parallel multi-file in one request** | "Parallel" = `asyncio.gather` over per-file `asyncio.to_thread(compress_content, ...)` within the request handler — no net-new infra. |
| **OCR timeout but compression succeeded** | OCR raises `asyncio.TimeoutError` at `wait_for(...,300)` (`upload.py:1044`) → request 408/500 via handler. BUT `_save_original_file` ran at `1024` BEFORE OCR, so the **compressed original is already persisted** and previewable. Compression success is independent of OCR outcome. |
| **Compression slow but OCR fast** | Persist (`1024`) precedes OCR (`1044`); compression runs synchronously before persist. To avoid compression latency dominating, `compress_content` has a soft internal bound (single pass, no iterative quality search) and is wrapped so it can never hang OCR — worst case it returns original quickly. |
| **Compression succeeds, OCR fails to extract (empty result)** | Unrelated: storage already holds the compressed original; OCR error handling unchanged. |
| **`temp_path` unlink in finally** | `finally` at `upload.py:1353-1363` unlinks `temp_path` regardless. Compression never touched `temp_path`, so cleanup semantics are unchanged. |
| **Fire-and-forget task leakage (test stability)** | Do NOT spawn detached tasks for compression; run inline so xdist `--dist=loadfile` workers don't leak (`tests-ci` gotcha). |

### 13.9 Logging / observability invariants for every edge case

- One structured event per outcome: `compression.decision` (INFO, every file), `compression.fallback` (WARNING, on `fallback_*`), `gcs.upload` (INFO/WARNING). `extra={filename, content_type, original_size, stored_size, saved_pct, strategy, decision, fallback_reason, storage_backend, duration_ms}`.
- **Never** an f-string message, **never** a banner (`"="*60`) — bring `upload.py:259` into compliance, not add to the violation.
- **Never** pass `session_id`/`tenant_id`/`user_id`/`request_id` in `extra` — auto-injected by `JSONFormatter` via `get_log_context`.
- **No double logging**: fallback is WARNING (recoverable), never `logger.error` before a raise — and compression never raises.
- Log filenames/sizes only — never file bytes, GSTIN, tokens, PII (auto-redacted, but don't rely on it).

### 13.x Checklist

- [ ] Create `app/services/file_processing/compression.py` with `CompressionResult` + `compress_content` wrapped in a total try/except that always returns (never raises).
- [ ] Create `app/services/file_processing/_pdf_classifier.py::is_native_pdf(text, page_count, *, mode)` and delegate `pdf_handler._has_sufficient_text` (302-308) and `library_gemini._extract_pdf_native` to it WITHOUT changing either's semantics or raise-vs-branch control flow.
- [ ] Create `app/services/file_processing/compression_metrics.py` (events `compression.decision`, `compression.fallback`, `gcs.upload`) mirroring `diagnostic_metrics.py`.
- [ ] Add `compression_*` Settings to `app/config.py` (enabled, min_exempt=512000, image_target=204800, image_max_px=1200, scanned_pdf_target_per_page=307200, pdf_rebuild_enabled=False); NO `os.getenv` outside config.
- [ ] Modify `upload.py:_save_original_file` to take compressed bytes, keep the single `upload_bytes` write (258), replace f-string log (259) with `logger.info("upload.original_saved", extra={...})`.
- [ ] Wire `compress_content` at `upload.py:1024-1025` (after read 864, after page validation 1019); pass `result.data`; leave OCR (1044) reading `temp_path`.
- [ ] Add `compress: bool = True` to `_attachments.py` `upload_attachment` (108) and `upload_file_standalone` (196); compress a COPY of `file_content` before the multipart build; extend `extra` at 153/222.
- [ ] Pass `compress=False` from `_steps_company.py:335` (provisioning logo opt-out).
- [ ] 0-byte: Genie skip + ERPHome existing `102-106` guard, never reach compression with empty bytes.
- [ ] `< 500 KB` exempt; **exactly 500 KB eligible** (off-by-one unit test).
- [ ] Exactly 30 MB accepted (Genie); `> 30 MB` 413 before compression; assert FE `constants.ts:1893` already 30/32/5/10 (no FE downgrade).
- [ ] ERPHome cap stays 10 MB (`_shared.py:111`); do NOT raise to 30 MB.
- [ ] Image: huge-dimension downscale ≤1200 px (LANCZOS) toward ≤200 KB; preserve filename+extension.
- [ ] Image: RGBA/P/LA → RGB before JPEG (mirror `image_handler.py:159-161`); CMYK → RGB; animated GIF → `skipped_unsupported`.
- [ ] Image: `exif_transpose`; decompression-bomb / corrupt / truncated → `fallback_error`, store original.
- [ ] Compress strictly on a COPY (BytesIO) — never `temp_path` (guard against the `image_handler.py:162` in-place footgun).
- [ ] PDF rebuild OFF by default → native + scanned both `skipped_native`/store original.
- [ ] PDF rebuild ON (guarded): fitz on a COPY at 200 DPI; encrypted/corrupt → `fallback_error`; keep `.pdf` extension.
- [ ] PDF page-limit/page-count rejection (954-1019) precedes compression — verify no compression on rejected PDFs.
- [ ] `compressed >= original` anywhere → `fallback_regression`, persist/send original, no retry, WARNING.
- [ ] Unknown/no extension, extension/content mismatch → content-sniff (Pillow then fitz); skip/fallback; extension preserved.
- [ ] Non-ASCII + very long filenames pass through unchanged; duplicate-filename suffixer (249-254) intact; verify `_purge_upload_artifacts` cancel still matches.
- [ ] GCS down/timeout/permission-denied → `_save_original_file` returns `None` non-blocking, OCR completes, WARNING logged.
- [ ] Misconfigured GCS degrades to Local with WARNING; dev smoke asserts `type(get_storage_client()).__name__ == "GCSStorageClient"`.
- [ ] ERPNext upload failure keeps `{"success": False}` envelope; compression exception never bubbles into multipart build.
- [ ] Compressed JPEG keeps `image/jpeg` content_type in the multipart tuple 3rd element (never as a header); filename+extension match.
- [ ] OCR timeout after persist → compressed original already stored + previewable; compression independent of OCR result.
- [ ] Concurrent / multi-file uploads: per-request `content`/`temp_path`; `asyncio.gather` + `asyncio.to_thread`, no detached tasks, no `temp_path` mutation.
- [ ] Every outcome emits exactly one structured event; no f-strings/banners; no request-scoped ids in `extra`; no PII/bytes logged; fallback at WARNING (no double-logging).
- [ ] Tests: `tests/unit/file_processing/test_compression.py` (all rows incl. corrupt/animated/CMYK/regression/min-exempt boundary), `tests/api/test_upload_compression.py` (persist-compressed + GCS-failure monkeypatch fallback, mirror `test_upload_cancel.py`), `tests/api/v1/test_attachment_compression.py` (respx ERPNext + `compress=False` provisioning case) — keep new code out of `pyproject.toml omit`, target ≥85%.

---

I have enough confirmed detail to write a precise, deeply-grounded test strategy section. I have verified: pytest config + the coverage `omit` list (upload.py, all OCR handlers, pdf/image handlers are omitted — so compression logic must live in a NEW non-omitted module), conftest GCS pinning at lines 209-213, the upload_cancel test template, the BRS GCS LocalStorageClient + monkeypatch-failure idiom, the storage_client unit test, the ERPNext adapter patch pattern (`_get_erpnext_adapter` / `monkeypatch.setattr(WorkerExpenseService, ...)`), respx usage, and FE upload.test.ts + MSW server.ts + validate-upload-files.test.ts.

## 14. Testing Strategy & Test Files (unit/integration/e2e/perf)

### 14.1 Guiding constraints (verified against freshly-pulled code)

These three CI realities dictate *where* every test and every line of production code must live, and are the single most important inputs to this strategy:

1. **The Genie OCR surface is coverage-exempt.** `pyproject.toml [tool.coverage.run].omit` (lines 109-145) excludes `app/api/v1/endpoints/upload.py`, `app/services/file_analysis.py`, and **every** handler/engine: `handlers/pdf_handler.py`, `handlers/image_handler.py`, `handlers/docx_handler.py`, `handlers/excel_handler.py`, `handlers/text_handler.py`, and all `ocr/*` incl. `library_gemini.py`. **Consequence:** any compression code placed in those files contributes **zero** to the 75 % gate. Therefore the compression engine MUST be a NEW non-omitted module: **`app/services/file_processing/compression.py`** (+ `compression_metrics.py`, `_pdf_classifier.py`). `upload.py` keeps only the thin wiring at the `_save_original_file` seam (upload.py:1024-1025) — and that wiring is validated by `tests/api/` endpoint tests (which exercise behavior, not line coverage).
2. **There is no live-GCS test path.** `tests/conftest.py:209-213` does `os.environ.setdefault("GCS_BUCKET_NAME", "")` / `GCS_UPLOAD_BUCKET=""`, forcing `get_storage_client()` (storage_client.py:399-418) to always return `LocalStorageClient`. The GCS code path is validated two ways only: (a) drive the prod-shaped `upload_bytes`/`download_bytes` contract through `LocalStorageClient` via `get_storage_client()` (tmp-backed), and (b) monkeypatch `storage.upload_bytes` to raise and assert fallback. Real GCS in DEV is a **manual smoke check (§14.9)**, not unit CI.
3. **Test conventions:** `asyncio_mode=auto` → write `async def test_*` with **no** `@pytest.mark.asyncio`; `-n 4 --dist=loadfile`, 60 s per-test timeout; `tests/unit/` is a strict pure-logic lane (its `conftest.py` warns if a unit test requests `db_session`/`client`); mock only boundaries (respx for HTTP, fakeredis for Redis), never internal services. Coverage gate is **75 %** on the merged dataset (deploy.yml:111-253); new code targets ≥85 %.

### 14.2 Files to CREATE / MODIFY (test artifacts)

**Backend — fixtures (shared sample corpus):**

- CREATE `tests/fixtures/files/__init__.py` — byte-builders (NOT checked-in binaries, so the repo stays lean and the corpus is deterministic). Mirror the `_png()` helper in `tests/api/v1/test_file_attachments.py:52-113`. Signatures:
  ```python
  def png_bytes(w=1600, h=1600, color=(180,40,40)) -> bytes      # large image -> compress
  def png_small(w=64, h=64) -> bytes                              # tiny -> exempt
  def jpeg_bytes(w=2000, h=2000, quality=95) -> bytes
  def png_with_alpha(w=1300, h=1300) -> bytes                    # RGBA -> exercises RGBA/P->RGB path
  def native_pdf_bytes(pages=3, chars_per_page=400) -> bytes      # PyMuPDF text page -> native, skip
  def scanned_pdf_bytes(pages=2, dpi=200) -> bytes               # image-only pages -> scanned, compress
  def docx_bytes(paragraphs=20) -> bytes                          # native -> skip
  def xlsx_bytes(rows=50) -> bytes                                # native -> skip
  def csv_bytes(rows=100) -> bytes                                # native -> skip
  def txt_bytes(n=2000) -> bytes
  def oversize_png(target_mb=31) -> bytes                         # > MAX_FILE_SIZE (30MB) -> reject
  def decompression_bomb_png() -> bytes                           # Pillow MAX_IMAGE_PIXELS trip
  def zip_bomb_disguised_as_pdf() -> bytes                        # nested-zip bytes w/ .pdf name
  ```
- CREATE `tests/conftest.py` additions (or a `tests/fixtures/files/conftest.py`) exposing each as a pytest fixture so both unit and api lanes reuse them.

**Backend — unit tests (coverage-bearing, beside existing peers):**

- CREATE `tests/unit/file_processing/test_compression.py` — the core engine (`app/services/file_processing/compression.py`).
- CREATE `tests/unit/file_processing/test_pdf_classifier.py` — the shared `is_native_pdf` helper (`_pdf_classifier.py`) — must prove BOTH existing semantics are preserved (see §14.4).
- CREATE `tests/unit/file_processing/test_compression_metrics.py` — log-based metric emit (`compression_metrics.py`), mirroring the `diagnostic_metrics.py` idiom.
- MODIFY `tests/unit/core/test_storage_client.py` — add content-type-from-suffix + round-trip-after-compression assertions (already covers traversal/serve at lines 68-84).
- MODIFY `pyproject.toml` — **do NOT** add `compression.py`/`compression_metrics.py`/`_pdf_classifier.py` to `[tool.coverage.run].omit`; if a `MetricsRegistry._init_compression_metrics()` is added, ensure `metrics.py` is not omitted (it isn't).

**Backend — api/integration tests (behavior, not coverage):**

- CREATE `tests/api/test_upload_compression.py` — Genie OCR upload end-to-end persist-compressed assertions (template: `tests/api/test_upload_cancel.py`).
- CREATE `tests/api/v1/test_erphome_attach_compression.py` — ERPHome adapter-hook behavior across `doc_social.upload_attachment` + `file_proxy.upload_erp_file` (template: adapter-mock pattern from `tests/api/test_documents.py:55-68` + `monkeypatch.setattr(WorkerExpenseService, ...)` from `tests/api/v1/test_employee_expense_routes.py`).
- CREATE `tests/integration/test_compression_storage_roundtrip.py` — full compress→persist→serve loop via `get_storage_client()` (LocalStorageClient, tmp-backed) proving the "GCS path" contract.

**Frontend:**

- MODIFY `tests/unit/lib/api/upload.test.ts` — assert 30 MB constant agreement + (if added) compression-metadata pass-through.
- MODIFY `tests/unit/lib/utils/validate-upload-files.test.ts` — client-side size/extension limits (file already exists).
- CREATE `tests/unit/lib/utils/compression-copy.test.ts` (only if compression-transparency copy is added to a **non-excluded** `lib/utils` module — components dirs are coverage-excluded per jest.config.js).
- MODIFY `tests/mocks/handlers.ts` + `tests/mocks/server.ts` — MSW stubs returning the new optional `original_size`/`stored_size`/`was_compressed` fields on the upload response.
- CREATE `tests/e2e/specs/upload-compression-transparency.spec.ts` (+ steps/mocks) — Playwright transparency flow (**non-gating**; documentation/local only, see §14.7).

### 14.3 Unit tests — compression engine (`test_compression.py`)

Target the new `compression.py` public surface. Expected signature (decided in the implementation plan; restated for the test contract):
```python
async def compress_for_storage(content: bytes, filename: str, content_type: str | None) -> CompressionResult
# CompressionResult: data: bytes, strategy: str, original_size: int, stored_size: int,
#                    was_compressed: bool, fallback_reason: str | None
```
Invariant under test everywhere: **the input `content` bytes object is never mutated** (the ImageHandler in-place footgun, image_handler.py:162, must not be replicated) and **the filename/extension is never changed** (serve content-type is suffix-derived, upload.py:1985-1988).

- [ ] `test_large_image_compressed_under_threshold` — `png_bytes(1600,1600)` → `was_compressed=True`, `stored_size <= 200*1024`, longest side ≤1200 px (Pillow reopen to assert).
- [ ] `test_image_keeps_extension_and_format` — `.jpg` in → `.jpg` out, `image/jpeg`; `.png` → `.png`. Strategy label recorded.
- [ ] `test_rgba_and_palette_images_converted_to_rgb_for_jpeg` — `png_with_alpha` re-encoded to JPEG stays valid (mirrors image_handler.py:159-161 conversion).
- [ ] `test_tiny_image_exempt_min_500kb` — `png_small` < 500 KB → `was_compressed=False`, `fallback_reason="below_min_size"`, bytes returned **identical** to input (Decision 8 MIN exemption).
- [ ] `test_input_bytes_not_mutated` — assert `content == original_copy` after call (no in-place mutation).
- [ ] `test_scanned_pdf_compressed_per_page_budget` — `scanned_pdf_bytes(2)` → `was_compressed=True`, `stored_size <= 300*1024 * page_count`, output still opens in PyMuPDF with the same page count.
- [ ] `test_native_pdf_passthrough` — `native_pdf_bytes(3,400)` → `was_compressed=False`, identical bytes, `strategy="native_pdf_skip"`.
- [ ] `test_docx_xlsx_csv_txt_passthrough` — each native format returned unchanged (Decision 7 scope: only images + scanned PDFs compress).
- [ ] `test_compression_regression_falls_back_to_original` — when compressed ≥ original (e.g. already-optimized JPEG), return original bytes, `was_compressed=False`, `fallback_reason="no_size_gain"` (Decision 9).
- [ ] `test_corrupt_image_falls_back_no_raise` — garbage bytes with `.png` name → returns original, `fallback_reason="compression_error"`, **does not raise** (preserves `_save_original_file` non-blocking contract).
- [ ] `test_corrupt_pdf_falls_back_no_raise` — same for PDF path.
- [ ] `test_optional_container_rebuild_flag_off_by_default` — DOCX/XLSX rebuild stays disabled unless the Settings flag is set (Decision 7 guarded/off-by-default); flip via monkeypatched `settings`.
- [ ] `test_parallel_multi_page_uses_to_thread_gather` — patch `asyncio.to_thread` spy; assert scanned-PDF pages compressed concurrently (Decision 6: `asyncio.gather` + `to_thread`, no event-loop blocking).

### 14.4 Unit tests — shared PDF classifier (`test_pdf_classifier.py`)

The new `_pdf_classifier.py` is delegated to from BOTH existing call sites; the test must **prove neither site's behavior changes** (the two have *different* semantics per the seam gotchas):

- [ ] `test_total_chars_semantics_matches_pdf_handler` — `is_native_pdf(text, page_count, mode="total")` reproduces `PDFHandler._has_sufficient_text` (`len(text) >= 50*page_count`, pdf_handler.py:302-308), including the mixed-density case (1 empty + 1 dense page → native).
- [ ] `test_avg_chars_semantics_matches_library_gemini` — `mode="avg"` reproduces `library_gemini._extract_pdf_native` (`total/page_count < 50`, library_gemini.py:613-617), including the *same mixed-density case → scanned* (proves the two modes diverge as designed).
- [ ] `test_zero_pages_guarded` — `page_count=0` does not ZeroDivisionError.
- [ ] `test_thresholds_unchanged` — assert `MIN_TEXT_LENGTH`/`MIN_PDF_TEXT_PER_PAGE` still 50; helper does not redefine them.
- [ ] `test_compression_routing_uses_helper` — compression's image-vs-scanned-vs-native decision calls the same helper (no third, drifting classifier).

> Regression guard for the refactor: add a delegation test that imports `PDFHandler` and `LibraryGeminiProcessor` and asserts each still returns its original verdict on the corpus — even though those files are coverage-omitted, the **assertions** still run and would fail CI on a behavior change.

### 14.5 API tests — Genie OCR persist-compressed (`test_upload_compression.py`)

Template: `tests/api/test_upload_cancel.py` (monkeypatch `file_router.process`, `app.dependency_overrides` for `get_feature_service`/`get_usage_service`, `reset_upload_registry_for_tests` autouse, POST `/api/v1/upload?session_id=&upload_id=` with `files={"file": (name, BytesIO, ctype)}`, assert via `get_storage_client().download_bytes("sessions/{sid}/originals/{filename}")`).

The crucial assertion split: **OCR receives the ORIGINAL temp bytes; storage receives the COMPRESSED bytes** (Decisions 1-4). Capture what OCR saw by having `fake_process` read `temp_path` and stash its bytes:
```python
seen = {}
async def fake_process(path, *_a, **_kw):
    seen["ocr_bytes"] = Path(path).read_bytes()
    return _fake_extraction()
monkeypatch.setattr(file_router, "process", fake_process)
```

- [ ] `test_large_image_stored_compressed_ocr_sees_original` — upload `png_bytes(1600,1600)`; `seen["ocr_bytes"] == original`; `download_bytes(originals)` is smaller than original; same filename/extension.
- [ ] `test_ocr_output_unchanged_vs_baseline` — extraction text returned to the client is byte-identical whether or not compression ran (compression must not touch the OCR result). **This is the "assert OCR output UNCHANGED" gate.**
- [ ] `test_native_pdf_stored_as_is` — native PDF persisted unchanged (`download_bytes == upload bytes`).
- [ ] `test_tiny_image_stored_as_is_min_exempt` — < 500 KB persisted unchanged.
- [ ] `test_compression_failure_persists_original_no_500` — monkeypatch `compression.compress_for_storage` to raise → endpoint returns 200, original persisted, fallback logged (caplog event `compression.fallback`).
- [ ] `test_no_session_id_skips_persist_no_compression_attempted` — when `session_id` absent, `_save_original_file` not called; compression skipped; OCR still runs.
- [ ] `test_cancel_purge_still_matches_compressed_file` — re-run the `test_cancel_after_completed_upload_purges_artifacts` assertions against a compressed upload to prove the dup-filename purge matcher still finds it (same filename → no regression).
- [ ] `test_serve_compressed_file_content_type_suffix_derived` — GET `/upload/sessions/{sid}/originals/{filename}` returns 200 with the suffix-derived `Content-Type` (no rename broke it).
- [ ] `test_oversize_rejected_before_compression` — `oversize_png(31)` → 413/400 from the pre-existing size check (upload.py:760/919) **before** any compression call (assert compress spy not called).
- [ ] `test_decompression_bomb_does_not_oom` — `decompression_bomb_png` → compression catches/falls back; assert no unbounded allocation (Pillow `MAX_IMAGE_PIXELS` honored).
- [ ] `test_existing_upload_happy_path_regression` — plain `.txt` upload still returns the same `UploadResponse` shape (no breakage to non-image/non-scanned flow).

### 14.6 API tests — ERPHome attach (`test_erphome_attach_compression.py`)

Per Decision 5, ERPHome persists to **ERPNext only** (no GCS). Mock the ERPNext adapter (pattern: `tests/api/test_documents.py:55-68` patch of `_get_erpnext_adapter`, or respx on `{BASE_URL}/api/method/upload_file`). Seed `authenticated_user_with_tenant` + `test_erp_connection`.

- [ ] `test_doc_attachment_image_compressed_before_post` — POST `/erp/{doctype}/{name}/attachment` with a large image; capture the bytes the adapter's `upload_attachment` received (`mock_adapter.upload_attachment` AsyncMock) → smaller than uploaded, same filename/`content_type`.
- [ ] `test_standalone_file_upload_image_compressed` — POST `/erp/files/upload` (`file_proxy.upload_erp_file` → `upload_file_standalone`) compresses likewise.
- [ ] `test_native_pdf_attachment_passthrough` — adapter receives unchanged bytes.
- [ ] `test_provisioning_logo_opt_out_not_compressed` — call path with `compress=False` (the `_steps_company.py:335` branding logo) → adapter receives the **original** logo bytes (the one in-adapter call that must be excluded).
- [ ] `test_attach_compression_failure_sends_original` — compression raises → adapter still POSTs original bytes; success envelope returned (the adapter try/except must not surface the error).
- [ ] `test_attach_size_cap_still_10mb` — file > `_MAX_ATTACHMENT_SIZE_BYTES` (10 MB, `_shared.py:111`) → 413; cap is **not** silently raised to 30 MB (the 30 MB is Genie-only).
- [ ] `test_cross_tenant_attach_returns_404` — mandatory BOLA test, assert `== 404` strictly (mirrors `tests/api/v1/test_file_attachments.py`).
- [ ] `test_employee_expense_receipt_compressed` — `WorkerExpenseService.upload_attachments` route compresses (adapter-level hook covers it; monkeypatch the service or adapter and assert compressed bytes reached `erp.upload_attachment`).
- [ ] `test_no_ocr_triggered_on_erphome_path` — assert `file_router.process` / `library_gemini` are never imported/called on this surface.

### 14.7 Integration + storage-contract tests

`tests/integration/test_compression_storage_roundtrip.py` and the additions to `tests/unit/core/test_storage_client.py`:

- [ ] `test_compress_then_persist_then_serve_roundtrip` — `compress_for_storage` → `get_storage_client().upload_bytes(path, data, content_type)` → `get_serve_response` returns the compressed bytes with suffix-derived media type and `Content-Disposition: inline` (storage_client.py:378-396 contract).
- [ ] `test_storage_upload_failure_bubbles_as_none_fallback` — monkeypatch `storage.upload_bytes` to raise `OSError` (mirror `test_brs_gcs_upload.py:148-160`); `_save_original_file` returns `None`, OCR still proceeds (non-blocking contract preserved).
- [ ] `test_content_type_guessed_from_suffix` — `upload_bytes` content-type matches `mimetypes.guess_type` for `.jpg/.png/.pdf` (preview correctness).
- [ ] `test_objectcreator_no_overwrite_simulated` — second upload to an existing path is renamed by the dup suffixer (no overwrite), proving prod IAM (objectCreator-only) won't 403 (storage seam gotcha).
- [ ] `test_local_client_selected_under_test_env` — assert `type(get_storage_client()).__name__ == "LocalStorageClient"` (guards the "GCS misconfig degrades silently to local" trap; confirms the test harness is actually exercising the intended client).

### 14.8 Security tests

Placed in `tests/unit/file_processing/test_compression.py` (engine-level) and `tests/api/test_upload_compression.py` (endpoint-level):

- [ ] `test_decompression_bomb_image_rejected_or_fallback` — Pillow `Image.MAX_IMAGE_PIXELS` enforced; a pixel-bomb PNG triggers fallback-to-original, never OOM (engine sets/keeps a sane `MAX_IMAGE_PIXELS`).
- [ ] `test_zip_bomb_disguised_as_pdf_does_not_explode` — nested-zip bytes named `*.pdf` → PyMuPDF open fails fast → fallback, no recursive extraction.
- [ ] `test_polyglot_extension_mismatch` — content-type says image but bytes are a PDF (and vice versa) → classifier trusts magic bytes / safe fallback, never crashes.
- [ ] `test_no_pii_or_bytes_in_logs` — `caplog` asserts `compression.decision`/`compression.fallback` events carry only `filename, content_type, original_size_bytes, stored_size_bytes, saved_pct, strategy, fallback_reason` — **never** raw bytes, tokens, GSTIN (redaction-by-default still verified, logging seam).
- [ ] `test_filename_with_traversal_is_sanitized_pre_persist` — `../../etc/passwd.png` is sanitized by `_save_original_file` (upload.py:241-244) before storage; compression preserves the *sanitized* extension.
- [ ] `test_oversize_not_loaded_unboundedly` — oversize input rejected by the size guard before bytes are passed to Pillow/PyMuPDF (no unbounded in-memory decode).

### 14.9 DEV GCS integration smoke (manual / non-CI)

Because conftest pins GCS off, the "works in DEV with GCS" requirement (Decision 10) is a **documented manual smoke**, not unit CI. Add a guarded, opt-in test marked `external` + skipped unless env is set:

- CREATE `tests/integration/test_gcs_dev_smoke.py` with `pytestmark = pytest.mark.skipif(not os.getenv("RUN_GCS_SMOKE"), reason="dev-only")` and `@pytest.mark.external`.
- [ ] `test_dev_gcs_upload_compressed_and_download_roundtrip` — when `RUN_GCS_SMOKE=1` and `GCS_UPLOAD_BUCKET=erpsense-backend-uploads-dev` + ADC present (`gcloud auth application-default login`), upload a compressed image, `download_bytes`, assert round-trip and that `get_storage_client()` is `GCSStorageClient`.
- [ ] Document the runbook in the PR: `RUN_GCS_SMOKE=1 GCS_UPLOAD_BUCKET=erpsense-backend-uploads-dev pytest tests/integration/test_gcs_dev_smoke.py -m external`. Note the env mismatch caveat: local `.env` `GCS_BUCKET_NAME=erpsense-ocr-dev` is **not** Terraform-provisioned — prefer `erpsense-backend-uploads-dev`.

### 14.10 Frontend tests (Jest + MSW; Playwright non-gating)

Coverage-placement rule (jest.config.js:16-138): logic that must count toward coverage lives in non-excluded `lib/api`/`lib/utils`. Components in `components/chat/**`, `components/erp/**`, `lib/hooks/**` are excluded → transparency UI is validated by Playwright (non-gating) + RTL where feasible, not by the coverage gate.

- [ ] `upload.test.ts` — `test_uploadFile_timeout_300000_unchanged` and `test_uploadFile_threads_session_and_upload_id` regression (extend existing cases 30-128).
- [ ] `upload.test.ts` — `test_upload_response_passes_through_compression_metadata` — when BE returns `original_size`/`stored_size`/`was_compressed`, the typed `UploadResponse` surfaces them (after `types/chat.ts:442-464` extension).
- [ ] `validate-upload-files.test.ts` — `test_30mb_per_file_32mb_total_5_files_10_pages` confirming FE constants (`constants.ts:1893-1896`) agree with BE (`upload.py:81-84`); explicitly assert **no 10 MB** appears on the Genie path (the prompt's "reconcile to 30MB" is already satisfied here — guard against accidental down-edit).
- [ ] `validate-upload-files.test.ts` — `test_below_500kb_is_accepted` (MIN-exempt has no FE rejection; ensure no new FE rule blocks small files).
- [ ] `tests/mocks/handlers.ts` — add a `POST /api/v1/upload` handler returning compression metadata; wire any new handler module into `tests/mocks/server.ts`.
- [ ] `upload-compression-transparency.spec.ts` (Playwright, **non-gating**) — upload a large image, assert chat `file-preview-modal.tsx` header shows "compressed from X" copy (i18n key under `chat:file_preview`), preview `<img>` loads via `useAuthedFileUrl` blob URL (no raw `<img src>` to protected path), and the same filename/extension renders.
- [ ] Confirm any new chat-input copy is **i18n'd** (existing size-error strings in `chat-input.tsx:214/247/271/298` are hardcoded English — do not add more raw strings; a reviewer will flag it).

### 14.11 Performance / latency-budget test

- CREATE `tests/perf/test_compression_latency.py` (lives under `tests/perf/`, which is `--ignore`'d in default `addopts` and only run explicitly — does NOT gate, matches the `perf` marker convention). Mark `@pytest.mark.perf`.
- [ ] `test_single_image_compression_under_budget` — `png_bytes(1600,1600)` compresses in < 750 ms wall-clock (asserts the synchronous-inline cost stays within a sane fraction of the 300 s OCR timeout, `_OCR_PROCESSING_TIMEOUT`).
- [ ] `test_scanned_pdf_10_pages_parallel_faster_than_serial` — assert `asyncio.gather`+`to_thread` path beats a serial baseline by a meaningful margin (proves Decision 6 parallelism is real, not nominal).
- [ ] `test_compression_adds_no_event_loop_blocking` — assert CPU work runs in `to_thread` (spy), so the request loop is not blocked during compression.
- [ ] Document: run with `pytest tests/perf/ -m perf` locally; budgets are advisory, not CI-gating.

### 14.12 Per-pass verification commands (run every iteration)

- [ ] Backend fast loop: `cd erpsense-backend && ruff format --check app/ tests/ && ruff check app/ tests/ && mypy app/ --ignore-missing-imports && pytest tests/unit/file_processing/ -p no:warnings`
- [ ] Backend full gate: `pytest tests/ --cov=app --cov-append --cov-report=term-missing --cov-fail-under=75 -q` (or `scripts/run_ci_locally.sh`)
- [ ] Alembic single-head check stays green (`alembic heads` == 1) — no migration is expected for this feature; if one is added it must keep a single head.
- [ ] Frontend: `cd erpsense-frontend && npm run format && npm run lint && npm run type-check && npm test -- tests/unit/lib/api/upload.test.ts tests/unit/lib/utils/validate-upload-files.test.ts`
- [ ] Frontend full gate: `npm run test:coverage -- --testPathIgnorePatterns='/integration/'` (lines/statements ≥80, branches/functions ≥75; new files ≥85).

### 14.13 "Existing behavior stays unbroken" guards (explicit checklist)

- [ ] `test_upload_cancel.py` suite still passes unchanged (compression keeps the same filename → purge matcher unaffected).
- [ ] `test_storage_client.py` traversal + serve tests still pass (storage layer untouched; telemetry added at the caller, not inside `storage_client.py`).
- [ ] OCR extraction output is byte-identical pre/post-feature (`test_ocr_output_unchanged_vs_baseline`).
- [ ] PDF native/scanned routing verdict unchanged at both classifier call sites (`test_pdf_classifier.py` delegation guards).
- [ ] ERPHome 10 MB cap and non-OCR behavior unchanged (`test_attach_size_cap_still_10mb`, `test_no_ocr_triggered_on_erphome_path`).
- [ ] FE Genie 30 MB/32 MB/5/10 limits unchanged; no accidental 10 MB introduction (`validate-upload-files.test.ts`).
- [ ] Provisioning branding-logo push is excluded from compression (`test_provisioning_logo_opt_out_not_compressed`).

### 14.x Checklist

- [ ] CREATE `tests/fixtures/files/__init__.py` with all byte-builders (png/jpeg/rgba/native-pdf/scanned-pdf/docx/xlsx/csv/txt/oversize/decompression-bomb/zip-bomb) + register as fixtures
- [ ] CREATE `tests/unit/file_processing/test_compression.py` (engine: large-image, jpeg/png extension, RGBA→RGB, tiny-exempt, no-mutation, scanned-pdf budget, native-pdf passthrough, docx/xlsx/csv/txt passthrough, size-regression fallback, corrupt-image/pdf fallback-no-raise, rebuild-flag-off, parallel to_thread/gather)
- [ ] CREATE `tests/unit/file_processing/test_pdf_classifier.py` (total vs avg semantics parity, zero-page guard, thresholds unchanged, compression-routing reuse, delegation regression for PDFHandler + LibraryGeminiProcessor)
- [ ] CREATE `tests/unit/file_processing/test_compression_metrics.py` (event-name + extra fields, no raw bytes/PII, mirrors `diagnostic_metrics.py`)
- [ ] MODIFY `tests/unit/core/test_storage_client.py` (content-type-from-suffix, round-trip-after-compression)
- [ ] CREATE `tests/api/test_upload_compression.py` (OCR-sees-original/storage-sees-compressed, OCR-output-unchanged, native-pdf as-is, tiny-exempt, failure→original no-500, no-session skip, cancel-purge matches compressed, serve content-type, oversize-pre-compression, decompression-bomb, txt happy-path regression)
- [ ] CREATE `tests/api/v1/test_erphome_attach_compression.py` (doc image compressed, standalone compressed, native passthrough, provisioning logo opt-out, failure→original, 10MB cap, cross-tenant 404 strict, expense receipt compressed, no-OCR)
- [ ] CREATE `tests/integration/test_compression_storage_roundtrip.py` (compress→persist→serve, upload failure→None fallback, content-type guess, no-overwrite dup-suffix, LocalStorageClient selected)
- [ ] CREATE `tests/integration/test_gcs_dev_smoke.py` (guarded `RUN_GCS_SMOKE`, `@pytest.mark.external`, GCSStorageClient selected + round-trip)
- [ ] CREATE `tests/perf/test_compression_latency.py` (`@pytest.mark.perf`: single-image budget, 10-page parallel vs serial, no event-loop blocking)
- [ ] Add security cases (decompression bomb, zip-bomb-as-pdf, polyglot mismatch, no-PII-in-logs, traversal-sanitized, oversize-not-loaded-unbounded) across engine + endpoint test files
- [ ] Ensure `compression.py` / `compression_metrics.py` / `_pdf_classifier.py` are NOT added to `pyproject.toml [tool.coverage.run].omit`
- [ ] Verify NO compression code lands in coverage-omitted files (`upload.py`, `handlers/*`, `ocr/*`, `file_analysis.py`)
- [ ] MODIFY `erpsense-frontend/tests/unit/lib/api/upload.test.ts` (300s timeout regression, session/upload_id threading, compression-metadata pass-through)
- [ ] MODIFY `erpsense-frontend/tests/unit/lib/utils/validate-upload-files.test.ts` (30/32/5/10 agreement, no 10MB on Genie path, below-500KB accepted)
- [ ] MODIFY `erpsense-frontend/tests/mocks/handlers.ts` + wire into `tests/mocks/server.ts` (upload response with compression metadata)
- [ ] CREATE `erpsense-frontend/tests/e2e/specs/upload-compression-transparency.spec.ts` (+ steps/mocks) — non-gating Playwright transparency flow with blob-URL preview + i18n copy
- [ ] Confirm all new FE user-facing copy is i18n'd (no new hardcoded strings near `chat-input.tsx`)
- [ ] Run backend fast loop (`ruff` + `mypy app/ --ignore-missing-imports` + `pytest tests/unit/file_processing/`) green
- [ ] Run backend full gate `pytest tests/ --cov=app --cov-fail-under=75` green; new modules ≥85%
- [ ] Confirm `alembic heads` == 1 (no new migration, or single head if added)
- [ ] Run FE `format` + `lint` + `type-check` + targeted unit tests green
- [ ] Run FE `npm run test:coverage` green (≥80 lines/statements, ≥75 branches/functions; new files ≥85)
- [ ] Run existing `tests/api/test_upload_cancel.py` + `tests/unit/core/test_storage_client.py` to confirm no regressions
- [ ] Manual DEV GCS smoke executed and documented in PR (`RUN_GCS_SMOKE=1 GCS_UPLOAD_BUCKET=erpsense-backend-uploads-dev`)

---

I have all the verified anchors I need. Writing the section now.

## 15. Dev-Environment Rollout, Backward-Compat & Kill Switch

This section makes the auto-compression feature ship to the **dev Cloud Run environment** (not just a developer's laptop), default OFF, with a one-flag kill switch, zero downtime, no DB migration, full backward-compat with already-stored originals, and concrete monitoring/rollback. It builds on the seams confirmed on `feat/auto-compression` (HEAD `1d3f285c`).

The grounding facts that drive every decision here:

- The Genie persist sink is selected by `app/services/storage_client.py:get_storage_client()` (lines 399–418): GCS is chosen when `settings.gcs_bucket_name or settings.gcs_upload_bucket` is non-empty, else `LocalStorageClient(settings.upload_dir)` with only a WARNING. So "works in DEV with GCS" means: in dev Cloud Run the env must resolve `get_storage_client()` to `GCSStorageClient`, and a misconfig degrades **silently** to local disk — we must assert against that.
- Deployed dev injects **only** `GCS_UPLOAD_BUCKET` (`terraform/modules/cloud-run/main.tf:609-620`, the `phase2_enabled` dynamic env block), never `GCS_BUCKET_NAME`. The provisioned bucket is `erpsense-backend-uploads-dev` (`terraform/modules/phase2/main.tf:101-129`). Local `.env` instead sets `GCS_BUCKET_NAME=erpsense-ocr-dev` + `GCS_UPLOAD_BUCKET=erpsense-backend-uploads-dev`. The realistically-working dev bucket is therefore `erpsense-backend-uploads-dev` via `gcs_upload_bucket`.
- IAM on that bucket is least-privilege: `roles/storage.objectCreator` + `roles/storage.objectViewer` only (`terraform/modules/phase2/main.tf:35-38`, `local.gcs_upload_roles`). **No overwrite, no delete, no signBlob.** Compression keeps the same filename, and `_save_original_file`'s dup-suffixer (`upload.py:249-254`) already guarantees fresh paths — so `objectCreator` (create-new-only) keeps working unchanged.
- No schema is touched. The compressed bytes flow through the **single** existing `upload_bytes` write at `upload.py:258` (inside `_save_original_file`, lines 225-264). No new column, no new table, no Alembic revision — the `alembic heads == 1` CI gate (`deploy.yml:111-253`) stays green with zero migration work.

### 15.1 Feature flag (default OFF) — the kill switch

All compression behavior is gated behind config flags so the feature is dark on merge and can be killed without a redeploy of code. Per the be-logging gotcha (R03: "no `os.getenv` outside `app/config.py`"), every knob lives in `Settings`.

**MODIFY `app/api/v1/endpoints/erpsense-backend/app/config.py`** — add a block beside the existing OCR/storage fields (the same idiom as `brs_orphan_cleanup_enabled` at `config.py:521`, which is the canonical "safe-off, enable per env once provisioned" precedent):

```python
# Genie upload auto-compression (Section 15 rollout). Safe-off by default;
# staged-enable per env once the dev uploads bucket + GCS path are verified.
upload_compression_enabled: bool = False          # master kill switch
upload_compression_image_enabled: bool = True     # sub-toggle: images
upload_compression_pdf_enabled: bool = True        # sub-toggle: scanned PDFs
upload_compression_min_bytes: int = 500 * 1024     # decision 8: <500KB exempt
upload_compression_image_max_bytes: int = 200 * 1024     # image target ceiling
upload_compression_image_max_dim: int = 1200             # image px ceiling
upload_compression_pdf_max_bytes_per_page: int = 300 * 1024  # scanned-PDF/page
upload_compression_container_rebuild: bool = False  # decision 7: OFF by default
```

Two-level gating matters: `upload_compression_enabled=False` short-circuits everything (true kill switch); the per-type sub-toggles allow disabling only PDFs (riskier reportlab/PyMuPDF rebuild) while keeping images, without a code change. `upload_compression_container_rebuild` honors decision 7 (native-PDF/DOCX container rebuild is guarded and OFF).

**MODIFY `app/api/v1/endpoints/upload.py:upload_file()`** at the compression seam (after `content = await file.read()` at line 864 and after PDF page validation ends ~1019, feeding into the `_save_original_file` call at 1024-1025). The very first check is the flag, so the disabled path is byte-for-byte identical to today:

```python
# Pseudocode at the wiring point (upload.py ~1020-1024)
bytes_to_persist = content                      # default: original (today's behavior)
if settings.upload_compression_enabled and session_id:
    bytes_to_persist = await maybe_compress_upload(
        content=content,                        # a COPY is made inside; never temp_path
        filename=original_filename,
        content_type=content_type,
    )                                            # returns original on failure/regression
if session_id:
    await _save_original_file(session_id, original_filename, bytes_to_persist)
# OCR call at 1044 still reads the UNTOUCHED temp_path (original)
```

When `upload_compression_enabled` is `False`, `bytes_to_persist is content` and the call at 1024-1025 is unchanged — **flag-off == current production behavior**, which is the backward-compat guarantee at the request level.

The compression logic itself (`maybe_compress_upload`) lives in a **NEW non-omitted module** `app/services/file_processing/compression.py` (per tests-ci gotcha: `upload.py` and the OCR handlers are in `[tool.coverage.run].omit`, so coverage-bearing logic must not live there). That module owns the per-type strategy, the fallback contract (decision 9: on failure or `compressed >= original`, return `content`, never raise), and the structured telemetry (`compression.decision` / `compression.fallback` events per be-logging).

### 15.2 Dev GCS: bucket, service account, env/secrets (Terraform)

The dev uploads bucket and Cloud Run SA already exist; **no new infra is strictly required** because compressed bytes use the same `upload_bytes(path, data, content_type)` contract through the same bucket. The rollout work is verification + wiring the new flags as env vars so they're togglable per environment.

**Bucket — already provisioned, verify only.** `terraform/modules/phase2/main.tf:101-129` creates `${name_prefix}-uploads-${env}` = `erpsense-backend-uploads-dev` when `phase2_enabled=true` (dev sets it true at `terraform/dev.tfvars:161`). `public_access_prevention=enforced`, `uniform_bucket_level_access=true`, 365-day delete lifecycle, `force_destroy=true` in dev only. Compressed objects inherit all of this. **No bucket change needed.**

**Service account / Workload Identity — already correct, verify only.** Cloud Run runs as the platform remote-state `cloud_run_base_sa_email` (`terraform/main.tf:91`, `local.service_account`), which holds `objectCreator + objectViewer` on the bucket (`phase2/main.tf:35-38,135-140`). Auth is ADC (no key file in Dockerfile/terraform); `storage.Client(project=...)` resolves credentials from the Cloud Run SA implicitly. Because compression keeps the same filename and never overwrites/deletes, **no new IAM role is required** — `objectCreator` covers the create-new write of the compressed object, `objectViewer` covers the preview byte-proxy `download_bytes`/`exists`. Do **not** add `signBlob`/`tokenCreator`: the v4 signed-URL helper (`storage_client.py:335-376`) is dead and the preview path is the byte proxy (`get_serve_response`, `storage_client.py:378-396`).

**Wire the new flags as Cloud Run env vars — MODIFY Terraform.** The flags must be settable per environment so we can stage-enable dev → staging → prod without code edits:

- **MODIFY `terraform/modules/cloud-run/main.tf`** — add a dynamic env block mirroring the existing `phase2_enabled` block at lines 609-620 (do not piggyback on `phase2_enabled`; compression has its own lifecycle):

```hcl
dynamic "env" {
  for_each = {
    UPLOAD_COMPRESSION_ENABLED         = tostring(var.upload_compression_enabled)
    UPLOAD_COMPRESSION_IMAGE_ENABLED   = tostring(var.upload_compression_image_enabled)
    UPLOAD_COMPRESSION_PDF_ENABLED     = tostring(var.upload_compression_pdf_enabled)
    UPLOAD_COMPRESSION_CONTAINER_REBUILD = tostring(var.upload_compression_container_rebuild)
  }
  content { name = env.key  value = env.value }
}
```

- **MODIFY `terraform/modules/cloud-run/variables.tf`** — add `variable "upload_compression_enabled" { type = bool default = false }` and siblings (default all the booleans to the safe-off/safe-default values; `container_rebuild` default `false`).
- **MODIFY `terraform/main.tf`** — pass `upload_compression_enabled = var.upload_compression_enabled` (etc.) into the `cloud-run` module call (~line 320-322, beside `gcs_upload_bucket_name`), and add the top-level `variable` declarations in `terraform/variables.tf`.
- **MODIFY `terraform/dev.tfvars`** — initially set `upload_compression_enabled = false` (merge dark). The staged-enable (15.5) is a one-line tfvars flip to `true`.
- Leave `terraform/staging.tfvars` and `terraform/prod.tfvars` at `false` until dev soak passes.

These are plain (non-secret) booleans, so they go in the env block, **not** Secret Manager — no secret rotation involved. The case-insensitive Pydantic `Settings` (`env_file=.env`, `case_sensitive=False`) maps `UPLOAD_COMPRESSION_ENABLED` → `upload_compression_enabled` automatically; no parsing code needed.

**MODIFY `env.example`** (and dev `.env`) — document the new vars next to the existing GCS block (`env.example:51-58`) so local/dev parity is explicit, defaulting `UPLOAD_COMPRESSION_ENABLED=false`.

**Dev GCS pre-flight (the silent-degrade trap).** Per be-storage gotcha, `LocalStorageClient` warns but does not fail when no bucket is configured, so a broken GCS env degrades to ephemeral local disk. Before flipping the flag in dev, assert the bucket exists and ADC can write:

```bash
gcloud storage buckets describe gs://erpsense-backend-uploads-dev    # exists?
# round-trip as the Cloud Run SA via impersonation (objectCreator+objectViewer):
gcloud storage cp /tmp/probe.png gs://erpsense-backend-uploads-dev/sessions/_probe/originals/probe.png \
  --impersonate-service-account="$CLOUD_RUN_BASE_SA_EMAIL"
gcloud storage cat gs://erpsense-backend-uploads-dev/sessions/_probe/originals/probe.png | head -c 8
```

### 15.3 No DB migration / zero-downtime

- **No migration.** Compressed bytes are persisted through the existing single `upload_bytes` at `upload.py:258`; no new column/table/index. `alembic heads` stays at 1, so the backend CI "Check Alembic heads" step (`deploy.yml:111-253`) is unaffected and no `alembic upgrade` runs at deploy.
- **Zero-downtime.** Cloud Run does a rolling revision swap; the new revision carries `UPLOAD_COMPRESSION_ENABLED=false`, so its behavior is identical to the old revision. Flipping the flag later is an env-var-only revision (no image rebuild), again a rolling swap. No traffic interruption, no draining concern.
- **No worker / no new infra for execution.** Per decision 6, compression is synchronous-inline inside the request (`asyncio.gather` + `asyncio.to_thread` within `upload_file`). There is no queue, no Cloud Tasks change, no new container — rollout touches only env vars + code already shipped dark.

### 15.4 Backward-compat, idempotency, and the preview/cancel invariants

**Already-stored originals keep working.** Files written before the feature was enabled live at `sessions/{sid}/originals/{filename}` and are served by `serve_original_file` (`upload.py:1927-2004`) whose `media_type` is **suffix-derived** (`MIME_TYPE_MAP` at 1908-1924, fallback `mimetypes`). Because compression **preserves the original filename and extension** (decision 4), old and new objects are indistinguishable to the serve path — a pre-compression original and a post-compression compressed object render identically in both the chat preview (`buildPreviewUrl` → `useAuthedFileUrl` blob, `chat-utils.ts:19-28`) and the BE byte proxy. **No re-processing or migration of historical files** — they are simply served as-is.

**Cancel/purge stays correct.** `DELETE /upload/{upload_id}` → `_purge_upload_artifacts` (`upload.py:1366-1510`) deletes the saved original by listing `sessions/{sid}/originals` and matching `{stem}.{ext}` / `{stem}_{counter}.{ext}`. Since compression keeps the same stem+ext, purge matches the compressed object unchanged. (Note: bucket IAM lacks `objectAdmin`/delete in prod — that is a pre-existing constraint of the purge path, not introduced here; dev `force_destroy` and local disk delete fine, and this rollout does not change delete semantics.)

**Idempotency.** Each upload writes a fresh object (the dup-suffixer renames on collision, `upload.py:249-254`); compression is a pure function of the in-memory `content` with a deterministic fallback to `content`. Re-uploading the same file produces a new path, never overwriting an existing object — which is exactly what `objectCreator`-only IAM requires. There is no read-modify-write, so no idempotency-key/race concern.

**Frontend backward-compat.** The FE Genie limits are already 30MB/32MB/5/10 (`constants.ts:1893-1896`) — decision 8's "30MB not 10MB" is already satisfied on this surface; **do not lower it.** Any compression-transparency UI (optional `original_size`/`stored_size`/`was_compressed` on `UploadResponse` in `types/chat.ts:442-464`) is purely additive — when the BE flag is OFF those fields are absent and the existing `file-preview-modal.tsx` header (`formatBytes(attachment.fileSize)`, line 164-167) renders exactly as today. No FE deploy is required to ship the BE feature dark.

### 15.5 Staged enable, monitoring, rollback

**Staged enable (per-env, flag-only):**
1. Merge with `upload_compression_enabled=false` everywhere (dark).
2. Run the 15.2 dev GCS pre-flight; confirm `get_storage_client()` resolves to `GCSStorageClient` in dev (assert via a dev smoke that the WARNING from `storage_client.py:412-417` is **absent** and a probe object lands in GCS, not local disk).
3. Flip `terraform/dev.tfvars` → `upload_compression_enabled=true`; `terraform apply` (env-var-only revision). Soak in dev.
4. Promote to staging tfvars, then prod tfvars, one env at a time, with the monitoring below green at each step. If PDF rebuild looks risky in soak, keep `upload_compression_pdf_enabled=false` / `upload_compression_container_rebuild=false` and ship images first.

**Monitoring/alerts (be-logging idiom — log-based, primary).** Emit named events from `compression.py` via the `diagnostic_metrics.py` pattern (`logger.info("<event>", extra={...})`, never f-strings, never request-scoped ids which are auto-injected):
- `compression.decision` — `extra={filename, content_type, strategy, decision, original_size_bytes, compressed_size_bytes, saved_bytes, saved_pct, duration_ms, storage_backend}`.
- `compression.fallback` — `extra={filename, fallback_reason ("error"|"size_regression"|"below_min"), original_size_bytes, duration_ms}` at **WARNING** (recoverable, not a raise — no double-logging vs `main.py` handlers).
- `gcs.upload` — `extra={storage_path, storage_backend, bytes, status, duration_ms}`, emitted at the `_save_original_file` caller seam (replacing the existing f-string banner log at `upload.py:259`, bringing it into CLAUDE.md compliance).

Optionally add a registry `_init_compression_metrics()` (mirror `_init_brs_metrics`, `metrics.py:864`) wired into `__post_init__` **and** `reset_metrics()` for the Prometheus `/api/v1/metrics` scrape: `compression_decisions_total{strategy,decision}`, `compression_fallback_total{reason}`, `gcs_upload_total{backend,status}`, histograms `compression_bytes_saved`, `gcs_upload_latency_seconds`. Keep label cardinality low; **never** use `tenant_id` as a registry label.

Cloud Monitoring (platform-side, `erpsense-platform/terraform/modules/monitoring`, out of backend repo scope — flag for the platform team) derives:
- **Alert: fallback rate** — `compression.fallback` count / `compression.decision` count > 25% over 30m (signals a broken compressor → falling back excessively, still safe but ineffective).
- **Alert: silent-degrade** — any `gcs.upload` with `storage_backend="LocalStorageClient"` in a deployed env (means GCS misconfig → ephemeral disk; page immediately).
- **Alert: latency** — `gcs_upload_latency_seconds` or compression `duration_ms` p95 regression vs the flag-off baseline (compression must not push the request near the 300s OCR timeout).

**Rollback / kill switch (no redeploy of code):**
- **Instant kill:** set `upload_compression_enabled=false` in the relevant tfvars and `terraform apply` (env-var-only revision, rolling, ~seconds). The next upload persists the raw `content` again — identical to pre-feature behavior. Already-stored compressed objects remain perfectly serveable (same filename/suffix), so a kill is non-destructive and needs no data cleanup.
- **Code rollback:** Cloud Run "rollback to previous revision" if the disabled-path code path itself regresses (unlikely — flag-off is a no-op branch). No DB rollback exists or is needed.
- **Partial kill:** flip `upload_compression_pdf_enabled` / `upload_compression_image_enabled` independently to disable only the risky type.

### 15.x Checklist

- [ ] Add `upload_compression_enabled` (+ image/pdf sub-toggles, `min_bytes`, image `max_bytes`/`max_dim`, pdf `max_bytes_per_page`, `container_rebuild`) to `app/config.py` Settings, all safe-off/safe-default, beside the `brs_orphan_cleanup_enabled` precedent.
- [ ] Create `app/services/file_processing/compression.py` (`maybe_compress_upload`) — NOT in `[tool.coverage.run].omit`; copies in-memory `content`, never touches `temp_path`; returns original on failure/`compressed>=original`/below-min; never raises.
- [ ] Wire the flag gate + `bytes_to_persist` at `upload.py:upload_file()` (after read 864 / page-validation ~1019), feeding `_save_original_file` at 1024-1025; confirm flag-off path is byte-identical to current behavior and OCR still reads untouched `temp_path` at 1044.
- [ ] Confirm `_save_original_file` (`upload.py:225-264`) still does exactly ONE `upload_bytes` write (line 258); do not add a second write (dup-suffixer would rename it).
- [ ] Confirm compressed output keeps original filename + extension so `serve_original_file` (`upload.py:1927-2004`) suffix-derived `media_type` and `_purge_upload_artifacts` (`upload.py:1366-1510`) stem/ext matching both keep working.
- [ ] Replace the f-string/banner log at `upload.py:259` with structured `logger.info("upload.original_saved"/"gcs.upload", extra={...})`; add `compression.decision` (INFO) + `compression.fallback` (WARNING) events in `compression.py` (no f-strings, no request-scoped ids in extra).
- [ ] (Optional) Add `MetricsRegistry._init_compression_metrics()` wired into `__post_init__` AND `reset_metrics()`; verify it exports at `GET /api/v1/metrics`.
- [ ] Add `upload_compression_*` variables to `terraform/modules/cloud-run/variables.tf` (booleans, defaults false/true per knob).
- [ ] Add a dynamic `env` block in `terraform/modules/cloud-run/main.tf` (mirror the `phase2_enabled` block at 609-620) emitting `UPLOAD_COMPRESSION_*` vars; do NOT couple to `phase2_enabled`.
- [ ] Declare top-level `variable "upload_compression_*"` in `terraform/variables.tf` and pass them into the `cloud-run` module in `terraform/main.tf` (beside `gcs_upload_bucket_name`).
- [ ] Set `upload_compression_enabled=false` in `terraform/dev.tfvars`, `staging.tfvars`, `prod.tfvars` for the dark merge.
- [ ] Document `UPLOAD_COMPRESSION_*` in `env.example` and local `.env` next to the GCS block (env.example:51-58), default `false`.
- [ ] Verify NO Alembic migration is added; `alembic heads` == 1 (backend CI gate).
- [ ] Dev GCS pre-flight: `gcloud storage buckets describe gs://erpsense-backend-uploads-dev`; SA-impersonated `cp`/`cat` round-trip with `objectCreator+objectViewer`.
- [ ] Confirm `get_storage_client()` resolves to `GCSStorageClient` in dev (assert the `storage_client.py:412-417` WARNING is absent and the probe lands in GCS, not local disk).
- [ ] Confirm NO new IAM role is required (no overwrite/delete/`signBlob`); compression keeps fresh paths under `objectCreator`-only.
- [ ] Verify already-stored (pre-feature) originals still preview via `serve_original_file` and chat `buildPreviewUrl` blob path with the feature ON — no historical re-processing.
- [ ] Verify FE Genie limits remain 30MB/32MB/5/10 (`constants.ts:1893-1896`); do NOT lower to 10MB; any `UploadResponse` compression-transparency fields (`types/chat.ts:442-464`) are additive and absent when flag is OFF.
- [ ] `terraform apply` dev with flag OFF (zero-downtime rolling revision); confirm uploads behave exactly as before.
- [ ] Flip `dev.tfvars` → `upload_compression_enabled=true`; `terraform apply` (env-only revision); soak in dev.
- [ ] Platform team: add Cloud Monitoring log-based metrics + alerts in `erpsense-platform/.../monitoring` — fallback-rate >25%/30m, any deployed `storage_backend="LocalStorageClient"` (silent-degrade page), p95 compression/upload latency regression vs flag-off baseline.
- [ ] Promote flag to `staging.tfvars` then `prod.tfvars` one env at a time, alerts green at each step; keep PDF/`container_rebuild` toggles OFF if soak flags risk.
- [ ] Rehearse kill switch: flip `upload_compression_enabled=false` + `terraform apply`; confirm next upload persists raw `content` and previously-compressed objects still serve (non-destructive).
- [ ] Document rollback runbook: env-flag kill (seconds) vs Cloud Run revision rollback; note no DB rollback exists/needed.

---

Confirmed the key anchors against HEAD `1d3f285c` on `feat/auto-compression`: `_save_original_file` def at 225, single `upload_bytes` at 258, `content = await file.read()` at 864, `_save_original_file(...)` call at 1025, coverage `omit` list includes upload.py/library_gemini/pdf_handler/image_handler/file_analysis (so new compression code must live elsewhere), CI gate is `--cov-fail-under=75`, and deps confirm PyMuPDF/reportlab/Pillow present, pikepdf/img2pdf absent.

## 16. Autonomous One-Pass Execution & Verification Gates

This section defines how to implement the entire auto-compression + GCS-persist + preview + ERPHome-attach + observability feature in a single autonomous pass on branch `feat/auto-compression` (HEAD `1d3f285c`), without weekly phases. It gives a dependency-ordered build order, a multi-agent execution recipe (implement → self-review → adversarial security review → fix loop), the verification gates run **every iteration**, and explicit done-criteria. The governing principle from the corrected decisions is preserved throughout: **OCR reads the original `temp_path`; storage saves the compressed copy; compression operates on a COPY of in-memory `content`, never `temp_path`; on any failure or size-regression, store the original (non-blocking, logged); storage is per-surface (Genie→GCS, ERPHome→ERPNext).**

### 16.1 Pre-flight invariants (assert before writing any code)

These are the cross-cutting constraints that, if violated, silently break existing behavior. Re-grep each before the first edit and re-assert after each iteration.

- **Coverage placement.** `pyproject.toml [tool.coverage.run].omit` (lines 114–147) excludes `app/api/v1/endpoints/upload.py` (145), `app/services/file_processing/ocr/library_gemini.py` (127), `app/services/file_processing/handlers/pdf_handler.py` (140), `app/services/file_processing/handlers/image_handler.py` (141), `app/services/file_analysis.py` (147). **All coverage-bearing compression logic MUST live in NEW non-omitted modules** (`app/services/file_processing/compression.py`, `app/services/file_processing/handlers/_pdf_classifier.py`, `app/services/file_processing/compression_metrics.py`). The wiring edits inside `upload.py` and `_attachments.py` are thin call-sites; the heavy logic that must count toward the 75% gate is in the new modules.
- **Filename/extension immutability.** `serve_original_file` (`upload.py:1933-1995`) derives `media_type` from `MIME_TYPE_MAP[suffix]`; FE `chat-utils.ts:buildPreviewUrl` + `EXT_TO_MIME` also key off extension. Compression MUST return the **same filename + extension** (a JPEG stays `.jpg`/`image/jpeg`). Never let the dup-filename suffixer (`upload.py:249-254`) fire a second time — **overwrite the single `upload_bytes` write at `upload.py:258`**, do not add a 2nd write.
- **No temp_path mutation.** `ImageHandler._resize_if_needed` (`image_handler.py:139-167`, `img.save(file_path)` at 162) overwrites `temp_path` in place before OCR; PDF page-count validation + OCR both read `temp_path`. Compression input is a `bytes`/`BytesIO` copy of `content` only.
- **Non-blocking contract.** `_save_original_file` returns `None` on failure (OCR proceeds). A compression exception MUST be caught and fall back to original bytes — it must never raise into the request or block OCR (`upload.py:1044` `file_router.process(temp_path, languages=languages)`).
- **Logging idiom.** Event-name-first `logger.info("event.name", extra={...})`, no f-strings, no `"="*60` banners. Replace the existing f-string log at `upload.py:259` with a structured event. Never log request-scoped ids (request_id/tenant_id/session_id/user_id) in `extra` — they're auto-injected by `JSONFormatter` (`core/logging.py:57-90`) via `request_context.get_log_context()`. Never log file bytes/PII.
- **Config not os.getenv.** All thresholds (image px/KB, scanned-PDF KB/page, min-exempt bytes, rebuild flag) go in `app/config.py` Settings, not `os.getenv` (R03).
- **ERPHome cap reality.** `_shared.py:111 _MAX_ATTACHMENT_SIZE_BYTES = 10MB` is the genuine ERPHome cap on `doc_social.py`/`file_proxy.py` — do NOT raise to 30MB. The 30MB reconciliation is the Genie surface only, and the FE already has 30MB (`constants.ts:1893-1896`) — do NOT lower it.
- **Dead-code / IAM traps.** Do not switch preview to `_generate_signed_url_sync` (`storage_client.py:335-376`, dead, needs ungranted `signBlob`). Cloud Run SA has `objectCreator+objectViewer` only — overwriting an existing GCS object may 403; the suffixer's fresh-path discipline avoids this for new uploads (overwriting the single in-flight write is fine because it's a never-before-written path per upload).

### 16.2 Dependency-ordered build order (one pass, bottom-up)

Build leaf modules first (independently unit-testable, coverage-counted), then wire call-sites, then config/limits, then FE, then tests, then the review loop. Each step lists files to CREATE / MODIFY and the gate it must pass before proceeding.

**Step 0 — Config & flags (no behavior change yet).**
- MODIFY `app/config.py` (Settings, near GCS fields 243–245 / size limits 525–526): add
  ```
  compression_enabled: bool = True
  compression_image_max_bytes: int = 200 * 1024          # decision 7
  compression_image_max_dimension: int = 1200            # decision 7
  compression_scanned_pdf_max_bytes_per_page: int = 300 * 1024
  compression_min_exempt_bytes: int = 500 * 1024         # decision 8 (<500KB never compressed)
  compression_pdf_rebuild_enabled: bool = False          # decision 7 (native-PDF container rebuild OFF by default, guarded)
  compression_jpeg_quality_start: int = 85
  ```
  Gate: `ruff check`, `mypy app/`. No runtime change (nothing reads these yet).

**Step 1 — Shared native/scanned classifier (factor without changing OCR behavior).**
- CREATE `app/services/file_processing/handlers/_pdf_classifier.py`:
  ```python
  def avg_chars_per_page(text: str, page_count: int) -> float: ...
  def is_native_pdf(text: str, page_count: int, *, mode: Literal["total", "average"] = "total",
                    threshold: int = 50) -> bool:
      # mode="total"   -> len(text) >= threshold * page_count   (PDFHandler semantics)
      # mode="average" -> avg_chars_per_page(...) >= threshold   (library_gemini semantics)
  ```
- MODIFY `pdf_handler.py:_has_sufficient_text` (302–308): keep method, body becomes `return is_native_pdf(text, page_count, mode="total", threshold=self.MIN_TEXT_LENGTH)` — **preserves TOTAL semantics**.
- MODIFY `library_gemini.py:_extract_pdf_native` (589–618): replace inline `avg < MIN_PDF_TEXT_PER_PAGE` check with `if not is_native_pdf(text, page_count, mode="average", threshold=self.MIN_PDF_TEXT_PER_PAGE): raise FileProcessingException(...)` — **preserves AVERAGE semantics AND the raise control-flow** (gotcha: must not collapse raise↔branch).
- Existing-behavior guard: do NOT change `MIN_TEXT_LENGTH`/`MIN_PDF_TEXT_PER_PAGE` (=50), 200 DPI, or routing. Both call-sites are pure delegates; OCR verdicts are byte-identical.
- Gate: `ruff`, `mypy`, `pytest tests/unit/file_processing/test_pdf_classifier.py` (new, see 16.4).

**Step 2 — Compression core (the coverage-bearing engine).**
- CREATE `app/services/file_processing/compression.py`. Pure, sync-CPU functions wrapped for `asyncio.to_thread`; no FastAPI, no storage, no temp_path. Signatures:
  ```python
  @dataclass(frozen=True)
  class CompressionResult:
      data: bytes; original_size: int; compressed_size: int
      strategy: str            # "image_jpeg" | "image_png" | "scanned_pdf" | "none"
      decision: str            # "compressed" | "skipped_exempt" | "skipped_native" | "fallback_original"
      fallback_reason: str | None

  def classify_for_compression(filename: str, content_type: str | None, content: bytes) -> str:
      # -> "image" | "scanned_pdf" | "native"  (uses _pdf_classifier on a BytesIO copy)

  def compress_image(content: bytes, *, max_bytes: int, max_dim: int, quality_start: int) -> bytes:
      # Pillow on BytesIO; replicate ImageHandler RGBA/P->RGB (image_handler.py:159-161);
      # downscale to max_dim, iterative JPEG quality step-down (or PNG optimize), keep mode/format suffix-compatible.

  def compress_scanned_pdf(content: bytes, *, max_bytes_per_page: int) -> bytes:
      # PyMuPDF rasterize @200 DPI (pattern from pdf_handler._pdf_to_images:310-326) ->
      # Pillow JPEG per page -> rebuild via reportlab (pikepdf/img2pdf NOT available).

  async def compress_for_storage(filename, content_type, content, settings) -> CompressionResult:
      # decision tree honoring decisions 7/8/9; runs CPU work via asyncio.to_thread;
      # decision 6 'parallel' = asyncio.gather + to_thread for multi-page; NEVER raises (returns fallback_original).
  ```
  Rules baked in: `< compression_min_exempt_bytes` → `skipped_exempt` (decision 8); `native` → `skipped_native` unless `compression_pdf_rebuild_enabled` (decision 7, off by default); on exception OR `compressed_size >= original_size` → `decision="fallback_original"`, `data=content` (decision 9, no retry). Output for images stays same suffix/MIME; scanned PDF stays `.pdf`/`application/pdf`.
- Gate: `pytest tests/unit/file_processing/test_compression.py` (new).

**Step 3 — Compression telemetry (log-based + optional registry).**
- CREATE `app/services/file_processing/compression_metrics.py` mirroring `tally_sync/diagnostic_metrics.py:record_run` shape:
  ```python
  logger = logging.getLogger(__name__)
  def record_compression(*, filename, content_type, strategy, decision, original_size,
                         compressed_size, saved_pct, duration_ms, storage_backend,
                         fallback_reason=None, surface): ...  # logger.info("compression.decision"/"compression.fallback", extra={...})
  def record_gcs_upload(*, storage_backend, status, size_bytes, duration_ms, storage_path_suffix): ...  # "gcs.upload"
  ```
- MODIFY `app/core/metrics.py` (OPTIONAL Prometheus scrape): add `MetricsRegistry._init_compression_metrics()` (mirror `_init_brs_metrics:864`) defining `compression_decisions_total{strategy,decision,surface}`, `compression_fallback_total{reason,surface}`, `gcs_upload_total{backend,status}`, `Histogram compression_bytes_saved`, `gcs_upload_latency_seconds`. Wire into `__post_init__` (490–506) AND `reset_metrics()` (980–999) — **both, or pytest leaks state**. Keep label cardinality low; never use tenant_id as a label (it's in the log `extra`).
- Gate: `pytest tests/unit/core/test_metrics_compression.py` + existing `tests/` metrics tests still green.

**Step 4 — Genie persist wiring (THE single seam).**
- MODIFY `app/api/v1/endpoints/upload.py`:
  - At the existing call site `upload.py:1025` `await _save_original_file(session_id, original_filename, content)` (which already runs after read@864 and after PDF page-count validation, before OCR@1044 and before unlink@1359 — **no reordering needed**): compute compressed bytes first and pass them in. Cleanest: extend `_save_original_file` (225–264) to accept the already-compressed `CompressionResult` (or call `compression.compress_for_storage` *inside* it on a copy of `content`) and write `result.data` at the **single** `upload_bytes(storage_path, content, content_type)` at line 258 — overwrite the bytes argument, do not add a write.
  - Replace the f-string log at `upload.py:259` with `logger.info("upload.original_saved", extra={...})` and call `compression_metrics.record_compression(..., surface="genie")` + `record_gcs_upload(...)` with `storage_backend = type(get_storage_client()).__name__`.
  - Guard with `settings.compression_enabled`; on `decision="fallback_original"` log WARNING `compression.fallback` (recoverable, do not raise — "no double logging" means do not log+raise).
  - Existing-behavior guard: OCR at 1044 still reads untouched `temp_path`; cancel-purge (`_purge_upload_artifacts` 1366+) still matches `{stem}.{ext}`/`{stem}_{counter}.{ext}` because filename is unchanged.
- Gate: `pytest tests/api/test_upload_compression.py` (new, mirrors `test_upload_cancel.py`).

**Step 5 — ERPHome attach wiring (compression only, NO GCS, NO OCR).**
- MODIFY `app/adapters/erpnext/_attachments.py`:
  - `upload_attachment` (77–168): before multipart build at line 108, add `compress: bool = True` kwarg; if `compress` and `settings.compression_enabled`, compute compressed COPY of `file_content` via `compression.compress_for_storage(file_name, content_type, file_content, settings)`; use `result.data` in the `files = {"file": (file_name, ..., content_type or "application/octet-stream")}` tuple (keep filename+content_type as the multipart tuple's 3rd element, NOT a header). Wrap in try/except so failure → original bytes, normal success envelope preserved.
  - `upload_file_standalone` (170–233): same hook before line 196 + `compress: bool = True`.
  - Extend the existing `logger.info("Attachment uploaded", extra={size:...})` at 147/222 with `original_size/compressed_size/compressed/strategy` and `record_compression(..., surface="erphome")`. **No GCS write here** (per-surface).
- MODIFY `app/services/erpnext_provisioning/_steps_company.py:335`: pass `compress=False` (logo already optimized by BrandingService — the only adapter caller that must opt out).
- Existing-behavior guard: callers `doc_social.py:335`, `file_proxy.py:189`, `employee_expense_service.py:617/651`, `employee_travel_service.py:750` inherit compression unchanged (no per-caller edits). ERPHome cap stays 10MB.
- Gate: `pytest tests/api/v1/test_erphome_attach_compression.py` (new, respx-mocks ERPNext `POST {BASE_URL}/api/method/upload_file`).

**Step 6 — Frontend (additive transparency only; size limits already correct).**
- MODIFY `src/types/chat.ts` (`UploadResponse`/`ExtractionMetadata` 442–464): add optional `original_size?: number; stored_size?: number; was_compressed?: boolean` (purely additive, no breakage).
- MODIFY backend `UploadResponse` return (`upload.py:1111-1116`) to populate these from the `CompressionResult` so the FE can show "compressed from X".
- MODIFY `src/components/chat/file-preview-modal.tsx` (header 164–167) to surface "compressed from X → Y" when `was_compressed`; add i18n keys under `chat:file_preview` (`chat.json:134`). Do NOT touch `OCR_MAX_*` constants (already 30/32/5/10).
- DO NOT add new inline endpoints (register `${ERP_DOCS.BASE}/file` under `API_ENDPOINTS.ERP_DOCS` only if `erp-documents.ts` is otherwise edited). Preview must keep using the blob-URL/`useAuthedFileUrl` pattern (JWT-in-header), never raw `<img src>` to a protected path.
- Coverage placement: put any new pure logic in a NON-excluded `src/lib/utils/*.ts` (component dirs are excluded), or add a per-file `coverageThreshold` entry. Migrate the hardcoded English error strings in `chat-input.tsx:214/247/271/298` to i18n only if those lines are touched.
- Gate: `npm run lint && npm run type-check && npm test -- tests/unit/lib/api/upload.test.ts tests/unit/lib/utils/`.

**Step 7 — Tests, smoke, docs (see 16.4 / 16.5).** Step 8 — review loop (16.3).

### 16.3 Multi-agent execution recipe (implement → review → security → fix)

Each step above runs through a 4-role loop before its gate is considered passed. Roles are sequential sub-agents over the same working tree; the loop repeats until zero Must-Fix findings.

1. **Implementer** — writes the minimal diff for the current step using the project skills: `backend-dev`/`backend-api` for `compression.py`/`upload.py`/`_attachments.py`, `backend-models`/`backend-migrations` only if a DB column is added (none required here — Genie→GCS, ERPHome→ERPNext, no new table), `frontend-dev`/`frontend-components`/`frontend-accessibility` for the preview, `testing-test-writing` for tests, `global-modularity` (keep new files < 500 lines; `compression.py` split into `compression.py` + helpers if it grows), `global-error-handling`/`global-validation`/`global-conventions`/`global-coding-style` for idioms.
2. **Self / code reviewer** — run `/code-review` (effort `high`) on the diff. Focus checklist: filename/extension preserved; no 2nd `upload_bytes` write; compression never mutates `temp_path`; fallback never raises/blocks OCR; coverage-bearing code outside the omit list; no f-string logs / no banner logs; no request-scoped ids in `extra`; classifier semantics unchanged at both sites; ERPHome cap stays 10MB; per-surface storage (no rogue GCS write on ERPHome).
3. **Adversarial security reviewer** — run `/security-review` on the branch diff. Threat checklist: path traversal via filename in storage path (reuse `_save_original_file` sanitizer 241–244 and `serve_original_file` sanitize 1975); BOLA on preview (session-owner verify `upload.py:1972`, cross-tenant must 404); decompression-bomb / pixel-bomb (cap pixels before Pillow decode, respect `MAX_DIMENSION`, guard PDF page count before rasterize); zip-bomb-style PDF (page-count already validated 954–1019 before compression); resource exhaustion (200 DPI rasterize of a 10-page PDF bounded by `MAX_PAGES_PER_PDF=10` + `asyncio.to_thread`, 300s OCR timeout untouched); no PII/bytes in logs; signed-URL/IAM not introduced; multipart Content-Type stays the tuple's 3rd element (no header injection); compressed output content-type matches suffix (no MIME confusion in preview).
4. **Fixer** — applies findings, re-runs the step gate. Loop until clean, then advance.

A final full-suite pass runs the loop once more across the **entire** diff (all steps) before declaring done.

### 16.4 Where new tests go (must count toward 75% backend / 85% new)

- `tests/unit/file_processing/test_pdf_classifier.py` — total vs average semantics; empty-page-offset-by-dense-page edge (proves the two modes diverge as intended).
- `tests/unit/file_processing/test_compression.py` — image downscale+quality stepdown keeps suffix/MIME; RGBA/P→RGB; `<500KB` → `skipped_exempt`; native PDF → `skipped_native`; scanned PDF → per-page bound; `compressed>=original` → `fallback_original`; exception in Pillow/PyMuPDF → `fallback_original` (monkeypatch to raise); never raises.
- `tests/unit/core/test_metrics_compression.py` — registry counters/histograms exist, `reset_metrics()` clears them.
- `tests/api/test_upload_compression.py` — mirror `tests/api/test_upload_cancel.py`: autouse `reset_upload_registry_for_tests`; monkeypatch `file_router.process` to instant fake; override `get_feature_service`/`get_usage_service`; POST `/api/v1/upload?session_id=&upload_id=` with `files={'file':(name, BytesIO, ctype)}`; assert `get_storage_client().download_bytes('sessions/{sid}/originals/{filename}')` is the **compressed** bytes (smaller) for a large image, and the **original** for a `<500KB` file and for a failure (monkeypatch `compression.compress_for_storage` to raise) — and that **OCR still ran on the original temp_path** (assert the fake's recorded input size == original). Assert filename/extension unchanged so cancel-purge still matches.
- `tests/api/v1/test_erphome_attach_compression.py` — respx-mock `POST {BASE_URL}/api/method/upload_file`; seed `test_erp_connection` + `authenticated_user_with_tenant`; assert posted multipart body is compressed for a large image, original on `compress=False` (provisioning path) and on fallback; assert **no GCS write** occurs; cross-tenant access asserts `== 404` strictly.
- Storage-failure fallback: monkeypatch `get_storage_client().upload_bytes` to raise `OSError` (mirror `tests/unit/compliance/brs/test_brs_gcs_upload.py:148-160`) → `_save_original_file` returns `None`, request still 200, OCR result returned. No live-GCS test (conftest pins buckets to `''` → LocalStorageClient at 209–213); real-GCS-in-DEV is a manual smoke (16.5), asserting the GCS client is actually selected, not silently degraded to local.
- Frontend: extend `tests/unit/lib/api/upload.test.ts` (30MB still asserted, new metadata fields parsed); add MSW handler if a new endpoint is registered (`tests/mocks/handlers.ts` → `server.ts`). Pure FE logic in a non-excluded `src/lib/utils/*.ts`.

### 16.5 Verification gates (run EVERY iteration)

Backend (from `erpsense-backend/`):
- `ruff format --check app/ tests/ && ruff check app/ tests/`
- `mypy app/ --ignore-missing-imports`
- `alembic heads` == exactly 1 (no new migration expected; assert anyway — CI checks it before tests).
- Fast lane: `pytest tests/unit/ -p no:warnings` (then per-step targeted file).
- Full gate: `pytest tests/ --ignore=tests/e2e --ignore=tests/perf --cov=app --cov-append --cov-report=xml --cov-fail-under=75` (asyncio_mode=auto → `async def test_*`, no `@pytest.mark.asyncio`; respx for HTTP, fakeredis for Redis; never mock internal services).

Frontend (from `erpsense-frontend/`):
- `npm run format && npm run lint && npm run type-check`
- `npm run test:coverage -- --testPathIgnorePatterns='/integration/|CAChatPanel|erp-global-search'` then the inline node 75% gate on `coverage/coverage-summary.json`; new files ≥85% (statements/lines ≥80, branches/functions ≥75 global). Playwright e2e/visual do NOT gate — local/documentation only.

Security checklist (every iteration, codified as `/security-review` + the 16.3 step-4 list): path traversal, BOLA/cross-tenant 404, decompression/pixel bomb, resource exhaustion bounds, no PII/bytes in logs, no signed-URL/IAM escalation, MIME/suffix consistency, multipart content-type handling.

Build/smoke (once per full pass): backend `python -c "import app.main"` (import-clean) + run app, POST a >1MB image to `/api/v1/upload?session_id=...` with **GCS configured in DEV** (`GCS_UPLOAD_BUCKET=erpsense-backend-uploads-dev`, ADC via `gcloud auth application-default login`) and assert: object exists in GCS, stored size < original, preview `GET /api/v1/upload/sessions/{sid}/originals/{filename}` returns 200 with correct suffix-derived content-type, OCR result unaffected; assert the selected client is `GCSStorageClient` (not silently Local). FE `npm run build`.

### 16.6 Done-criteria

- All 16.5 gates green on the full diff: ruff/mypy clean, single Alembic head, backend `pytest` ≥75% (with new compression modules measured, since they're outside the omit list), FE lint/type-check/jest ≥ thresholds (new ≥85%), FE build passes.
- `/code-review` (high) and `/security-review` report zero Must-Fix.
- Genie: large image/scanned-PDF stored compressed to GCS with same filename+extension; `<500KB` stored as-is; failure/regression stores original; OCR result byte-identical to pre-change (read from untouched `temp_path`); preview + cancel-purge unaffected.
- ERPHome: large attachment compressed before ERPNext `upload_file` POST; provisioning logo opted out (`compress=False`); 10MB cap unchanged; no GCS write on this surface.
- Observability: `compression.decision`/`compression.fallback`/`gcs.upload` structured events emitted (no f-strings, no PII, no duplicated request-scoped ids); optional registry counters export at `GET /api/v1/metrics`.
- No regression: existing upload, OCR routing, classifier verdicts (both call-sites), preview, cancel/purge, and all out-of-scope sinks (import-files, branding logo, FileStorageService) behave exactly as before.
- DEV-with-GCS manual smoke passes and confirms `GCSStorageClient` is actually selected.

### 16.x Checklist

- [ ] Re-grep all seam anchors on `feat/auto-compression` HEAD before editing (`upload.py:258/259/864/1025`, `_attachments.py:108/196/147/222`, `_steps_company.py:335`, `pdf_handler.py:302-308`, `library_gemini.py:589-618`).
- [ ] `config.py`: add `compression_enabled`, `compression_image_max_bytes` (200KB), `compression_image_max_dimension` (1200), `compression_scanned_pdf_max_bytes_per_page` (300KB), `compression_min_exempt_bytes` (500KB), `compression_pdf_rebuild_enabled` (False), `compression_jpeg_quality_start` (85) — no `os.getenv`.
- [ ] CREATE `handlers/_pdf_classifier.py` with `is_native_pdf(text, page_count, *, mode, threshold)` + `avg_chars_per_page`.
- [ ] Delegate `PDFHandler._has_sufficient_text` → `is_native_pdf(..., mode="total")` (TOTAL semantics unchanged).
- [ ] Delegate `library_gemini._extract_pdf_native` → `is_native_pdf(..., mode="average")` keeping the `raise FileProcessingException` control-flow (AVERAGE semantics unchanged).
- [ ] Verify `MIN_TEXT_LENGTH`/`MIN_PDF_TEXT_PER_PAGE`=50, 200 DPI, and routing are untouched.
- [ ] CREATE `file_processing/compression.py`: `CompressionResult`, `classify_for_compression`, `compress_image` (BytesIO copy, RGBA/P→RGB, downscale+quality stepdown, same suffix/MIME), `compress_scanned_pdf` (PyMuPDF 200 DPI → Pillow JPEG → reportlab rebuild), `compress_for_storage` (decision tree, `asyncio.to_thread`, never raises, fallback to original).
- [ ] Bake decisions: `<500KB`→skip; native→skip unless rebuild flag (off); compressed≥original→fallback; any exception→fallback; no retry.
- [ ] Never operate on `temp_path`; always a copy of in-memory `content`.
- [ ] CREATE `file_processing/compression_metrics.py` with `record_compression`/`record_gcs_upload` (event-name-first `logger.info`, extra carries sizes/pct/strategy/duration_ms/backend/surface, no PII, no request-scoped ids).
- [ ] OPTIONAL: `metrics.py` `_init_compression_metrics()` (decisions/fallback/gcs counters + bytes-saved/latency histograms); wire into `__post_init__` AND `reset_metrics()`.
- [ ] WIRE `upload.py:1025`/`_save_original_file:258` to persist `CompressionResult.data` via the SINGLE `upload_bytes` (overwrite bytes arg; no 2nd write; suffixer never re-fires).
- [ ] Replace f-string log at `upload.py:259` with `logger.info("upload.original_saved", extra={...})` + `record_compression(surface="genie")` + `record_gcs_upload`.
- [ ] Confirm OCR at `upload.py:1044` still reads untouched `temp_path`; persist still before unlink@1359; cancel-purge still matches filename.
- [ ] Populate `UploadResponse` (`upload.py:1111-1116`) with `original_size/stored_size/was_compressed`.
- [ ] WIRE `_attachments.py:upload_attachment` (before 108) + `upload_file_standalone` (before 196): `compress: bool=True`, compress COPY of `file_content`, feed multipart tuple, try/except→original, success envelope preserved.
- [ ] Extend `_attachments.py` logs (147/222) + `record_compression(surface="erphome")`; assert NO GCS write on ERPHome path.
- [ ] `_steps_company.py:335`: pass `compress=False` (logo opt-out).
- [ ] Confirm ERPHome cap stays 10MB (`_shared.py:111`); inherited callers (doc_social/file_proxy/expense×2/travel) unedited.
- [ ] FE `types/chat.ts`: add optional `original_size`/`stored_size`/`was_compressed`.
- [ ] FE `components/chat/file-preview-modal.tsx`: surface "compressed from X → Y"; add `chat:file_preview` i18n keys; keep blob-URL/`useAuthedFileUrl` pattern.
- [ ] FE: do NOT change `OCR_MAX_*` (already 30/32/5/10); do NOT lower to 10MB; do NOT touch `attachments-panel.tsx` 10MB (different sink).
- [ ] CREATE `tests/unit/file_processing/test_pdf_classifier.py` (both modes, mixed-density edge).
- [ ] CREATE `tests/unit/file_processing/test_compression.py` (image suffix/MIME, RGBA→RGB, exempt, native skip, scanned bound, regression fallback, exception fallback, never-raises).
- [ ] CREATE `tests/unit/core/test_metrics_compression.py` (registry exists + reset clears) — if registry metrics added.
- [ ] CREATE `tests/api/test_upload_compression.py` (compressed-to-GCS-stub, original for <500KB, fallback on raise, OCR-read-original assertion, filename unchanged) mirroring `test_upload_cancel.py`.
- [ ] CREATE `tests/api/v1/test_erphome_attach_compression.py` (respx ERPNext upload_file; compressed body; `compress=False` original; no-GCS; cross-tenant `==404`).
- [ ] Add storage-failure fallback test (monkeypatch `upload_bytes`→OSError → 200, OCR proceeds).
- [ ] Extend FE `tests/unit/lib/api/upload.test.ts` (30MB asserted, new metadata parsed); place new FE logic in non-excluded `lib/utils`; MSW handler wired if new endpoint registered.
- [ ] Run per-step gate: `ruff format --check && ruff check && mypy app/ && pytest <targeted>`.
- [ ] Run `/code-review` (high) on diff; resolve all Must-Fix.
- [ ] Run `/security-review` on branch; resolve all Must-Fix (path traversal, BOLA/404, pixel/zip bomb, resource bounds, no PII/bytes, no signed-URL/IAM, MIME consistency, multipart content-type).
- [ ] Full backend gate: `pytest tests/ --ignore=tests/e2e --ignore=tests/perf --cov=app --cov-fail-under=75` green; `alembic heads`==1.
- [ ] Full FE gate: `npm run format && lint && type-check && test:coverage` ≥ thresholds; `npm run build` passes.
- [ ] DEV manual smoke with GCS configured: large image → stored compressed, same suffix/content-type, preview 200, OCR unaffected, `GCSStorageClient` actually selected (not Local fallback).
- [ ] Final full-diff review loop (code + security) zero Must-Fix; confirm no regression in out-of-scope sinks (import-files, branding logo, FileStorageService).

---

## Master Implementation Checklist

> Single source of truth for the auto-compression + GCS preview + OCR/ERPHome-attach feature. Organized by WORKSTREAM. Every item names exact files. Re-grep line numbers before editing (upload.py is ~2004 lines; line numbers drift). Honor all 10 CORRECTED DECISIONS. Do not break existing functionality.

---

### Workstream 0: Pre-flight & Branch Hygiene
- [ ] Confirm on branch `feat/auto-compression` (HEAD ~`1d3f285c`); pull latest; do NOT work on default branch.
- [ ] `cd erpsense-backend && pip install -r requirements.txt` (verify PyMuPDF, python-docx, openpyxl, pandas, Pillow, reportlab all import; confirm pikepdf/img2pdf are NOT present and NOT relied upon).
- [ ] Re-grep and re-confirm every seam anchor line number in `erpsense-backend/app/api/v1/endpoints/upload.py` before any edit: `upload_file()` def, `content = await file.read()`, `NamedTemporaryFile`, `_save_original_file()` call + def, single `upload_bytes` write, `file_router.process(...)` OCR call, `finally` unlink, `serve_original_file()`, limit constants, `MIME_TYPE_MAP`.
- [ ] Re-confirm `alembic heads` == 1 baseline (so a later migration, if any, keeps single-head gate green).

---

### Workstream 1: Dependencies & Config
- [ ] `erpsense-backend/requirements.txt` — confirm no new deps needed (PyMuPDF/Pillow/reportlab present). Do NOT add pikepdf/img2pdf. If a pin bump is required, justify and keep `alembic heads` unaffected.
- [ ] `erpsense-backend/app/config.py` — add compression Settings fields (all snake_case, defaults preserve current behavior):
  - [ ] `compression_enabled: bool = True` (kill switch).
  - [ ] `compression_image_max_bytes: int = 200 * 1024` (200KB target).
  - [ ] `compression_image_max_dimension: int = 1200` (1200px target).
  - [ ] `compression_scanned_pdf_max_bytes_per_page: int = 300 * 1024` (300KB/page target).
  - [ ] `compression_min_exempt_bytes: int = 500 * 1024` (files < 500KB are exempt — never compressed).
  - [ ] `compression_scanned_pdf_dpi: int = 200` (match PDFHandler rasterize DPI).
  - [ ] `compression_jpeg_quality_start: int = 85`, `compression_jpeg_quality_min: int = 40` (iterative quality floor).
  - [ ] `compression_container_rebuild_enabled: bool = False` (OPTIONAL DOCX/XLSX/native-PDF rebuild, default OFF / flagged).
  - [ ] `compression_erphome_enabled: bool = True` (per-surface toggle for ERPHome adapter hook).
  - [ ] Each field documented with a comment; do NOT use `os.getenv` outside config.py (R03).
- [ ] `erpsense-backend/env.example` — add the new env vars (commented) with dev-safe defaults; document that real GCS in dev uses ADC (`gcloud auth application-default login`) and bucket `erpsense-backend-uploads-dev`.
- [ ] Limit reconciliation (DECISION 8): confirm `upload.py` Genie constants stay `MAX_FILE_SIZE=30MB / MAX_TOTAL_SIZE=32MB / MAX_FILES_PER_SESSION=5 / MAX_PAGES_PER_PDF=10`; do NOT change to 10MB. `MIN_EXEMPT=500KB` is a compression gate only, NOT a rejection limit.
- [ ] Confirm ERPHome `_MAX_ATTACHMENT_SIZE_BYTES` in `erpsense-backend/app/api/v1/endpoints/erp_documents/_shared.py:111` stays 10MB (DECISION: ERPHome cap is genuinely 10MB, NOT 30MB) unless product explicitly requests otherwise — document the divergence.
- [ ] Frontend constants: confirm `erpsense-frontend/src/lib/constants.ts:1893-1896` already = 30MB/32MB/5/10 (no-op reconciliation; do NOT lower to 10MB).

---

### Workstream 2: GCS & Storage Layer
- [ ] `erpsense-backend/app/services/storage_client.py` — verify (no behavior change) `get_storage_client()` selects GCS when `gcs_bucket_name or gcs_upload_bucket` set, else `LocalStorageClient` with WARNING; `@lru_cache(maxsize=1)` intact.
- [ ] Do NOT switch preview to signed URLs (`_generate_signed_url_sync` is dead AND needs `iam.serviceAccounts.signBlob` not granted) — keep `get_serve_response` byte proxy.
- [ ] Do NOT add a 2nd `upload_bytes` write anywhere (dup-filename suffixer would rename). Single write only, at the existing `_save_original_file` sink.
- [ ] Respect least-privilege IAM (objectCreator + objectViewer only): never overwrite an existing object path in prod (suffixer/fresh paths only); never call `delete()` on a path the SA can't delete (cancel-purge already uses listing+match — leave it).
- [ ] Keep storage layer transport-only: do NOT add compression/event-name logging inside `storage_client.py` (emit structured logs at the caller seam). The existing %-style DEBUG logs may stay (out of scope to refactor) — but do not extend them.
- [ ] DEV-env verification note: assert GCS client actually selected when bucket configured (smoke), not silent fallback to Local.

---

### Workstream 3: Compression Engine (module structure & interfaces)
- [ ] CREATE `erpsense-backend/app/services/file_processing/compression.py` (NEW, NON-omitted from coverage so it counts toward 75%):
  - [ ] Module-level `logger = logging.getLogger(__name__)`.
  - [ ] `@dataclass CompressionResult`: `data: bytes`, `original_size: int`, `compressed_size: int`, `strategy: str`, `was_compressed: bool`, `fallback_reason: str | None`, `duration_ms: float`.
  - [ ] `def classify_for_compression(content: bytes, filename: str, content_type: str | None) -> str` returning one of `"image" | "scanned_pdf" | "native_pdf" | "native_other" | "exempt_small"`.
  - [ ] `async def compress_for_storage(content: bytes, filename: str, content_type: str | None, settings) -> CompressionResult` — top-level orchestrator (copy of bytes only; never a path).
  - [ ] Enforce DECISION 8 MIN exempt: if `len(content) < compression_min_exempt_bytes` → return as-is `was_compressed=False, strategy="exempt_small"`.
  - [ ] Enforce DECISION 9 fallback: on ANY exception OR `compressed_size >= original_size` → return ORIGINAL bytes, `was_compressed=False`, `fallback_reason` set, log WARNING, never raise.
  - [ ] DECISION 1/2/3/4: operate on in-memory copy, preserve filename+extension exactly (output bytes only; caller keeps same name), never touch temp_path, single persist downstream.
  - [ ] DECISION 6: parallelism within request = `asyncio.gather` + `asyncio.to_thread` for CPU-bound Pillow/PyMuPDF work; no new worker/infra.
- [ ] CREATE `erpsense-backend/app/services/file_processing/handlers/_pdf_classifier.py` (NEW, shared helper):
  - [ ] `def avg_chars_per_page(text: str, page_count: int) -> float`.
  - [ ] `def is_native_pdf(text: str, page_count: int, *, mode: str) -> bool` where `mode="total"` (PDFHandler semantics: `len(text) >= 50*page_count`) and `mode="average"` (library_gemini semantics: `avg < 50`). Parameterize so NEITHER call site's behavior changes.
  - [ ] `MIN_TEXT_LENGTH = 50` constant (single source); document the two semantics divergence.
- [ ] WIRE delegate (preserve exact existing behavior + control flow):
  - [ ] `erpsense-backend/app/services/file_processing/handlers/pdf_handler.py:_has_sufficient_text` → return `is_native_pdf(text, page_count, mode="total")` (keep method, keep BRANCH-to-OCR control flow; do NOT change MIN_TEXT_LENGTH value or 200 DPI).
  - [ ] `erpsense-backend/app/services/file_processing/ocr/library_gemini.py:_extract_pdf_native` (~589-618) → use helper with `mode="average"` but KEEP the `raise FileProcessingException` control flow (do NOT collapse raise-vs-branch).
- [ ] Module size: if `compression.py` approaches 500 lines, split image/PDF/rebuild into submodules under `app/services/file_processing/compressors/`.

---

### Workstream 4: Image Compressor
- [ ] In `compression.py` (or `compressors/image.py`): `async def compress_image(content: bytes, filename: str, content_type: str | None, settings) -> CompressionResult`:
  - [ ] Open via `PIL.Image.open(BytesIO(content))` — NEVER from temp_path.
  - [ ] Replicate `ImageHandler._resize_if_needed` RGBA/P→RGB conversion (image_handler.py:159-161) for JPEG compatibility; preserve transparency for PNG when not converting to JPEG.
  - [ ] Resize so longest side ≤ `compression_image_max_dimension` (1200px), preserving aspect ratio; skip resize if already smaller.
  - [ ] Iteratively re-encode (JPEG quality `85 → 40` step-down, or PNG `optimize=True`) until `≤ compression_image_max_bytes` (200KB) or quality floor reached.
  - [ ] CRITICAL: preserve original extension + matching content_type (recompressed JPEG stays `.jpg`/`image/jpeg`; PNG stays `.png`) — preview/serve content-type is suffix-derived (DECISION 4).
  - [ ] Handle multi-frame GIF/animated WEBP/TIFF: if animated, leave as-is (fallback) rather than flatten to first frame.
  - [ ] Run CPU-bound encode via `asyncio.to_thread`.
  - [ ] Fallback if encoded ≥ original or any error.
- [ ] Edge cases covered: corrupt/truncated image bytes; zero-byte; CMYK JPEG; 16-bit PNG; tiny image already under target; non-image masquerading as image extension.

---

### Workstream 5: Scanned PDF Compressor
- [ ] In `compression.py` (or `compressors/pdf.py`): `async def compress_scanned_pdf(content: bytes, filename: str, settings) -> CompressionResult`:
  - [ ] Open via `fitz.open(stream=content, filetype="pdf")` — NEVER temp_path.
  - [ ] Classify native vs scanned using `_pdf_classifier.is_native_pdf` (mode matching the active runtime path — verify under `library_gemini`/`file_router.process` which classifier actually decides routing; default to `mode="average"` to match active OCR). If native → skip (strategy `native_pdf`, not compressed).
  - [ ] For scanned: rasterize each page at `compression_scanned_pdf_dpi` (200) via `page.get_pixmap(matrix=fitz.Matrix(dpi/72, dpi/72))` (mirror PDFHandler._pdf_to_images pattern).
  - [ ] Compress each page image (JPEG) toward `compression_scanned_pdf_max_bytes_per_page` (300KB/page).
  - [ ] Rebuild PDF from compressed page images using PyMuPDF (`fitz` insert image pages) and/or reportlab (NO pikepdf/img2pdf). Keep `.pdf` extension + `application/pdf` content_type.
  - [ ] Parallelize per-page rasterize+encode with `asyncio.gather(*[asyncio.to_thread(...) ...])` (DECISION 6).
  - [ ] Respect `MAX_PAGES_PER_PDF=10` already validated upstream; do not re-rasterize beyond.
  - [ ] Fallback if rebuilt PDF ≥ original or any error.
- [ ] Edge cases: encrypted/password PDF (skip, fallback); PDF with embedded vector text on some pages + scanned on others (mixed-density — honor classifier verdict, don't corrupt native text pages); single-page; broken/corrupt PDF stream; PDF already small.

---

### Workstream 6: Optional Container Rebuild (DOCX / XLSX / native-PDF) [FLAGGED, default OFF]
- [ ] Gate ALL of this behind `compression_container_rebuild_enabled` (default False). When off, native PDF/DOCX/XLSX/CSV pass through as-is (DECISION 7).
- [ ] In `compressors/container.py` (NEW): recompress ZIP-container office files (DOCX/XLSX are ZIP) at higher deflate level and/or re-zip media; for native PDF, lossless object stream rebuild via PyMuPDF `save(garbage=4, deflate=True)`.
- [ ] Validity guard: after rebuild, re-open the file to assert it still parses (python-docx / openpyxl / fitz) before accepting; otherwise fallback to original.
- [ ] NEVER touch CSV/TXT/MD (no container to rebuild) — always as-is.
- [ ] Fallback if rebuilt ≥ original, parse-check fails, or any error.
- [ ] Document that this path is OFF by default and experimental; ensure tests cover both flag states.

---

### Workstream 7: Genie OCR-Flow Integration (upload.py)
- [ ] `erpsense-backend/app/api/v1/endpoints/upload.py` — compute compressed bytes AFTER `content = await file.read()` (~864) AND AFTER PDF page-count validation block ends (~1019), BEFORE the `_save_original_file` call (~1024-1025):
  - [ ] `compressed = await compression.compress_for_storage(content, original_filename, file.content_type, settings)` guarded by `if settings.compression_enabled` (else passthrough result).
  - [ ] Substitute `compressed.data` for `content` in the `_save_original_file(session_id, original_filename, compressed.data)` call — THE single wiring point.
  - [ ] OCR call at ~1044 MUST keep reading the untouched `temp_path` (original) — do NOT feed compressed bytes to OCR (DECISION 2: OCR reads original, storage saves compressed).
  - [ ] Do NOT reorder: persist already runs before OCR and before `finally` unlink — deferral satisfied; just swap the bytes arg.
  - [ ] Compression failure/exception must NEVER block OCR or raise (DECISION 9) — `_save_original_file` stays non-blocking (returns None on failure).
  - [ ] Guard early-validation raises between read(864) and save(1024) (cumulative-size, double size-check, PDF page limit) — they short-circuit before save, so no compression runs there (fine).
- [ ] `_save_original_file()` (~225-264): keep single `upload_bytes` write at ~258; keep filename+extension; keep dup-suffixer untouched. Do NOT add a 2nd write.
- [ ] Replace the f-string banner log (~259 `[UPLOAD] Saved original file...` and `'='*60` banners) with structured `logger.info("upload.original_saved", extra={...})` (see Logging workstream). Bring this seam into CLAUDE.md compliance.
- [ ] Verify cancel/purge path (`cancel_upload` + `_purge_upload_artifacts`) still matches `{stem}.{ext}`/`{stem}_{counter}.{ext}` — unchanged because filename preserved.
- [ ] Optionally extend `UploadResponse` / extraction metadata with `original_size`, `stored_size`, `was_compressed`, `strategy` (additive, for FE transparency).

---

### Workstream 8: ERPHome Attachment + OCR-Attach Entity Integration
- [ ] `erpsense-backend/app/adapters/erpnext/_attachments.py` — adapter-level compression hook (covers all 5 in-scope callers in one place):
  - [ ] `AttachmentsMixin.upload_attachment` (~77-168): add `compress: bool = True` kwarg; compress a COPY of `file_content` immediately BEFORE the multipart build (~108) when `compress and settings.compression_erphome_enabled`.
  - [ ] `AttachmentsMixin.upload_file_standalone` (~170-233): same hook before multipart build (~196); add `compress: bool = True`.
  - [ ] Keep content_type as the multipart tuple's 3rd element (NOT a request header); keep stripping client Content-Type so httpx sets the multipart boundary; keep multisite Host header.
  - [ ] Wrap compression in try/except so failure falls back to original bytes and still returns the normal `{success: ...}` envelope (never bubble unhandled).
  - [ ] NO OCR, NO GCS on this path (DECISION 5: ERPHome → ERPNext only). Do NOT import file_router/library_gemini here.
- [ ] Opt-out: `erpsense-backend/app/services/erpnext_provisioning/_steps_company.py` (~335) logo push → pass `compress=False` (already-optimized branding logo).
- [ ] Verify the 4 in-scope service/route callers inherit compression with no edits: `doc_social.py:upload_attachment` (~335), `file_proxy.py:upload_erp_file` (~189), `employee_expense_service.py` (~617, ~651), `employee_travel_service.py` (~750).
- [ ] OCR-Attach Entity feature: if scope includes attaching an OCR'd entity result to an ERP doc, confirm it routes ONLY through the adapter (ERPNext sink) — no GCS; OCR (if any) reuses the Genie pipeline but storage stays per-surface. Document the exact route/service.
- [ ] Reconcile size cap copy: keep `_MAX_ATTACHMENT_SIZE_BYTES=10MB` 413 messages on doc_social + file_proxy consistent.

---

### Workstream 9: Preview / Download & On-Demand Unpack
- [ ] `erpsense-backend/app/api/v1/endpoints/upload.py:serve_original_file()` (~1927-2004): no change to media_type derivation — relies on `MIME_TYPE_MAP[suffix]` (suffix-derived). Compression preserved extension, so preview content-type stays correct. Re-verify auth (header or `?token`), UUID session check, session-owner verify, filename sanitize all intact.
- [ ] Confirm `get_serve_response` byte proxy serves compressed bytes transparently (same path/filename).
- [ ] No on-demand "unpack" needed for lossy image/scanned-PDF compression (compressed IS the stored artifact). For OPTIONAL container rebuild, the rebuilt file is still a valid container — no unpack needed. Document that there is no separate "original" retained (lossy) — confirm product accepts storing compressed-only; if originals must be retained, that is net-new infra and explicitly OUT of this pass.
- [ ] Frontend preview (no functional change required):
  - [ ] `erpsense-frontend/src/lib/hooks/chat-utils.ts:buildPreviewUrl` + `EXT_TO_MIME` — key off filename+extension (preserved) — verify still resolves.
  - [ ] `erpsense-frontend/src/components/chat/file-preview-modal.tsx` — `useAuthedFileUrl` blob path unchanged.
  - [ ] ERPHome `erpsense-frontend/src/components/erp/file-preview-modal.tsx` — content-type from proxy header, robust to changes.

---

### Workstream 10: Logging & Metrics (every GCS op + compression + fallback)
- [ ] Logging idiom (mirror `feature_service.py` / `diagnostic_metrics.py`, NOT `upload.py`/`storage_client.py`): module-level `logger = logging.getLogger(__name__)`; first arg = STABLE EVENT NAME; all data in `extra={}`; NO f-strings; NO request-scoped ids in extra (auto-injected).
- [ ] CREATE `erpsense-backend/app/services/file_processing/compression_metrics.py` (NEW, log-based, mirrors `diagnostic_metrics.py`):
  - [ ] `record_compression_decision(strategy, decision, original_size, compressed_size, saved_pct, duration_ms, ...)` → `logger.info("compression.decision", extra={...})`.
  - [ ] `record_compression_fallback(reason, original_size, strategy, ...)` → `logger.warning("compression.fallback", extra={...})`.
  - [ ] `record_gcs_upload(backend, status, storage_path, size_bytes, duration_ms)` → `logger.info("gcs.upload", extra={...})`.
  - [ ] Safe fields only: filename, content_type, original_size_bytes, compressed_size_bytes, saved_bytes, saved_pct, strategy, decision, fallback_reason, storage_path, storage_backend, duration_ms. NEVER file bytes/tokens/PII.
- [ ] OPTIONAL registry metrics in `erpsense-backend/app/core/metrics.py`:
  - [ ] Add `MetricsRegistry._init_compression_metrics()` with `Counter compression_decisions_total{strategy,decision}`, `Counter compression_fallback_total{reason}`, `Counter gcs_upload_total{backend,status}`, `Histogram compression_bytes_saved`, `Histogram gcs_upload_latency_seconds`.
  - [ ] Call it in BOTH `__post_init__` AND `reset_metrics()` (or pytest leaks counter state).
  - [ ] Keep label cardinality LOW; NEVER use tenant_id as a label (goes in log extra only).
  - [ ] Surfaces automatically at `GET /api/v1/metrics` (observability.py export).
- [ ] Call sites:
  - [ ] Genie: in `upload.py` at the `_save_original_file` seam — emit `compression.decision` + `gcs.upload` + replace banner with `upload.original_saved`. Backend label via `type(get_storage_client()).__name__`.
  - [ ] ERPHome: in `_attachments.py` — extend existing `logger.info("Attachment uploaded", extra={size:...})` (~147/~222) with `original_size`/`compressed_size`/`was_compressed` (no GCS event here — ERPNext sink).
- [ ] Levels: INFO = decision/save OK; WARNING = fallback (failed or compressed≥original); never `logger.error` before raising (No-double-logging rule).
- [ ] If a new contextvar is introduced (likely unnecessary): append to `_ALL_CONTEXT_VARS` and use set/reset-in-finally. Prefer NO new contextvar.

---

### Workstream 11: Security Review & Hardening
- [ ] Path/filename: keep using existing filename sanitizer in `_save_original_file` (~241-244) and `serve_original_file` (~1975); never derive storage path from raw user filename without sanitize; preserve extension only (no traversal).
- [ ] No SSRF/host injection on ERPHome adapter: keep multisite Host header logic intact; do not let compression alter request headers.
- [ ] Do not log file content, tokens, GSTIN, or PII anywhere (rely on `redact_dict` only as backstop, not primary).
- [ ] Resource exhaustion / decompression-bomb defense:
  - [ ] Set `PIL.Image.MAX_IMAGE_PIXELS` guard (or catch `Image.DecompressionBombError`) before decoding; fallback on bomb.
  - [ ] Cap PDF page processing at existing `MAX_PAGES_PER_PDF` (10); do not rasterize unbounded.
  - [ ] Bound per-image dimension processing (already validated upstream ≤30MB; reject/fallback on absurd pixel counts).
- [ ] CPU/timeout: compression runs inside the 300s request OCR window; ensure `asyncio.to_thread` work cannot wedge the event loop; consider a soft internal time budget → fallback to original if exceeded.
- [ ] Tenant isolation: storage path stays `sessions/{sid}/originals/...`; session-owner verify in serve unchanged; cross-tenant must 404.
- [ ] IAM: no new permissions required (objectCreator+objectViewer cover upload/download/exists); confirm no `delete()`/overwrite on existing object introduced.
- [ ] Adapter compression must NOT change `is_private` semantics or leak private files to public.

---

### Workstream 12: Edge Cases & Robustness Catalog
- [ ] File < 500KB → exempt, stored as-is, `was_compressed=False` (DECISION 8).
- [ ] Compressed ≥ original → store ORIGINAL, fallback logged, no retry (DECISION 9).
- [ ] Compression raises (corrupt bytes, encrypted PDF, animated GIF, bomb) → store ORIGINAL, never block OCR/attach (DECISION 9).
- [ ] No `session_id` on Genie upload → `_save_original_file` not called (no persist) — compression still skipped/irrelevant; verify no crash.
- [ ] Native PDF / DOCX / XLSX / CSV / TXT / MD with rebuild flag OFF → pass through unchanged.
- [ ] Image already ≤ targets → minimal/no re-encode, prefer original if not smaller.
- [ ] Scanned PDF mixed native+scanned pages → classifier verdict honored; native-text pages not corrupted.
- [ ] Filename with unicode/spaces/`..`/path separators → sanitized; extension preserved.
- [ ] Zero-byte upload → fallback, no crash.
- [ ] Non-image bytes with image extension (and vice versa) → classify safely, fallback.
- [ ] ERPHome provisioning logo (`compress=False`) → NOT compressed.
- [ ] ERPHome employee_expense/travel list-of-tuples reuse → compress a COPY, never mutate shared refs.
- [ ] LocalStorageClient (dev w/o GCS) path → compression + persist still works.
- [ ] GCS configured (dev/prod) → compression + single upload_bytes write to GCS works; no overwrite of existing path.
- [ ] Concurrent uploads in one session → dup-suffixer + cancel-purge still consistent.
- [ ] Very large multi-page scanned PDF near 30MB / 10 pages → bounded, within time budget or fallback.

---

### Workstream 13: Backend Tests
> Keep coverage-bearing code OUT of `pyproject.toml [tool.coverage.run].omit` (upload.py + OCR engines are omitted — that is why `compression.py`/`compression_metrics.py`/`_pdf_classifier.py` are NEW non-omitted modules). Gate = 75% on merged dataset. `asyncio_mode=auto` (no `@pytest.mark.asyncio`). No live GCS (conftest forces LocalStorageClient).
- [ ] CREATE `erpsense-backend/tests/unit/file_processing/test_compression.py`:
  - [ ] `classify_for_compression` for image / scanned_pdf / native_pdf / native_other / exempt_small.
  - [ ] `compress_image`: shrinks large PNG/JPEG ≤200KB & ≤1200px; preserves extension/content-type; RGBA→RGB; already-small no-op; corrupt bytes → fallback; animated GIF → fallback; bomb → fallback.
  - [ ] `compress_scanned_pdf`: scanned PDF shrinks ≤300KB/page, stays `.pdf`; native PDF skipped; encrypted → fallback; corrupt → fallback; per-page parallelism works.
  - [ ] MIN-exempt (<500KB) returns original `was_compressed=False`.
  - [ ] Fallback when compressed ≥ original.
- [ ] CREATE `erpsense-backend/tests/unit/file_processing/test_pdf_classifier.py`:
  - [ ] `is_native_pdf(mode="total")` matches old PDFHandler semantics on boundary cases.
  - [ ] `is_native_pdf(mode="average")` matches old library_gemini semantics; mixed-density divergence preserved.
- [ ] CREATE `erpsense-backend/tests/unit/file_processing/test_compression_metrics.py`: event names + extra fields emitted; no PII; fallback → WARNING.
- [ ] CREATE/EXTEND `erpsense-backend/tests/api/test_upload_compression.py` (mirror `tests/api/test_upload_cancel.py`):
  - [ ] Monkeypatch `file_router.process` to instant fake; override `get_feature_service`/`get_usage_service`; use `reset_upload_registry_for_tests`.
  - [ ] POST `/api/v1/upload?session_id=&upload_id=` with large image → assert stored bytes (via `get_storage_client().download_bytes('sessions/{sid}/originals/{filename}')`) are SMALLER than uploaded and filename/ext preserved.
  - [ ] Assert OCR received the ORIGINAL temp_path (fake captures bytes/size) — compressed not fed to OCR.
  - [ ] `compression_enabled=False` → stored bytes == original.
  - [ ] Compression-failure monkeypatch (`compress_for_storage` raises internally) → original stored, OCR still runs, 200 response.
  - [ ] No `session_id` → no persist, no crash.
  - [ ] Cancel after compressed upload → `_purge_upload_artifacts` still deletes (filename match).
- [ ] CREATE `erpsense-backend/tests/api/v1/test_erphome_attach_compression.py` (respx mocking ERPNext `POST {BASE_URL}/api/method/upload_file`):
  - [ ] `upload_attachment` compresses large image before POST (assert posted multipart bytes smaller); content_type preserved.
  - [ ] `compress=False` (provisioning logo path) → original bytes posted.
  - [ ] Compression failure → original bytes posted, success envelope returned.
  - [ ] `upload_file_standalone` same coverage.
  - [ ] Cross-tenant attach must assert `== 404` strictly (BOLA).
- [ ] EXTEND `erpsense-backend/tests/unit/core/test_storage_client.py` if needed: round-trip of compressed bytes; monkeypatch `upload_bytes` → raise OSError → `_save_original_file` returns None non-blocking (mirror `test_brs_gcs_upload.py` failure idiom).
- [ ] EXTEND `erpsense-backend/tests/unit/core/test_metrics.py` (if registry metrics added): `_init_compression_metrics` present after `reset_metrics()`; counters/histograms increment.
- [ ] Delegate-behavior regression: tests confirming `pdf_handler._has_sufficient_text` and `library_gemini._extract_pdf_native` outputs UNCHANGED after refactor (snapshot old vs new on boundary inputs).
- [ ] Container-rebuild tests (flag ON and OFF): DOCX/XLSX/native-PDF rebuild produces valid parseable file or falls back; CSV/TXT untouched.

---

### Workstream 14: Frontend Tests
> Place coverage-bearing logic in NON-excluded paths (`src/lib/utils/*`, `src/lib/api/*`); component dirs are excluded. New code ≥85%; gate 75% branches/functions, 80% lines/statements.
- [ ] EXTEND `erpsense-frontend/tests/unit/lib/api/upload.test.ts`: assert 30MB/32MB/5/10 limits unchanged; if `UploadResponse` extended with compression fields, assert they pass through.
- [ ] If `types/chat.ts` `UploadResponse`/`ExtractionMetadata` extended with `original_size`/`stored_size`/`was_compressed`: add type-shape tests + any util that formats "compressed from X".
- [ ] If a compression-transparency util added (e.g. `formatCompressionSavings`): CREATE `erpsense-frontend/tests/unit/lib/utils/<name>.test.ts` (≥85%).
- [ ] MSW: if any new/changed endpoint response shape, update `erpsense-frontend/tests/mocks/handlers.ts` (or `handlers/*.ts`) wired via `tests/mocks/server.ts`; mock at API boundary only.
- [ ] Confirm preview tests (if present) still pass given filename+extension preserved (no rename).
- [ ] If ERPHome client-side size guard added for parity (optional): test in `attachment-upload.tsx`/`attach-field-input.tsx` area via a non-excluded util; i18n the copy (no hardcoded English).
- [ ] If `fetchFileBlob` inline `/erp/file` endpoint is touched: register it in `API_ENDPOINTS.ERP_DOCS` and update tests (no inline endpoints rule).

---

### Workstream 15: Dev-Environment, Rollout, Backward-Compat & Kill Switch
- [ ] Kill switch verified: `compression_enabled=False` → exact pre-feature behavior (original bytes stored on both surfaces); `compression_erphome_enabled=False` → ERPHome passthrough; `compression_container_rebuild_enabled=False` (default) → native files untouched.
- [ ] Backward-compat: existing previews, cancel/purge, OCR all unchanged; no migration required unless metadata persisted (if a DB column added, create Alembic migration keeping `alembic heads`==1).
- [ ] Dev GCS smoke (manual): with ADC (`gcloud auth application-default login`) and bucket `erpsense-backend-uploads-dev`, upload an image via Genie → confirm compressed object exists in GCS, preview renders, content-type correct. Assert GCS client (not Local) was selected.
- [ ] Verify env mismatch note: local `.env` `GCS_BUCKET_NAME=erpsense-ocr-dev` may not exist out-of-band — prefer/confirm `GCS_UPLOAD_BUCKET=erpsense-backend-uploads-dev` is the working dev bucket; document.
- [ ] Platform-side (OUT of backend scope, mention only): log-based metric + alert definitions for `compression.fallback`/`gcs.upload` in `erpsense-platform/terraform/modules/monitoring`.
- [ ] No new infra/worker introduced (DECISION 6: synchronous inline only).

---

### Workstream 16: Verification Gates (run after each meaningful change)
- [ ] Backend fast loop: `cd erpsense-backend && ruff format --check app/ tests/ && ruff check app/ tests/ && mypy app/ --ignore-missing-imports`.
- [ ] Backend targeted: `pytest tests/unit/file_processing -p no:warnings && pytest tests/api/test_upload_compression.py tests/api/v1/test_erphome_attach_compression.py -p no:warnings`.
- [ ] Backend full gate: `pytest tests/ --cov=app --cov-fail-under=75 -q` (or `scripts/run_ci_locally.sh`); confirm Alembic single head.
- [ ] Frontend: `cd erpsense-frontend && npm run format && npm run lint && npm run type-check && npm run test:coverage -- --testPathIgnorePatterns=/integration/` (lines/statements ≥80, branches/functions ≥75, new files ≥85).
- [ ] `/preflight` passes on both repos.

---

### Repeatable gates (run EVERY pass — repeat until clean)
- [ ] Run `/code-review` (effort high) on the current diff; triage every finding; fix or justify; re-run until no new actionable findings.
- [ ] Run `/security-review` on the pending diff; address path-traversal, decompression-bomb, tenant-isolation, IAM-overwrite, PII-in-logs, header/SSRF, private-file-leak; re-run until clean.
- [ ] Run `/simplify` on changed files for reuse/altitude/efficiency; ensure no file exceeds 500 lines (split per `global-modularity`); ensure single-responsibility per module.
- [ ] Re-grep all `upload.py` seam anchors to confirm line numbers/symbols didn't drift; confirm STILL exactly ONE `upload_bytes` write for originals.
- [ ] Confirm NO new f-string log messages introduced; all new logs are event-name + `extra={}`; no request-scoped ids duplicated in extra.
- [ ] Confirm DECISIONS 1-10 still hold: copy-of-bytes (not temp_path); OCR reads original / storage saves compressed; single deferred write; filename+ext preserved; per-surface storage (Genie→GCS, ERPHome→ERPNext); synchronous inline (gather+to_thread); compress only images + scanned PDF (rebuild flag OFF by default); MIN-500KB exempt + MAX-30MB Genie; fallback-to-original on failure/regression (logged, no retry); works in DEV with GCS.
- [ ] Confirm coverage-bearing code is NOT in `[tool.coverage.run].omit`; coverage gate (BE 75%, FE 80/75, new ≥85) still met.
- [ ] Re-run full backend + frontend verification gates (Workstream 16); confirm no existing tests broken (regression sweep).
- [ ] Confirm kill switches (`compression_enabled`, `compression_erphome_enabled`, `compression_container_rebuild_enabled`) each reproduce exact pre-feature behavior when off.
- [ ] Diff review: confirm no unrelated files changed; no secrets/keys committed; `env.example` updated, no real `.env` committed.

---

## Appendix — References
- Architecture diagram: `auto_compresion/auto_compression_pipeline.excalidraw`
- OCR active flow (source of truth): `auto_compresion/ocr_active_flow_complete.md`
- Spec: `auto_compresion/Auto_Compression_Files.docx`
- Reference flows: `full_flow.png`, `compression_pipeline.png`, `internal_rebuild.png`, `internal_copmplete.png`
