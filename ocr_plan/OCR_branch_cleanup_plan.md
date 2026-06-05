# OCR Branch Cleanup Plan

## Context

The `ocr_backend` (backend) and `ocr_implementation` (frontend) branches were meant for OCR/file-upload only, but they became "kitchen sink" branches with 6+ unrelated features mixed in via mega-commits. Someone has since partially cleaned `ocr_backend` (removed venv312, ocr_output, fixed mypy, fixed pre-commit), but non-OCR feature code is still mixed in on both branches.

Three features have already been extracted onto clean branches:
- `guardrails_clean` (PR #22) -- enhanced guardrails
- `smart_suggestion_clean` (PR #21) -- smart suggestions + response buttons
- `smart_suggestion_frontend_clean` (PR #18) -- suggestion buttons UI + loading messages

**Goal:** Create clean `ocr_clean` (backend) and `ocr_frontend_clean` (frontend) branches from current `main` containing ONLY OCR/file-upload code.

**Why not clean existing branches?** The commits are mega-commits that atomically mix OCR with guardrails, smart suggestions, greeting removal, test deletions, etc. You cannot revert parts of them. Cherry-picking OCR-only files onto a fresh branch is the only reliable approach.

---

## Phase 0: Pull Latest and Analyze Both OCR Branches

### Step 1: Fetch and switch to ocr_backend

```bash
cd erpsense-backend
git fetch origin
git checkout ocr_backend
git pull origin ocr_backend
```

### Step 2: Fetch and switch to ocr_implementation

```bash
cd erpsense-frontend
git fetch origin
git checkout ocr_implementation
git pull origin ocr_implementation
```

### Step 3: Analyze what is on each branch vs main

```bash
# Backend -- see all files different from main
cd erpsense-backend
git diff --name-only origin/main..ocr_backend
git log --oneline origin/main..ocr_backend

# Frontend -- see all files different from main
cd erpsense-frontend
git diff --name-only origin/main..ocr_implementation
git log --oneline origin/main..ocr_implementation
```

### What someone already fixed on ocr_backend (recent cleanup commits)

| Commit    | What was fixed                                            |
|-----------|-----------------------------------------------------------|
| `cd0a8fb` | Removed `venv312/` and `ocr_output/` from git tracking   |
| `5c16861` | Excluded `ocr_output/` and `venv312/` from pre-commit    |
| `6bb64d9` | Fixed 134 mypy errors, fixed tests, added coverage        |
| `eaa1144` | Added document perspective mapping for PO-to-Sales Order  |

### What is still contaminated (non-OCR code still present)

**Backend `ocr_backend` -- 7 commits ahead of main, 64 files changed:**

| Non-OCR Contamination  | Files                                    | Status                                                |
|------------------------|------------------------------------------|-------------------------------------------------------|
| Smart Suggestions      | `smart_suggestions.py`                   | Already extracted to `smart_suggestion_clean`          |
| Response Buttons       | `response_buttons.py`, test              | Already extracted to `smart_suggestion_clean`          |
| Guardrails rewrite     | `guardrails.py`, 2 test files            | Already extracted to `guardrails_clean`                |
| Supervisor refactor    | Mixed into `supervisor.py`               | OCR attachment plumbing tangled with non-OCR removals  |
| Warmup rewrite         | `app/core/warmup.py` (new)               | Separate PR                                           |
| Config removals        | SMTP/password-reset settings removed     | Non-OCR                                               |
| Prompt changes         | Non-OCR simplifications mixed with OCR   | Mixed                                                 |
| pyproject.toml         | Coverage, tool config changes            | Mixed                                                 |

**Frontend `ocr_implementation` -- 8 commits ahead of main, 54 files changed:**

| Non-OCR Contamination     | Files                                   | Status                                                        |
|---------------------------|-----------------------------------------|---------------------------------------------------------------|
| Smart Suggestions UI      | `suggested-buttons.tsx`                 | Already extracted to `smart_suggestion_frontend_clean`         |
| Avatar/Settings removal   | 4 files deleted, settings page gutted   | Separate PR (intentional)                                     |
| Auth forms restyling      | 2 form files restyled                   | Separate PR (intentional)                                     |
| Thinking indicator rewrite| `thinking-indicator.tsx`                | Non-OCR                                                       |
| Toast store               | `toast-store.ts` (new)                  | Non-OCR (upload code imports it -- will use existing `use-toast` hook instead) |
| Sidebar changes           | Avatar removal, delete confirm removed  | Non-OCR                                                       |
| Auth/session changes      | `use-auth.ts`, `auth-store.ts`, `auth.ts` | Non-OCR                                                    |
| 11 test files DELETED     | 2,000+ lines of tests gone              | Separate PR (intentional)                                     |

### Conclusion

The non-OCR code is baked into mega-commits (e.g. `314be1a` = "Add smart suggestions, guardrails, file preview serving, and enhanced OCR pipeline" -- one commit with 4 features). You cannot `git revert` parts of a commit.

**Resolution:** Cherry-pick only OCR files onto new clean branches from `main`.

---

## Full File-by-File Analysis: ocr_backend (64 files changed vs main)

### OCR-only files -- TAKE (~40 files)

- `app/services/file_processing/` -- entire directory (35 files: handlers, classifiers, OCR engines)
- `app/api/v1/endpoints/upload.py` (1,551 lines) -- file upload API
- `app/api/v1/endpoints/ocr_config.py` (258 lines) -- OCR config API
- `app/config/ocr_features.json` -- OCR feature flags
- `app/api/deps.py` -- `get_current_user_optional()` for file preview auth
- `tests/api/test_upload_session_limits.py`
- `tests/services/test_cost_estimation.py`
- `tests/services/test_file_processing_models.py`
- `tests/test_health.py`

Note: `app/core/warmup.py` (245 lines) is a new warmup module but will be a separate PR (not OCR-specific).

### Non-OCR files -- DROP (~12 files)

- `app/services/smart_suggestions.py` -- already on `smart_suggestion_clean`
- `app/services/response_buttons.py` -- already on `smart_suggestion_clean`
- `app/services/guardrails.py` rewrite -- already on `guardrails_clean`
- `tests/services/test_response_buttons.py` -- belongs to other PR
- `tests/services/test_guardrails.py` -- belongs to other PR
- `tests/agents/test_supervisor_guardrails.py` -- belongs to other PR

### Mixed files -- TAKE only OCR parts (~12 files)

| File | TAKE (OCR only) | DROP (non-OCR) |
|------|-----------------|----------------|
| `app/agents/supervisor.py` | attachments param, `_load_session_documents()`, `_build_message_with_context()`, document tag stripping | smart suggestions/response buttons integration, guardrails rewrite calls |
| `app/schemas/chat.py` | `FileAttachmentMetadata`, attachments fields, max_length=500000 | SuggestedButton (on smart_suggestion_clean) |
| `app/api/v1/endpoints/chat.py` | attachments pass-through | nothing major |
| `app/api/v1/router.py` | upload + ocr_config routes | -- |
| `app/config.py` | OCR settings: `ocr_approach`, `max_upload_size_mb`, `max_pdf_pages`, `google_api_key` | SMTP/password-reset setting removals |
| `app/main.py` | `LargeFileErrorMiddleware` | warmup rewrite (separate PR) |
| `app/services/conversation_memory.py` | `_strip_document_context()`, attachments param | datetime style changes |
| `requirements.txt` | OCR deps: aiofiles, PyMuPDF, python-docx, openpyxl, pandas, Pillow, google-cloud-storage | -- |
| `app/prompts/erpnext/system_en.txt` + `system_hi.txt` | "File Upload Response Guidelines" section | non-OCR prompt simplifications |
| `pyproject.toml` | OCR-related config | non-OCR tool config changes |
| `.env.example` | OCR env vars | -- |
| `.gitignore` | `ocr_output/`, `venv312/` entries | -- |

---

## Full File-by-File Analysis: ocr_implementation (54 files changed vs main)

### OCR-only files -- TAKE (4 new files)

- `src/components/chat/file-attachment.tsx` (204 lines) -- renders file attachment chips
- `src/components/chat/file-preview-modal.tsx` (206 lines) -- file preview overlay
- `src/components/chat/file-upload-button.tsx` (85 lines) -- upload button component
- `src/lib/api/upload.ts` (42 lines) -- upload API client

### Non-OCR files -- DROP (~30 files)

- `src/components/chat/suggested-buttons.tsx` -- already on `smart_suggestion_frontend_clean`
- `src/components/chat/thinking-indicator.tsx` -- complete rewrite (non-OCR)
- `src/app/(dashboard)/settings/page.tsx` -- avatar system gutted (non-OCR)
- `src/components/ui/image-crop-dialog.tsx` DELETED -- avatar removal
- `src/components/ui/user-avatar.tsx` DELETED -- avatar removal
- `src/lib/utils/crop-image.ts` DELETED -- avatar removal
- `src/components/forms/forgot-password-form.tsx` -- restyled (non-OCR)
- `src/components/forms/reset-password-form.tsx` -- restyled (non-OCR)
- `src/components/layout/sidebar.tsx` -- avatar/search removal (non-OCR)
- `src/stores/toast-store.ts` -- new store (non-OCR; upload code imports it but we will use existing `use-toast` hook from main instead)
- `src/lib/api/client.ts` -- error handling improvements (non-OCR)
- `src/lib/api/auth.ts` -- avatar endpoints removed (non-OCR)
- `src/lib/hooks/use-auth.ts` -- profile update removed (non-OCR)
- `src/stores/auth-store.ts` -- avatar state removed (non-OCR)
- `src/types/auth.ts` -- avatarUrl removed (non-OCR)
- `src/app/globals.css`, `tailwind.config.ts` -- animation removals (non-OCR)
- 11 test files DELETED (2,000+ lines) -- non-OCR test deletions (intentional, separate PR)
- 3 test files MODIFIED -- non-OCR assertion changes

Note: Avatar removal, auth form restyling, and test deletions are intentional changes but will be done as a separate PR, NOT mixed into OCR.

### Mixed files -- TAKE only OCR parts (~10 files)

| File | TAKE (OCR only) | DROP (non-OCR) |
|------|-----------------|----------------|
| `src/types/chat.ts` | `FileAttachmentMetadata`, `FileAttachment`, `UploadResponse`, attachments on Message/ChatState | `SuggestedButton` (on smart_suggestion_frontend_clean) |
| `src/stores/chat-store.ts` | `pendingAttachments` state + 4 attachment actions | session persistence changes |
| `src/lib/hooks/use-chat.ts` | `uploadFile()`, `cancelUpload()`, attachment handling in `sendMessage`. Replace `toast-store` import with existing `use-toast` hook | session persistence, title refetch removal |
| `src/components/chat/chat-container.tsx` | OCR props: isUploading, onFileUpload, pendingAttachments, onRemoveAttachment | -- |
| `src/components/chat/chat-input.tsx` | file upload UI, validation, PDF page counting | icon changes, Button removal |
| `src/components/chat/message-bubble.tsx` | attachment rendering for user/assistant messages | avatar removal, spinner replacement |
| `src/app/(dashboard)/chat/page.tsx` | OCR props from useChat | -- |
| `src/lib/constants.ts` | UPLOAD endpoint constant | title generation constant removals |
| `src/components/chat/index.ts` | OCR component exports: FileUploadButton, FileAttachment, FilePreviewModal | ThinkingIndicator export removal |
| `src/lib/api/chat.ts` | `uploadFile` method | -- |

---

## Execution Plan

### Phase 1: Create Backend `ocr_clean` from main

**Step 1:** Create branch

```bash
cd erpsense-backend
git checkout main && git pull && git checkout -b ocr_clean
```

**Step 2:** Checkout pure OCR files wholesale from ocr_backend

```bash
git checkout ocr_backend -- app/services/file_processing/
git checkout ocr_backend -- app/api/v1/endpoints/upload.py
git checkout ocr_backend -- app/api/v1/endpoints/ocr_config.py
git checkout ocr_backend -- app/config/ocr_features.json
git checkout ocr_backend -- app/api/deps.py
git checkout ocr_backend -- tests/api/test_upload_session_limits.py
git checkout ocr_backend -- tests/services/test_cost_estimation.py
git checkout ocr_backend -- tests/services/test_file_processing_models.py
git checkout ocr_backend -- tests/test_health.py
```

**Step 3:** Manually edit mixed files (add OCR parts only, keep everything else intact)

1. `app/config.py` -- add OCR settings block (ocr_approach, max_upload_size_mb, max_pdf_pages, google_api_key)
2. `app/api/v1/router.py` -- add upload + ocr_config router includes
3. `app/schemas/chat.py` -- add FileAttachmentMetadata, attachments fields, bump max_length to 500000
4. `app/agents/supervisor.py` -- add attachments param, `_load_session_documents()`, `_build_message_with_context()`, document tag stripping
5. `app/api/v1/endpoints/chat.py` -- add attachments pass-through to supervisor.process()
6. `app/services/conversation_memory.py` -- add `_strip_document_context()`, attachments param on add_message()
7. `app/main.py` -- add LargeFileErrorMiddleware class and middleware registration
8. `app/prompts/erpnext/system_en.txt` + `system_hi.txt` -- add "File Upload Response Guidelines" section only
9. `requirements.txt` -- add OCR dependencies (aiofiles, PyMuPDF, python-docx, openpyxl, pandas, Pillow, google-cloud-storage)
10. `pyproject.toml` -- add OCR-related config if any
11. `.env.example` -- add OCR env vars
12. `.gitignore` -- add ocr_output/, venv312/

**Step 4:** Verify imports resolve (upload.py may reference warmup or config that only exists on ocr_backend)

**Step 5:** Run tests, mypy, ruff

```bash
pytest tests/ -x
mypy app/
ruff check app/ tests/
ruff format --check app/ tests/
```

**Step 6:** Commit and push

```bash
git add -A
git commit -m "feat: add OCR file processing pipeline with multi-engine support"
git push -u origin ocr_clean
```

### Phase 2: Create Frontend `ocr_frontend_clean` from main

**Step 1:** Create branch

```bash
cd erpsense-frontend
git checkout main && git pull && git checkout -b ocr_frontend_clean
```

**Step 2:** Checkout pure OCR files from ocr_implementation

```bash
git checkout ocr_implementation -- src/components/chat/file-attachment.tsx
git checkout ocr_implementation -- src/components/chat/file-preview-modal.tsx
git checkout ocr_implementation -- src/components/chat/file-upload-button.tsx
git checkout ocr_implementation -- src/lib/api/upload.ts
```

**Step 3:** Manually edit mixed files (add OCR parts only)

1. `src/types/chat.ts` -- add FileAttachmentMetadata, FileAttachment, UploadResponse types; add attachments on Message/ChatState
2. `src/stores/chat-store.ts` -- add pendingAttachments state + 4 attachment actions
3. `src/lib/hooks/use-chat.ts` -- add uploadFile(), cancelUpload(), attachment handling in sendMessage. Replace toast-store import with existing use-toast hook from main
4. `src/components/chat/chat-container.tsx` -- add OCR props (isUploading, onFileUpload, pendingAttachments, onRemoveAttachment)
5. `src/components/chat/chat-input.tsx` -- add file upload UI, validation, PDF page counting
6. `src/components/chat/message-bubble.tsx` -- add attachment rendering
7. `src/app/(dashboard)/chat/page.tsx` -- wire OCR props from useChat
8. `src/lib/constants.ts` -- add UPLOAD endpoint constant
9. `src/components/chat/index.ts` -- add OCR component exports
10. `src/lib/api/chat.ts` -- add uploadFile method
11. `package.json` -- add pdfjs-dist dependency

**Step 4:** Run build, lint, and tests

```bash
npm install
npm run build
npm run lint
npm test
```

**Step 5:** Commit and push

```bash
git add -A
git commit -m "feat: add OCR file upload UI with preview and attachment support"
git push -u origin ocr_frontend_clean
```

### Phase 3: PRs and Merge Order

1. Create PRs for `ocr_clean` and `ocr_frontend_clean`
2. Verify zero overlap with PRs #18, #21, #22
3. Recommended merge order:
   - PR #22 (guardrails) first
   - PR #21 (smart suggestions) second
   - PR #18 (frontend suggestions) third
   - OCR backend fourth (may need small rebase on supervisor.py)
   - OCR frontend last (may need small rebase on types/chat.ts)

### Phase 4: Remaining Non-OCR Features (Separate PRs Later)

These features are on the OCR branches but intentionally NOT included in the clean OCR branches. Each needs its own PR:

| Feature                    | Source              | Scope                                                    |
|----------------------------|---------------------|----------------------------------------------------------|
| Warmup refactor            | `ocr_backend`       | Replace `app/agents/warmup.py` with `app/core/warmup.py` |
| Avatar/settings removal    | `ocr_implementation`| Delete avatar system, gut settings page                  |
| Auth forms restyling       | `ocr_implementation`| Restyle forgot/reset password forms                      |
| Thinking indicator rewrite | `ocr_implementation`| Replace animated dots with cycling messages              |
| Toast store                | `ocr_implementation`| New Zustand toast store replacing use-toast hook         |
| Chat session fixes         | `ocr_implementation`| Session persistence, reset on login/logout               |
| Test cleanup               | both                | Remove 8,000+ lines of deleted tests (intentional)       |

---

## Verification

- All existing tests pass (no test deletions on clean branches)
- New OCR tests pass (upload limits, cost estimation, file processing models, health)
- mypy clean, ruff clean, ruff format clean (backend)
- build + lint + jest pass (frontend)
- No overlap with PRs #18, #21, #22 (verify with `git diff --name-only`)
- Upload endpoint works: POST `/api/v1/files/upload/` accepts multipart file
- OCR config endpoint works: GET/PUT `/api/v1/ocr/config`
- Chat with attachments works: message + file sent, document context injected into agent
