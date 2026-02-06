# Smart Context Management System - Execution Plan

> **Status**: Planning
> **Last Updated**: 2024-02-06
> **Estimated Effort**: 12-16 days

---

## Executive Summary

Design an intelligent context management system that efficiently handles multiple file uploads across chat sessions while **reducing token usage by 60-70%** and maintaining contextual accuracy.

### Key Outcomes
- **Token Reduction**: ~15,000 to ~5,000 tokens per request (with 10 docs)
- **Cost Savings**: ~70% reduction in LLM API costs
- **Latency**: Context building < 500ms
- **Accuracy**: >90% relevant document retrieval

---

## Table of Contents

1. [Current State Analysis](#current-state-analysis)
2. [Target Architecture](#target-architecture)
3. [Context Management Strategy](#context-management-strategy)
4. [Data Models](#data-models)
5. [Implementation Phases](#implementation-phases)
6. [Edge Cases & Solutions](#edge-cases--solutions)
7. [API Design](#api-design)
8. [Technology Decisions](#technology-decisions)
9. [Success Metrics](#success-metrics)

---

## Current State Analysis

### Current Architecture (Problem)

```mermaid
%%{init: {'theme': 'base', 'themeVariables': { 'primaryColor': '#4a5568', 'primaryTextColor': '#e2e8f0', 'lineColor': '#718096', 'edgeLabelBackground': '#1a202c'}}}%%
flowchart TB
    subgraph CLIENT["CLIENT"]
        direction LR
        U[/"User"/]
    end

    subgraph BACKEND["BACKEND"]
        direction TB
        subgraph UPLOAD["Upload Flow"]
            OCR["OCR\nService"]
            FS[("File\nStorage")]
        end
        subgraph CHAT["Chat Flow"]
            CONCAT["Concatenate\nALL Documents"]
            LLM["Gemini\nLLM"]
        end
    end

    U -->|"1. Upload"| OCR
    OCR -->|"2. Store"| FS
    U -->|"3. Query"| CONCAT
    FS -->|"4. Load ALL"| CONCAT
    CONCAT -->|"5. Send ALL\n17,500+ tokens"| LLM
    LLM -->|"6. Response"| U

    style CONCAT fill:#c53030,color:#fff,stroke:#9b2c2c
    style CLIENT fill:#2d3748,color:#e2e8f0,stroke:#4a5568
    style BACKEND fill:#1a202c,color:#e2e8f0,stroke:#4a5568
    style UPLOAD fill:#2d3748,color:#e2e8f0,stroke:#4a5568
    style CHAT fill:#2d3748,color:#e2e8f0,stroke:#4a5568
```

### Token Growth Problem

| Message | Files Uploaded | Cumulative Files | Tokens Sent | Cost Impact |
|---------|---------------|------------------|-------------|-------------|
| 1 | 3 | 3 | 4,500 | $0.01 |
| 2 | 8 | 11 | 12,500 | $0.03 |
| 3 | 5 | 16 | 17,500 | $0.04 |
| 4 | 0 (query only) | 16 | 17,500 | $0.04 |
| 5 | 2 | 18 | 20,000 | $0.05 |
| ... | ... | ... | **Grows linearly** | High |

**Problem**: Every message sends ALL historical documents, even when irrelevant.

---

## Target Architecture

### High-Level Architecture (Solution)

```mermaid
%%{init: {'theme': 'base', 'themeVariables': { 'primaryColor': '#4a5568', 'primaryTextColor': '#e2e8f0', 'lineColor': '#718096', 'edgeLabelBackground': '#1a202c'}}}%%
flowchart TB
    subgraph CLIENT["CLIENT"]
        U[/"User"/]
    end

    subgraph BACKEND["BACKEND"]
        subgraph INGEST["Ingestion Pipeline"]
            OCR["OCR"]
            SUM["Summarizer"]
            EMB["Embedder"]
        end

        subgraph CONTEXT["Context Manager"]
            SCORE["Relevance\nScorer"]
            SELECT["Smart\nSelector"]
            BUDGET["Token\nBudget"]
        end

        subgraph STORAGE["Data Layer"]
            IDX[("Document\nIndex")]
            VEC[("Vector\nStore")]
            MEM[("Conversation\nMemory")]
        end

        LLM["Gemini LLM"]
    end

    U -->|"Upload"| OCR
    OCR --> SUM --> EMB
    EMB -->|"Index"| IDX
    EMB -->|"Embed"| VEC

    U -->|"Query"| SCORE
    IDX --> SCORE
    VEC --> SCORE
    SCORE --> SELECT --> BUDGET
    MEM --> BUDGET
    BUDGET -->|"~5,000 tokens"| LLM
    LLM -->|"Response"| U

    style BUDGET fill:#276749,color:#c6f6d5,stroke:#38a169
    style CLIENT fill:#2d3748,color:#e2e8f0,stroke:#4a5568
    style BACKEND fill:#1a202c,color:#e2e8f0,stroke:#4a5568
    style INGEST fill:#2c5282,color:#bee3f8,stroke:#3182ce
    style CONTEXT fill:#276749,color:#c6f6d5,stroke:#38a169
    style STORAGE fill:#744210,color:#fefcbf,stroke:#b7791f
```

### Detailed Component Architecture

```mermaid
%%{init: {'theme': 'base', 'themeVariables': { 'primaryColor': '#4a5568', 'primaryTextColor': '#e2e8f0', 'lineColor': '#718096'}}}%%
flowchart TB
    subgraph FRONTEND["FRONTEND - Next.js"]
        direction LR
        CHAT_UI["Chat UI"]
        UPLOAD_UI["Upload UI"]
    end

    subgraph API_LAYER["API LAYER - FastAPI"]
        direction LR
        CHAT_EP["/api/chat"]
        UPLOAD_EP["/api/upload"]
    end

    subgraph SERVICES["SERVICES"]
        direction TB
        subgraph CTX_MGR["Context Manager"]
            DOC_IDX["Document\nIndexer"]
            REL_SCORE["Relevance\nScorer"]
            CTX_BUILD["Context\nBuilder"]
            TOK_MGR["Token\nManager"]
        end

        subgraph EMB_SVC["Embedding Service"]
            EMBEDDER["Embedder"]
            VEC_SEARCH["Vector\nSearch"]
        end

        subgraph MEM_SVC["Memory Service"]
            CONV_MEM["Conversation\nMemory"]
            MSG_SUM["Message\nSummarizer"]
        end

        OCR_SVC["OCR Service"]
    end

    subgraph DATA["DATA LAYER"]
        direction LR
        REDIS[("Redis\nIndex + Vectors")]
        FILES[("File System\nJSON / MD")]
    end

    subgraph EXTERNAL["EXTERNAL"]
        direction LR
        GEMINI_LLM["Gemini\nLLM"]
        GEMINI_EMB["Gemini\nEmbeddings"]
    end

    CHAT_UI --> CHAT_EP
    UPLOAD_UI --> UPLOAD_EP

    CHAT_EP --> CTX_BUILD
    UPLOAD_EP --> OCR_SVC

    OCR_SVC --> DOC_IDX
    DOC_IDX --> EMBEDDER
    CTX_BUILD --> REL_SCORE
    REL_SCORE --> VEC_SEARCH
    CTX_BUILD --> CONV_MEM
    CTX_BUILD --> TOK_MGR
    TOK_MGR --> GEMINI_LLM

    EMBEDDER --> GEMINI_EMB
    DOC_IDX --> REDIS
    EMBEDDER --> REDIS
    OCR_SVC --> FILES

    style FRONTEND fill:#2d3748,color:#e2e8f0,stroke:#4a5568
    style API_LAYER fill:#2d3748,color:#e2e8f0,stroke:#4a5568
    style SERVICES fill:#1a202c,color:#e2e8f0,stroke:#4a5568
    style CTX_MGR fill:#276749,color:#c6f6d5,stroke:#38a169
    style EMB_SVC fill:#2c5282,color:#bee3f8,stroke:#3182ce
    style MEM_SVC fill:#702459,color:#fed7e2,stroke:#b83280
    style DATA fill:#744210,color:#fefcbf,stroke:#b7791f
    style EXTERNAL fill:#553c9a,color:#e9d8fd,stroke:#805ad5
```

---

## Context Management Strategy

### Document Lifecycle

```mermaid
%%{init: {'theme': 'base', 'themeVariables': { 'primaryColor': '#4a5568', 'primaryTextColor': '#e2e8f0', 'lineColor': '#718096'}}}%%
flowchart LR
    subgraph UPLOAD["Upload Phase"]
        S1["Uploading"]
        S2["Processing"]
        S3["Indexed"]
    end

    subgraph ACTIVE["Active Phase"]
        S4["Active\n(Full Content)"]
    end

    subgraph ARCHIVE["Archive Phase"]
        S5["Summarized\n(Summary Only)"]
        S6["Archived\n(Metadata Only)"]
    end

    S1 -->|"complete"| S2 -->|"OCR done"| S3 -->|"in window"| S4
    S4 -->|"exits window"| S5 -->|"session ages"| S6
    S5 -.->|"query match"| S4
    S6 -.->|"deep search"| S4

    style UPLOAD fill:#2c5282,color:#bee3f8,stroke:#3182ce
    style ACTIVE fill:#276749,color:#c6f6d5,stroke:#38a169
    style ARCHIVE fill:#744210,color:#fefcbf,stroke:#b7791f
```

### Context Selection Flow

```mermaid
%%{init: {'theme': 'base', 'themeVariables': { 'primaryColor': '#4a5568', 'primaryTextColor': '#e2e8f0', 'lineColor': '#718096'}}}%%
flowchart TB
    subgraph INPUT["INPUT"]
        QUERY["User Query"]
        DOCS["Session Documents"]
    end

    subgraph RELEVANCE["RELEVANCE SCORING"]
        direction TB
        CHECK1{"New Attachment?"}
        CHECK2{"Filename Match?"}
        CHECK3{"Semantic Sim > 0.7?"}
        CHECK4{"Semantic Sim > 0.4?"}
    end

    subgraph LEVELS["CONTENT LEVELS"]
        direction TB
        FULL["FULL\n~1500 tokens"]
        SUMMARY["SUMMARY\n~150 tokens"]
        META["METADATA\n~40 tokens"]
    end

    subgraph BUDGET["TOKEN BUDGET"]
        direction TB
        CHECK_B{"Under 8000?"}
        DOWNGRADE["Downgrade Level"]
        INCLUDE["Include in Context"]
    end

    subgraph OUTPUT["OUTPUT"]
        CONTEXT["Optimized Context\n~5000 tokens"]
    end

    QUERY --> CHECK1
    DOCS --> CHECK1
    CHECK1 -->|"Yes"| FULL
    CHECK1 -->|"No"| CHECK2
    CHECK2 -->|"Yes"| FULL
    CHECK2 -->|"No"| CHECK3
    CHECK3 -->|"Yes"| FULL
    CHECK3 -->|"No"| CHECK4
    CHECK4 -->|"Yes"| SUMMARY
    CHECK4 -->|"No"| META

    FULL --> CHECK_B
    SUMMARY --> CHECK_B
    META --> CHECK_B
    CHECK_B -->|"Yes"| INCLUDE
    CHECK_B -->|"No"| DOWNGRADE
    DOWNGRADE --> CHECK_B
    INCLUDE --> CONTEXT

    style INPUT fill:#2d3748,color:#e2e8f0,stroke:#4a5568
    style RELEVANCE fill:#2c5282,color:#bee3f8,stroke:#3182ce
    style LEVELS fill:#1a202c,color:#e2e8f0,stroke:#4a5568
    style BUDGET fill:#276749,color:#c6f6d5,stroke:#38a169
    style OUTPUT fill:#276749,color:#c6f6d5,stroke:#38a169
    style FULL fill:#276749,color:#c6f6d5,stroke:#38a169
    style SUMMARY fill:#c05621,color:#feebc8,stroke:#dd6b20
    style META fill:#4a5568,color:#e2e8f0,stroke:#718096
```

### Sliding Window Strategy

| Window | Messages | Content Level |
|--------|----------|---------------|
| **Active** | N, N-1, N-2 | Full content |
| **Summary** | N-3, N-4, N-5 | Summaries only |
| **Archive** | 1 to N-6 | Metadata only |

### Token Budget Allocation (8000 total)

| Component | Tokens | % |
|-----------|--------|---|
| System Prompt | 500 | 6% |
| Conversation Active | 1500 | 19% |
| Conversation Summaries | 500 | 6% |
| Documents Full | 2000 | 25% |
| Documents Summaries | 800 | 10% |
| Documents Metadata | 200 | 3% |
| User Query | 500 | 6% |
| Safety Buffer | 2000 | 25% |

### Content Levels

```mermaid
%%{init: {'theme': 'base', 'themeVariables': { 'primaryColor': '#4a5568', 'primaryTextColor': '#e2e8f0', 'lineColor': '#718096'}}}%%
flowchart TB
    subgraph FULL_BOX["FULL CONTENT"]
        FULL_T["~1500 tokens"]
        FULL_D["Complete OCR text\nAll tables\nAll extracted data"]
    end

    subgraph SUM_BOX["SUMMARY"]
        SUM_T["~150 tokens"]
        SUM_D["2-3 sentence summary\nKey entities\nMain amounts"]
    end

    subgraph META_BOX["METADATA"]
        META_T["~40 tokens"]
        META_D["Filename\nDocument type\nDate"]
    end

    FULL_BOX -->|"budget pressure"| SUM_BOX -->|"more pressure"| META_BOX

    style FULL_BOX fill:#276749,color:#c6f6d5,stroke:#38a169
    style SUM_BOX fill:#c05621,color:#feebc8,stroke:#dd6b20
    style META_BOX fill:#4a5568,color:#e2e8f0,stroke:#718096
```

---

## Data Models

### Entity Relationship

```mermaid
%%{init: {'theme': 'base', 'themeVariables': { 'primaryColor': '#4a5568', 'primaryTextColor': '#e2e8f0', 'lineColor': '#718096'}}}%%
flowchart TB
    subgraph SESSION_BOX["SESSION"]
        S_ID["session_id: UUID"]
        S_USER["user_id: UUID"]
        S_CREATED["created_at: DateTime"]
    end

    subgraph MSG_BOX["MESSAGE_GROUP"]
        M_IDX["group_idx: Int"]
        M_USER["user_message: String"]
        M_RESP["assistant_response: String"]
        M_SUM["summary: String"]
    end

    subgraph DOC_BOX["DOCUMENT_INDEX"]
        D_ID["doc_id: UUID"]
        D_FILE["filename: String"]
        D_TYPE["document_type: String"]
        D_FULL["full_content: Text"]
        D_SUM["summary: String"]
        D_META["metadata: JSON"]
    end

    subgraph EMB_BOX["EMBEDDING"]
        E_ID["doc_id: UUID"]
        E_VEC["embedding: Vector768"]
    end

    SESSION_BOX -->|"1:N"| MSG_BOX
    SESSION_BOX -->|"1:N"| DOC_BOX
    DOC_BOX -->|"1:1"| EMB_BOX

    style SESSION_BOX fill:#2c5282,color:#bee3f8,stroke:#3182ce
    style MSG_BOX fill:#702459,color:#fed7e2,stroke:#b83280
    style DOC_BOX fill:#276749,color:#c6f6d5,stroke:#38a169
    style EMB_BOX fill:#553c9a,color:#e9d8fd,stroke:#805ad5
```

### Document Index Schema

```python
DocumentIndex = {
    # Identity
    "doc_id": "uuid-string",
    "session_id": "uuid-string",
    "filename": "invoice_001.pdf",
    "file_type": ".pdf",

    # Position
    "message_group_idx": 2,
    "upload_order": 1,
    "uploaded_at": "2024-02-06T10:30:00Z",

    # Classification
    "document_type": "invoice",
    "document_subtype": "purchase_invoice",

    # Content Levels
    "full_content": "Invoice #12345\nDate: 2024-01-15\n...",
    "full_token_count": 1500,

    "summary": "Purchase invoice #12345 from Acme Corp dated Jan 15, 2024 for $5,000 total with 3 line items.",
    "summary_token_count": 45,

    "metadata": {
        "entities": ["Acme Corp", "John Smith"],
        "amounts": ["$5,000", "$1,500", "$2,000", "$1,500"],
        "dates": ["2024-01-15", "2024-02-15"],
        "key_terms": ["invoice", "purchase", "net 30", "payment"],
        "erp_fields": {
            "vendor": "Acme Corp",
            "invoice_number": "12345",
            "total": 5000,
            "currency": "USD"
        }
    },

    # Usage
    "reference_count": 3,
    "last_referenced_at": "2024-02-06T11:00:00Z"
}
```

---

## Implementation Phases

### Phase Overview

| Phase | Name | Duration | Depends On |
|-------|------|----------|------------|
| 1 | Document Indexing | 2-3 days | - |
| 2 | Context Builder | 2-3 days | Phase 1 |
| 3 | Semantic Search | 3-4 days | Phase 1 |
| 4 | Conversation Memory | 2-3 days | Phase 2 |
| 5 | Optimization | 2-3 days | Phase 3, 4 |
| - | **MVP** | - | Phase 2 |
| - | **Full** | - | Phase 5 |

### Phase Dependencies

```mermaid
%%{init: {'theme': 'base', 'themeVariables': { 'primaryColor': '#4a5568', 'primaryTextColor': '#e2e8f0', 'lineColor': '#718096'}}}%%
flowchart TB
    subgraph TIMELINE["IMPLEMENTATION TIMELINE"]
        direction TB

        subgraph WEEK1["Week 1"]
            P1["Phase 1\nDocument Indexing\n2-3 days"]
        end

        subgraph WEEK2["Week 2"]
            P2["Phase 2\nContext Builder\n2-3 days"]
            P3["Phase 3\nSemantic Search\n3-4 days"]
        end

        subgraph WEEK3["Week 3"]
            P4["Phase 4\nConversation Memory\n2-3 days"]
            P5["Phase 5\nOptimization\n2-3 days"]
        end
    end

    subgraph MILESTONES["MILESTONES"]
        MVP(["MVP\nFunctional"])
        FULL(["Full\nProduction"])
    end

    P1 --> P2
    P1 --> P3
    P2 --> P4
    P3 --> P5
    P4 --> P5
    P2 --> MVP
    P5 --> FULL

    style TIMELINE fill:#1a202c,color:#e2e8f0,stroke:#4a5568
    style WEEK1 fill:#2d3748,color:#e2e8f0,stroke:#4a5568
    style WEEK2 fill:#2d3748,color:#e2e8f0,stroke:#4a5568
    style WEEK3 fill:#2d3748,color:#e2e8f0,stroke:#4a5568
    style MILESTONES fill:#2d3748,color:#e2e8f0,stroke:#4a5568
    style P1 fill:#2c5282,color:#bee3f8,stroke:#3182ce
    style P2 fill:#276749,color:#c6f6d5,stroke:#38a169
    style P3 fill:#c05621,color:#feebc8,stroke:#dd6b20
    style P4 fill:#702459,color:#fed7e2,stroke:#b83280
    style P5 fill:#553c9a,color:#e9d8fd,stroke:#805ad5
    style MVP fill:#38a169,color:#fff,stroke:#276749
    style FULL fill:#3182ce,color:#fff,stroke:#2c5282
```

---

## Phase 1: Document Indexing Foundation

**Goal**: Store document metadata and summaries on upload

**Duration**: 2-3 days

### Upload Flow

```mermaid
%%{init: {'theme': 'base', 'themeVariables': { 'primaryColor': '#4a5568', 'primaryTextColor': '#e2e8f0', 'lineColor': '#718096'}}}%%
flowchart LR
    subgraph CLIENT["Client"]
        USER[/"User"/]
    end

    subgraph UPLOAD_FLOW["Upload Pipeline"]
        direction TB
        API["API\n/upload"]
        OCR["OCR\nService"]
        IDX["Document\nIndexer"]
        SUM["Summarizer"]
        META["Metadata\nExtractor"]
    end

    subgraph STORE["Storage"]
        REDIS[("Redis\nDocument Index")]
        FILES[("Files\nJSON/MD")]
    end

    USER -->|"1. Upload File"| API
    API -->|"2. Extract Text"| OCR
    OCR -->|"3. Index"| IDX
    IDX -->|"4a. Generate Summary"| SUM
    IDX -->|"4b. Extract Metadata"| META
    SUM -->|"5. Store"| REDIS
    META --> REDIS
    OCR -->|"Store Raw"| FILES
    REDIS -->|"6. Response"| USER

    style CLIENT fill:#2d3748,color:#e2e8f0,stroke:#4a5568
    style UPLOAD_FLOW fill:#2c5282,color:#bee3f8,stroke:#3182ce
    style STORE fill:#744210,color:#fefcbf,stroke:#b7791f
```

### Tasks

- [ ] **1.1** Create data models (DocumentIndex, DocumentMetadata)
- [ ] **1.2** Create DocumentIndexService with CRUD operations
- [ ] **1.3** Implement Summarizer service (Gemini API)
- [ ] **1.4** Implement metadata extraction (entities, amounts, dates)
- [ ] **1.5** Create TokenCounter utility
- [ ] **1.6** Integrate indexing into upload endpoint
- [ ] **1.7** Add index info to upload response

### Files to Create

```
app/services/context_manager/
├── __init__.py
├── models.py                 # DocumentIndex, DocumentMetadata, MessageGroup
├── document_index_service.py # CRUD for document index
├── summarizer.py             # LLM-based summarization
├── metadata_extractor.py     # Entity/amount extraction
└── token_counter.py          # Token counting utility
```

---

## Phase 2: Basic Context Builder

**Goal**: Build optimized context with token budgeting

**Duration**: 2-3 days

### Chat Flow

```mermaid
%%{init: {'theme': 'base', 'themeVariables': { 'primaryColor': '#4a5568', 'primaryTextColor': '#e2e8f0', 'lineColor': '#718096'}}}%%
flowchart LR
    subgraph CLIENT["Client"]
        USER[/"User"/]
    end

    subgraph CHAT_FLOW["Chat Pipeline"]
        direction TB
        API["API\n/chat"]
        CTX["Context\nBuilder"]
        SCORE["Relevance\nScorer"]
        BUDGET["Token\nBudget"]
    end

    subgraph DATA["Data"]
        REDIS[("Redis")]
    end

    subgraph LLM_BOX["LLM"]
        GEMINI["Gemini"]
    end

    USER -->|"1. Query"| API
    API -->|"2. Build Context"| CTX
    CTX -->|"3. Get Docs"| REDIS
    REDIS -->|"4. Documents"| SCORE
    SCORE -->|"5. Ranked"| BUDGET
    BUDGET -->|"6. ~5000 tokens"| GEMINI
    GEMINI -->|"7. Response"| USER

    style CLIENT fill:#2d3748,color:#e2e8f0,stroke:#4a5568
    style CHAT_FLOW fill:#276749,color:#c6f6d5,stroke:#38a169
    style DATA fill:#744210,color:#fefcbf,stroke:#b7791f
    style LLM_BOX fill:#553c9a,color:#e9d8fd,stroke:#805ad5
```

### Tasks

- [ ] **2.1** Create ContextBuilder service
- [ ] **2.2** Implement RelevanceScorer (filename + keyword matching)
- [ ] **2.3** Implement sliding window for conversation
- [ ] **2.4** Implement TokenBudgetManager
- [ ] **2.5** Create context assembly logic (full to summary to metadata)
- [ ] **2.6** Integrate into chat endpoint
- [ ] **2.7** Add context metrics to response (for debugging)

### Files to Create

```
app/services/context_manager/
├── context_builder.py        # Main context assembly
├── relevance_scorer.py       # Document relevance scoring
├── token_budget_manager.py   # Budget allocation and enforcement
└── context_assembler.py      # Assemble final context string
```

---

## Phase 3: Semantic Search

**Goal**: Find relevant documents by meaning

**Duration**: 3-4 days

### Embedding Flow

```mermaid
%%{init: {'theme': 'base', 'themeVariables': { 'primaryColor': '#4a5568', 'primaryTextColor': '#e2e8f0', 'lineColor': '#718096'}}}%%
flowchart TB
    subgraph INGEST["ON UPLOAD"]
        direction LR
        DOC["Document\nText"]
        EMB1["Embedder"]
        VEC1["Vector\n[768 dims]"]
        STORE["Vector\nStore"]

        DOC -->|"1. Embed"| EMB1 -->|"2. Generate"| VEC1 -->|"3. Store"| STORE
    end

    subgraph QUERY["ON QUERY"]
        direction LR
        QRY["User\nQuery"]
        EMB2["Embedder"]
        VEC2["Query\nVector"]
        SEARCH["Similarity\nSearch"]
        RESULTS["Top K\nDocuments"]

        QRY -->|"1. Embed"| EMB2 -->|"2. Generate"| VEC2 -->|"3. Search"| SEARCH -->|"4. Return"| RESULTS
    end

    subgraph EXTERNAL["GEMINI API"]
        GEMINI_EMB["Embedding\nModel"]
    end

    EMB1 --> GEMINI_EMB
    EMB2 --> GEMINI_EMB
    STORE -.->|"Vector DB"| SEARCH

    style INGEST fill:#2c5282,color:#bee3f8,stroke:#3182ce
    style QUERY fill:#276749,color:#c6f6d5,stroke:#38a169
    style EXTERNAL fill:#553c9a,color:#e9d8fd,stroke:#805ad5
```

### Tasks

- [ ] **3.1** Create Embedder service (Gemini embedding API)
- [ ] **3.2** Create VectorStore service (Redis VSS)
- [ ] **3.3** Generate embeddings on document upload
- [ ] **3.4** Implement semantic similarity search
- [ ] **3.5** Enhance RelevanceScorer with semantic signals
- [ ] **3.6** Add query embedding at request time
- [ ] **3.7** Benchmark embedding latency

### Files to Create

```
app/services/embedding/
├── __init__.py
├── embedder.py               # Generate embeddings via Gemini
├── vector_store.py           # Redis VSS operations
└── similarity_search.py      # Search and rank by similarity
```

---

## Phase 4: Conversation Memory Enhancement

**Goal**: Efficient long conversation handling

**Duration**: 2-3 days

### Memory Management Flow

```mermaid
%%{init: {'theme': 'base', 'themeVariables': { 'primaryColor': '#4a5568', 'primaryTextColor': '#e2e8f0', 'lineColor': '#718096'}}}%%
flowchart TB
    subgraph SAVE["ON MESSAGE SAVE"]
        direction TB
        MSG["New Message\nGroup"]
        STORE_FULL["Store Full\nContent"]
        CHECK{"Window\n> 3 groups?"}
        SUMMARIZE["Summarize\nOldest Group"]
    end

    subgraph LOAD["ON CONTEXT BUILD"]
        direction TB
        FETCH["Fetch All\nGroups"]
        ACTIVE["Active Window\nFull Content\n(last 3)"]
        HISTORY["History\nSummaries Only\n(older)"]
        COMBINE["Combined\nContext"]
    end

    subgraph MEMORY["REDIS"]
        MEM_STORE[("Conversation\nMemory")]
    end

    MSG --> STORE_FULL --> MEM_STORE
    MEM_STORE --> CHECK
    CHECK -->|"Yes"| SUMMARIZE --> MEM_STORE
    CHECK -->|"No"| DONE["Done"]

    FETCH --> MEM_STORE
    MEM_STORE --> ACTIVE
    MEM_STORE --> HISTORY
    ACTIVE --> COMBINE
    HISTORY --> COMBINE

    style SAVE fill:#2c5282,color:#bee3f8,stroke:#3182ce
    style LOAD fill:#276749,color:#c6f6d5,stroke:#38a169
    style MEMORY fill:#744210,color:#fefcbf,stroke:#b7791f
```

### Tasks

- [ ] **4.1** Enhance ConversationMemoryService with message groups
- [ ] **4.2** Implement automatic summarization trigger
- [ ] **4.3** Track document references per message
- [ ] **4.4** Implement cross-message reference resolution
- [ ] **4.5** Add conversation-level summary for very long sessions
- [ ] **4.6** Handle message editing with context updates

### Files to Modify/Create

```
app/services/
├── conversation_memory.py    # Enhanced with summaries
└── context_manager/
    └── conversation_manager.py  # Message group management
```

---

## Phase 5: Optimization and Edge Cases

**Goal**: Production-ready robustness

**Duration**: 2-3 days

### Edge Case Handling

```mermaid
%%{init: {'theme': 'base', 'themeVariables': { 'primaryColor': '#4a5568', 'primaryTextColor': '#e2e8f0', 'lineColor': '#718096'}}}%%
flowchart TB
    subgraph CASES["EDGE CASES"]
        direction TB
        subgraph LARGE["Large File > 10k tokens"]
            L1["Detect"] --> L2["Chunk"] --> L3["Embed Each"] --> L4["Select Relevant"]
        end

        subgraph MANY["Many Files > 10"]
            M1["Count"] --> M2["Rank All"] --> M3["Top 3 Full"] --> M4["Rest Metadata"]
        end

        subgraph AMBIG["Ambiguous Reference"]
            A1["'that file'"] --> A2["Semantic Match"] --> A3{"Confidence\n> 0.8?"}
            A3 -->|"Yes"| A4["Use Best"]
            A3 -->|"No"| A5["Ask User"]
        end

        subgraph CROSS["Cross-Message Ref"]
            C1["'earlier file'"] --> C2["Search Session"] --> C3["Promote to Active"]
        end
    end

    style CASES fill:#1a202c,color:#e2e8f0,stroke:#4a5568
    style LARGE fill:#2c5282,color:#bee3f8,stroke:#3182ce
    style MANY fill:#276749,color:#c6f6d5,stroke:#38a169
    style AMBIG fill:#c05621,color:#feebc8,stroke:#dd6b20
    style CROSS fill:#702459,color:#fed7e2,stroke:#b83280
```

### Tasks

- [ ] **5.1** Implement chunking for large files
- [ ] **5.2** Handle 10+ files per message gracefully
- [ ] **5.3** Add caching for repeated queries
- [ ] **5.4** Implement clarification prompts for ambiguity
- [ ] **5.5** Add analytics/logging for context efficiency
- [ ] **5.6** Performance testing and optimization
- [ ] **5.7** Error handling and fallbacks

---

## Edge Cases and Solutions

```mermaid
%%{init: {'theme': 'base', 'themeVariables': { 'primaryColor': '#4a5568', 'primaryTextColor': '#e2e8f0', 'lineColor': '#718096'}}}%%
flowchart LR
    subgraph PROBLEMS["PROBLEMS"]
        direction TB
        P1["Large File\n> 10k tokens"]
        P2["Many Files\n> 10"]
        P3["Ambiguous\nReference"]
        P4["Cross-Message\nReference"]
        P5["Conflicting\nInfo"]
        P6["Long Chat\n> 50 msgs"]
    end

    subgraph SOLUTIONS["SOLUTIONS"]
        direction TB
        S1["Chunk +\nSelective Retrieval"]
        S2["Top 3 Full +\nRest Metadata"]
        S3["Recency +\nSemantic + Ask"]
        S4["Session-wide\nSearch"]
        S5["Include All\nRelevant"]
        S6["Aggressive\nSummarization"]
    end

    P1 --> S1
    P2 --> S2
    P3 --> S3
    P4 --> S4
    P5 --> S5
    P6 --> S6

    style PROBLEMS fill:#742a2a,color:#feb2b2,stroke:#c53030
    style SOLUTIONS fill:#276749,color:#c6f6d5,stroke:#38a169
```

| Scenario | Detection | Solution | Fallback |
|----------|-----------|----------|----------|
| Very large file (>10k tokens) | Token count | Chunk into sections, embed each | Include summary only |
| Many files (10+) in one message | File count | Top 3 full + rest metadata | All as metadata |
| Ambiguous reference ("that file") | No direct match | Recency + semantic similarity | Ask user to clarify |
| Cross-message reference | "earlier", "before" keywords | Search all session documents | Include recent docs |
| Conflicting info across docs | Multiple high relevance | Include all relevant docs | Let LLM synthesize |
| Very long conversation (50+ msgs) | Message count | Keep only 5 groups active | Session summary |

---

## API Design

### Request Flow Architecture

```mermaid
%%{init: {'theme': 'base', 'themeVariables': { 'primaryColor': '#4a5568', 'primaryTextColor': '#e2e8f0', 'lineColor': '#718096'}}}%%
flowchart TB
    subgraph CLIENT["CLIENT"]
        REQ["POST /chat\n{message, session_id}"]
    end

    subgraph API["API LAYER"]
        ENDPOINT["Chat\nEndpoint"]
    end

    subgraph CONTEXT["CONTEXT MANAGER"]
        direction TB
        FETCH["Fetch\nDocuments"]
        SCORE["Score\nRelevance"]
        SELECT["Select\nContent"]
        ASSEMBLE["Assemble\nContext"]
    end

    subgraph DATA["DATA LAYER"]
        REDIS[("Redis")]
    end

    subgraph LLM["LLM"]
        GEMINI["Gemini"]
    end

    subgraph RESPONSE["RESPONSE"]
        RES["Response +\nContext Stats"]
    end

    REQ -->|"1"| ENDPOINT
    ENDPOINT -->|"2"| FETCH
    FETCH -->|"3"| REDIS
    REDIS -->|"4"| SCORE
    SCORE -->|"5"| SELECT
    SELECT -->|"6"| ASSEMBLE
    ASSEMBLE -->|"7 ~5k tokens"| GEMINI
    GEMINI -->|"8"| ENDPOINT
    ENDPOINT -->|"9 Save"| REDIS
    ENDPOINT -->|"10"| RES

    style CLIENT fill:#2d3748,color:#e2e8f0,stroke:#4a5568
    style API fill:#2d3748,color:#e2e8f0,stroke:#4a5568
    style CONTEXT fill:#276749,color:#c6f6d5,stroke:#38a169
    style DATA fill:#744210,color:#fefcbf,stroke:#b7791f
    style LLM fill:#553c9a,color:#e9d8fd,stroke:#805ad5
    style RESPONSE fill:#2d3748,color:#e2e8f0,stroke:#4a5568
```

### New Response Schema

```python
# Enhanced chat response
ChatResponse = {
    "response": "The invoice total is $5,000...",
    "language": "en",
    "session_id": "uuid",
    "conversation_length": 5,

    # NEW: Context statistics (optional, for debugging)
    "context_stats": {
        "total_tokens": 4500,
        "docs_full": ["doc_id_1", "doc_id_2"],
        "docs_summary": ["doc_id_3", "doc_id_4"],
        "docs_metadata": ["doc_id_5", "doc_id_6"],
        "conversation_groups_included": 3,
        "budget_utilization": 0.82
    }
}

# Enhanced upload response
UploadResponse = {
    "file_id": "uuid",
    "filename": "invoice.pdf",
    "extraction": {...},

    # NEW: Index information
    "doc_index": {
        "doc_id": "uuid",
        "summary": "Invoice #123 from Acme Corp...",
        "token_count": 1500,
        "document_type": "invoice",
        "key_entities": ["Acme Corp", "$5,000"]
    }
}
```

---

## Technology Decisions

### Technology Stack

```mermaid
%%{init: {'theme': 'base', 'themeVariables': { 'primaryColor': '#4a5568', 'primaryTextColor': '#e2e8f0', 'lineColor': '#718096'}}}%%
flowchart TB
    subgraph STACK["RECOMMENDED STACK"]
        direction TB

        subgraph EMB_CHOICE["EMBEDDING"]
            EMB_OPT1["Gemini\nEmbedding API"]
            EMB_OPT2["sentence-\ntransformers"]
            EMB_OPT1 ---|"SELECTED"| EMB_WHY["Consistency\nHigh Quality\n$0.0001/1k tok"]
        end

        subgraph VEC_CHOICE["VECTOR STORE"]
            VEC_OPT1["Redis VSS"]
            VEC_OPT2["Qdrant"]
            VEC_OPT3["Pinecone"]
            VEC_OPT1 ---|"SELECTED"| VEC_WHY["Already in stack\nNo new infra\n< 100k docs"]
        end

        subgraph SUM_CHOICE["SUMMARIZATION"]
            SUM_OPT1["Gemini LLM"]
            SUM_OPT2["BART/T5\nLocal"]
            SUM_OPT1 ---|"SELECTED"| SUM_WHY["Best quality\nAlready integrated\n$0.00025/sum"]
        end
    end

    style STACK fill:#1a202c,color:#e2e8f0,stroke:#4a5568
    style EMB_CHOICE fill:#2d3748,color:#e2e8f0,stroke:#4a5568
    style VEC_CHOICE fill:#2d3748,color:#e2e8f0,stroke:#4a5568
    style SUM_CHOICE fill:#2d3748,color:#e2e8f0,stroke:#4a5568
    style EMB_OPT1 fill:#276749,color:#c6f6d5,stroke:#38a169
    style VEC_OPT1 fill:#276749,color:#c6f6d5,stroke:#38a169
    style SUM_OPT1 fill:#276749,color:#c6f6d5,stroke:#38a169
    style EMB_WHY fill:#276749,color:#c6f6d5,stroke:#38a169
    style VEC_WHY fill:#276749,color:#c6f6d5,stroke:#38a169
    style SUM_WHY fill:#276749,color:#c6f6d5,stroke:#38a169
```

### Recommended Stack

| Component | Choice | Rationale |
|-----------|--------|-----------|
| **Embedding Model** | Gemini Embedding API | Consistency, quality, low cost |
| **Vector Store** | Redis VSS | Already in stack, sufficient scale |
| **Summarization** | Gemini LLM | Quality, already integrated |
| **Token Counter** | tiktoken | Accuracy for Gemini tokenizer |

### Cost Estimation (1000 conversations/month)

| Operation | Volume/Month | Unit Cost | Total |
|-----------|--------------|-----------|-------|
| Embeddings | 5,000 docs | $0.0001/1k tokens | ~$5 |
| Summaries | 5,000 docs | $0.00025/summary | ~$15 |
| LLM Queries | 10,000 msgs | $0.008/query (optimized) | ~$80 |
| **Total** | | | **~$100/month** |

**Savings**: ~$200/month compared to current approach (70% reduction)

---

## Success Metrics

### Performance Targets

**Token Usage: Current vs Target**

| Docs | Current | Target | Reduction |
|------|---------|--------|-----------|
| 1 | 1,500 | 1,500 | 0% |
| 5 | 7,500 | 3,000 | 60% |
| 10 | 15,000 | 5,000 | 67% |
| 20 | 22,500 | 7,000 | 69% |

| Metric | Current | Target | Improvement |
|--------|---------|--------|-------------|
| Tokens/request (10 docs) | ~15,000 | ~5,000 | 67% reduction |
| Context build latency | N/A | < 500ms | - |
| Relevant doc accuracy | N/A | > 90% | - |
| Cost/conversation (20 msgs) | ~$0.50 | ~$0.15 | 70% reduction |
| Max conversation length | ~20 msgs | 100+ msgs | 5x increase |

---

## File Structure (Final)

```
app/
├── services/
│   ├── context_manager/
│   │   ├── __init__.py
│   │   ├── models.py                 # Data models
│   │   ├── document_index_service.py # Document CRUD
│   │   ├── context_builder.py        # Main orchestrator
│   │   ├── relevance_scorer.py       # Scoring logic
│   │   ├── token_budget_manager.py   # Budget management
│   │   ├── context_assembler.py      # Final assembly
│   │   ├── summarizer.py             # LLM summarization
│   │   ├── metadata_extractor.py     # Entity extraction
│   │   └── token_counter.py          # Token utilities
│   │
│   ├── embedding/
│   │   ├── __init__.py
│   │   ├── embedder.py               # Gemini embeddings
│   │   ├── vector_store.py           # Redis VSS
│   │   └── similarity_search.py      # Search logic
│   │
│   └── conversation_memory.py        # Enhanced memory
│
├── api/v1/endpoints/
│   ├── chat.py                       # Modified
│   └── upload.py                     # Modified
│
└── agents/
    └── supervisor.py                 # Uses ContextBuilder
```

---

## Next Steps

1. **Review and Approve** this plan
2. **Confirm** technology choices (embedding, vector store)
3. **Begin Phase 1** - Document Indexing Foundation
4. **Daily standups** to track progress
5. **Test with real data** after each phase

---

## Appendix: Quick Reference

### Context Levels Cheat Sheet

| Level | Tokens | When Used | Example |
|-------|--------|-----------|---------|
| **FULL** | ~1500 | Just uploaded, explicitly referenced | Complete invoice text |
| **SUMMARY** | ~150 | In window, not referenced | "Invoice #123 from Acme, $5k total" |
| **METADATA** | ~40 | Older, for listing | "invoice.pdf, invoice, Acme Corp" |

### Relevance Signals

| Signal | Weight | Example |
|--------|--------|---------|
| Filename match | 0.4 | Query: "invoice.pdf" matches invoice.pdf |
| Explicit reference | 0.3 | Query: "the first file" matches file at index 1 |
| Keyword overlap | 0.2 | Query: "total amount" matches docs with "total" |
| Semantic similarity | 0.3 | Query: "how much to pay" matches invoice docs |
| Recency | 0.1 | Newer docs score higher |

---

*Document Version: 1.0*
*Status: Ready for Review*
