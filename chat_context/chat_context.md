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

### Current Architecture Flow

```mermaid
%%{init: {'theme': 'base', 'themeVariables': { 'primaryColor': '#4a5568', 'primaryTextColor': '#fff', 'primaryBorderColor': '#2d3748', 'lineColor': '#718096', 'secondaryColor': '#2d3748', 'tertiaryColor': '#1a202c', 'background': '#1a202c', 'mainBkg': '#2d3748', 'textColor': '#e2e8f0'}}}%%
flowchart TD
    A[User Upload] --> B[OCR Process] --> C[Store JSON/MD]
    C --> D[User Query] --> E["Pass ALL Docs"]
    E --> F[LLM] --> G[Response]
    style E fill:#e53e3e,color:#fff,stroke:#c53030
```

### Problem Visualization

```mermaid
%%{init: {'theme': 'base', 'themeVariables': { 'primaryColor': '#4a5568', 'primaryTextColor': '#fff', 'lineColor': '#718096'}}}%%
flowchart LR
    M1["Msg1: 3 files\n4500 tok"] --> CTX["ALL FILES\n17,500+ tokens"]
    M2["Msg2: 8 files\n8000 tok"] --> CTX
    M3["Msg3: 5 files\n5000 tok"] --> CTX
    CTX --> LLM["LLM\nHigh Cost"]
    style CTX fill:#e53e3e,color:#fff,stroke:#c53030
    style LLM fill:#742a2a,color:#feb2b2
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

### High-Level Architecture

```mermaid
%%{init: {'theme': 'base', 'themeVariables': { 'primaryColor': '#4a5568', 'primaryTextColor': '#fff', 'lineColor': '#718096'}}}%%
flowchart LR
    subgraph UP["Upload"]
        U1[File] --> U2[OCR] --> U3[Summary] --> U4[Embed]
    end
    subgraph ST["Storage"]
        S1[(Index)]
        S2[(Vectors)]
    end
    subgraph QR["Query"]
        Q1[Query] --> Q2[Score] --> Q3[Select] --> Q4[Budget]
    end
    subgraph LM["LLM"]
        L1[Context] --> L2[Gemini]
    end
    U4 --> S1 & S2
    S1 & S2 --> Q2
    Q4 --> L1
    style UP fill:#2c5282,color:#bee3f8
    style ST fill:#744210,color:#fefcbf
    style QR fill:#276749,color:#c6f6d5
    style LM fill:#702459,color:#fed7e2
```

### Detailed Component Architecture

```mermaid
%%{init: {'theme': 'base', 'themeVariables': { 'primaryColor': '#4a5568', 'primaryTextColor': '#e2e8f0', 'lineColor': '#a0aec0'}}}%%
flowchart LR
    subgraph FE["Frontend"]
        F1[Chat] & F2[Upload]
    end
    subgraph API["API"]
        A1[/chat] & A2[/upload]
    end
    subgraph SVC["Services"]
        CM[Context Mgr]
        EM[Embedder]
        MM[Memory]
        OCR[OCR]
    end
    subgraph DB["Storage"]
        R[(Redis)]
        FS[(Files)]
    end
    subgraph EXT["External"]
        G[Gemini]
    end
    F1 --> A1 --> CM --> G
    F2 --> A2 --> OCR --> CM
    CM --> EM --> R
    CM --> MM
    OCR --> FS
    style FE fill:#2d3748,color:#e2e8f0
    style API fill:#2d3748,color:#e2e8f0
    style SVC fill:#1a202c,color:#e2e8f0
    style DB fill:#2d3748,color:#e2e8f0
    style EXT fill:#2d3748,color:#e2e8f0
```

---

## Context Management Strategy

### Document Lifecycle State Machine

```mermaid
%%{init: {'theme': 'base', 'themeVariables': { 'primaryColor': '#4a5568', 'primaryTextColor': '#e2e8f0', 'lineColor': '#a0aec0'}}}%%
stateDiagram-v2
    [*] --> Uploading
    Uploading --> Processing --> Indexed
    Indexed --> Active: In window
    Active --> Summarized: Exits window
    Summarized --> Archived: Session ages
    Active --> Active: Referenced
    Summarized --> Active: Query match
    Archived --> Active: Deep search
```

### Context Selection Algorithm

```mermaid
%%{init: {'theme': 'base', 'themeVariables': { 'primaryColor': '#4a5568', 'primaryTextColor': '#e2e8f0', 'lineColor': '#a0aec0'}}}%%
flowchart LR
    subgraph Input
        Q[Query]
    end
    subgraph Score["Relevance Scoring"]
        S1{New file?} -->|Yes| FULL1[FULL]
        S1 -->|No| S2{Name match?}
        S2 -->|Yes| FULL2[FULL]
        S2 -->|No| S3{Sim>0.7?}
        S3 -->|Yes| FULL3[FULL]
        S3 -->|No| S4{Sim>0.4?}
        S4 -->|Yes| SUM[SUMMARY]
        S4 -->|No| META[METADATA]
    end
    subgraph Budget["Token Budget"]
        B1{Under?} -->|Yes| OK[Include]
        B1 -->|No| DG[Downgrade]
        DG --> B1
    end
    Q --> S1
    FULL1 & FULL2 & FULL3 & SUM & META --> B1
    style FULL1 fill:#276749,color:#c6f6d5
    style FULL2 fill:#276749,color:#c6f6d5
    style FULL3 fill:#276749,color:#c6f6d5
    style SUM fill:#c05621,color:#feebc8
    style META fill:#4a5568,color:#e2e8f0
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

### Content Level Comparison

```mermaid
%%{init: {'theme': 'base', 'themeVariables': { 'primaryColor': '#4a5568', 'primaryTextColor': '#e2e8f0'}}}%%
flowchart LR
    FULL["FULL ~1500t\nComplete text"] -->|pressure| SUM["SUMMARY ~150t\nKey points"] -->|more| META["META ~40t\nFilename only"]
    style FULL fill:#276749,color:#c6f6d5
    style SUM fill:#c05621,color:#feebc8
    style META fill:#4a5568,color:#e2e8f0
```

---

## Data Models

### Entity Relationship Diagram

```mermaid
%%{init: {'theme': 'base', 'themeVariables': { 'primaryColor': '#4a5568', 'primaryTextColor': '#e2e8f0', 'lineColor': '#a0aec0'}}}%%
erDiagram
    SESSION ||--o{ MESSAGE_GROUP : has
    SESSION ||--o{ DOCUMENT_INDEX : has
    DOCUMENT_INDEX ||--|| EMBEDDING : has
    SESSION { uuid id datetime created }
    MESSAGE_GROUP { int idx string user_msg string response string summary }
    DOCUMENT_INDEX { uuid id string filename string type string content string summary }
    EMBEDDING { uuid doc_id vector embedding }
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
%%{init: {'theme': 'base', 'themeVariables': { 'primaryColor': '#4a5568', 'primaryTextColor': '#e2e8f0', 'lineColor': '#a0aec0'}}}%%
flowchart LR
    P1[P1 Index] --> P2[P2 Context] & P3[P3 Search]
    P2 --> P4[P4 Memory]
    P3 & P4 --> P5[P5 Optimize]
    P2 --> MVP((MVP))
    P5 --> FULL((Full))
    style P1 fill:#2c5282,color:#bee3f8
    style P2 fill:#276749,color:#c6f6d5
    style P3 fill:#c05621,color:#feebc8
    style P4 fill:#702459,color:#fed7e2
    style P5 fill:#553c9a,color:#e9d8fd
    style MVP fill:#38a169,color:#fff
    style FULL fill:#3182ce,color:#fff
```

---

## Phase 1: Document Indexing Foundation

**Goal**: Store document metadata and summaries on upload

**Duration**: 2-3 days

### Sequence Diagram

```mermaid
%%{init: {'theme': 'base', 'themeVariables': { 'primaryColor': '#4a5568', 'primaryTextColor': '#e2e8f0', 'lineColor': '#a0aec0', 'actorBkg': '#2d3748'}}}%%
sequenceDiagram
    participant U as User
    participant API as API
    participant OCR as OCR
    participant IDX as Indexer
    participant DB as Redis
    U->>API: Upload
    API->>OCR: Process
    OCR-->>API: Content
    API->>IDX: Index
    IDX->>IDX: Summary + Meta
    IDX->>DB: Store
    API-->>U: Done
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

### Sequence Diagram

```mermaid
%%{init: {'theme': 'base', 'themeVariables': { 'primaryColor': '#4a5568', 'primaryTextColor': '#e2e8f0', 'lineColor': '#a0aec0', 'actorBkg': '#2d3748'}}}%%
sequenceDiagram
    participant U as User
    participant API as API
    participant CB as Builder
    participant DB as Redis
    participant LLM as Gemini
    U->>API: Message
    API->>CB: Build context
    CB->>DB: Get docs + history
    CB->>CB: Score + Select + Budget
    CB-->>API: Context
    API->>LLM: Send
    LLM-->>API: Response
    API-->>U: Reply
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

### Sequence Diagram

```mermaid
%%{init: {'theme': 'base', 'themeVariables': { 'primaryColor': '#4a5568', 'primaryTextColor': '#e2e8f0', 'lineColor': '#a0aec0', 'actorBkg': '#2d3748'}}}%%
sequenceDiagram
    participant IDX as Indexer
    participant EMB as Embedder
    participant VS as VectorDB
    participant API as Gemini
    Note over IDX,API: Upload
    IDX->>EMB: Embed doc
    EMB->>API: Get vector
    EMB->>VS: Store
    Note over IDX,API: Query
    EMB->>API: Embed query
    EMB->>VS: Search
    VS-->>EMB: Top K
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

### Sequence Diagram

```mermaid
%%{init: {'theme': 'base', 'themeVariables': { 'primaryColor': '#4a5568', 'primaryTextColor': '#e2e8f0', 'lineColor': '#a0aec0', 'actorBkg': '#2d3748'}}}%%
sequenceDiagram
    participant API as API
    participant MEM as Memory
    participant DB as Redis
    Note over API,DB: Save
    API->>MEM: Store msg
    MEM->>DB: Save
    alt Window exceeded
        MEM->>MEM: Summarize old
    end
    Note over API,DB: Load
    MEM->>DB: Fetch
    MEM-->>API: Active full + History summary
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

### Edge Case Handling Flow

```mermaid
%%{init: {'theme': 'base', 'themeVariables': { 'primaryColor': '#4a5568', 'primaryTextColor': '#e2e8f0', 'lineColor': '#a0aec0'}}}%%
flowchart LR
    subgraph Cases["Edge Cases"]
        C1[Large File] --> S1[Chunk + Select]
        C2[Many Files] --> S2[Top 3 Full]
        C3[Ambiguous] --> S3[Semantic + Ask]
        C4[Cross-ref] --> S4[Session Search]
    end
    style Cases fill:#2d3748,color:#e2e8f0
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
%%{init: {'theme': 'base', 'themeVariables': { 'primaryColor': '#4a5568', 'primaryTextColor': '#e2e8f0', 'lineColor': '#a0aec0'}}}%%
flowchart LR
    C1[Large File] --> S1[Chunk]
    C2[10+ Files] --> S2[Top 3]
    C3[Ambiguous] --> S3[Ask]
    C4[Cross-ref] --> S4[Search]
    C5[Conflict] --> S5[Include All]
    C6[Long Chat] --> S6[Summarize]
    style S1 fill:#276749,color:#c6f6d5
    style S2 fill:#276749,color:#c6f6d5
    style S3 fill:#276749,color:#c6f6d5
    style S4 fill:#276749,color:#c6f6d5
    style S5 fill:#276749,color:#c6f6d5
    style S6 fill:#276749,color:#c6f6d5
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

### Context Building Request Flow

```mermaid
%%{init: {'theme': 'base', 'themeVariables': { 'primaryColor': '#4a5568', 'primaryTextColor': '#e2e8f0', 'lineColor': '#a0aec0', 'actorBkg': '#2d3748'}}}%%
sequenceDiagram
    participant C as Client
    participant API as API
    participant CM as Context
    participant DB as Redis
    participant G as Gemini
    C->>API: POST /chat
    API->>CM: build_context
    CM->>DB: Get docs + conv
    CM-->>API: Context
    API->>G: Send
    G-->>API: Response
    API->>DB: Save
    API-->>C: Reply + stats
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

### Decision Matrix

```mermaid
%%{init: {'theme': 'base', 'themeVariables': { 'primaryColor': '#4a5568', 'primaryTextColor': '#e2e8f0', 'lineColor': '#a0aec0'}}}%%
flowchart LR
    E[Embedding] --> E1[Gemini API]
    V[Vector DB] --> V1[Redis VSS]
    S[Summarizer] --> S1[Gemini LLM]
    style E1 fill:#276749,color:#c6f6d5
    style V1 fill:#276749,color:#c6f6d5
    style S1 fill:#276749,color:#c6f6d5
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
