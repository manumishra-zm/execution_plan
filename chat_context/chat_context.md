# Smart Context Management System - Execution Plan

> **Status**: Planning
> **Last Updated**: 2024-02-06
> **Estimated Effort**: 12-16 days

---

## Executive Summary

Design an intelligent context management system that efficiently handles multiple file uploads across chat sessions while **reducing token usage by 60-70%** and maintaining contextual accuracy.

### Key Outcomes
- **Token Reduction**: ~15,000 → ~5,000 tokens per request (with 10 docs)
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
flowchart LR
    subgraph Current["Current Flow (Inefficient)"]
        A[User Upload] --> B[OCR Process]
        B --> C[Store JSON/MD]
        C --> D[User Query]
        D --> E["Pass ALL Documents"]
        E --> F[LLM - Gemini]
        F --> G[Response]
    end

    style E fill:#ff6b6b,color:#fff
    style Current fill:#fff5f5
```

### Problem Visualization

```mermaid
graph TD
    subgraph msg1["Message 1"]
        F1[File 1<br/>1500 tokens]
        F2[File 2<br/>1200 tokens]
        F3[File 3<br/>1800 tokens]
    end

    subgraph msg2["Message 2"]
        F4[File 4-11<br/>8000 tokens]
    end

    subgraph msg3["Message 3"]
        F5[File 12-16<br/>5000 tokens]
    end

    subgraph context["Context Sent to LLM"]
        ALL["ALL FILES<br/>17,500+ tokens<br/>💸 High Cost"]
    end

    msg1 --> context
    msg2 --> context
    msg3 --> context

    style ALL fill:#ff6b6b,color:#fff
    style context fill:#fff5f5
```

### Token Growth Problem

| Message | Files Uploaded | Cumulative Files | Tokens Sent | Cost Impact |
|---------|---------------|------------------|-------------|-------------|
| 1 | 3 | 3 | 4,500 | $0.01 |
| 2 | 8 | 11 | 12,500 | $0.03 |
| 3 | 5 | 16 | 17,500 | $0.04 |
| 4 | 0 (query only) | 16 | 17,500 | $0.04 |
| 5 | 2 | 18 | 20,000 | $0.05 |
| ... | ... | ... | **Grows linearly** | 💸💸💸 |

**Problem**: Every message sends ALL historical documents, even when irrelevant.

---

## Target Architecture

### High-Level Architecture

```mermaid
flowchart TB
    subgraph Upload["📤 Upload Pipeline"]
        U1[File Upload] --> U2[OCR Processing]
        U2 --> U3[Generate Summary]
        U3 --> U4[Extract Metadata]
        U4 --> U5[Generate Embedding]
        U5 --> U6[Store in Index]
    end

    subgraph Storage["💾 Smart Storage"]
        S1[(Document Index<br/>Redis)]
        S2[(Vector Store<br/>Embeddings)]
        S3[(Conversation<br/>Memory)]
    end

    subgraph Query["🔍 Query Pipeline"]
        Q1[User Query] --> Q2[Analyze Intent]
        Q2 --> Q3[Score Relevance]
        Q3 --> Q4[Select Documents]
        Q4 --> Q5[Build Context]
        Q5 --> Q6[Token Budget Check]
    end

    subgraph LLM["🤖 LLM Processing"]
        L1[Optimized Context] --> L2[Gemini LLM]
        L2 --> L3[Response]
    end

    U6 --> S1
    U5 --> S2
    S1 --> Q3
    S2 --> Q3
    S3 --> Q5
    Q6 --> L1

    style Upload fill:#e3f2fd
    style Storage fill:#fff3e0
    style Query fill:#e8f5e9
    style LLM fill:#fce4ec
```

### Detailed Component Architecture

```mermaid
flowchart TB
    subgraph Frontend["Frontend"]
        FE1[Chat UI]
        FE2[File Upload]
    end

    subgraph API["API Layer"]
        API1["/chat endpoint"]
        API2["/upload endpoint"]
    end

    subgraph Services["Backend Services"]
        subgraph ContextManager["Context Manager"]
            CM1[Document Indexer]
            CM2[Relevance Scorer]
            CM3[Context Builder]
            CM4[Token Manager]
        end

        subgraph Embedding["Embedding Service"]
            EM1[Embedder<br/>Gemini API]
            EM2[Vector Search]
        end

        subgraph Memory["Memory Service"]
            MM1[Conversation Memory]
            MM2[Message Summarizer]
        end

        subgraph OCR["OCR Service"]
            OCR1[File Router]
            OCR2[Text Extractor]
        end
    end

    subgraph Storage["Data Layer"]
        DB1[(Redis<br/>Documents + Vectors)]
        DB2[(File System<br/>JSON/MD)]
    end

    subgraph External["External"]
        EXT1[Gemini LLM]
        EXT2[Gemini Embeddings]
    end

    FE1 --> API1
    FE2 --> API2
    API1 --> CM3
    API2 --> OCR1
    OCR1 --> CM1
    CM1 --> EM1
    CM3 --> CM2
    CM2 --> EM2
    CM3 --> MM1
    CM3 --> CM4
    CM4 --> EXT1
    EM1 --> EXT2
    CM1 --> DB1
    EM1 --> DB1
    OCR1 --> DB2
```

---

## Context Management Strategy

### Document Lifecycle State Machine

```mermaid
stateDiagram-v2
    [*] --> Uploading: User uploads file

    Uploading --> Processing: Upload complete
    Processing --> Indexed: OCR + Index complete

    Indexed --> Active: In active window
    Active --> Referenced: Explicitly mentioned
    Referenced --> Active: Time passes

    Active --> Summarized: Exits active window
    Summarized --> Retrieved: Query matches
    Retrieved --> Active: Promoted back

    Summarized --> Archived: Session ages
    Archived --> Retrieved: Deep search match

    note right of Active: Full content included
    note right of Summarized: Summary only
    note right of Archived: Metadata only
```

### Context Selection Algorithm

```mermaid
flowchart TD
    START[User Query + Attachments] --> A{New files<br/>attached?}

    A -->|Yes| B[Add to FULL context]
    A -->|No| C[Skip]

    B --> D[Calculate relevance scores]
    C --> D

    D --> E{Explicit<br/>reference?}
    E -->|"'invoice.pdf'"| F[Add to FULL context]
    E -->|No| G[Check semantic similarity]

    G --> H{Similarity > 0.7?}
    H -->|Yes| I[Add to FULL context]
    H -->|No| J{Similarity > 0.4?}

    J -->|Yes| K[Add SUMMARY only]
    J -->|No| L[Add METADATA only]

    F --> M[Token Budget Check]
    I --> M
    K --> M
    L --> M

    M --> N{Under budget?}
    N -->|Yes| O[Include in context]
    N -->|No| P[Downgrade content level]

    P --> Q{Can downgrade?}
    Q -->|Yes| R[FULL→SUMMARY→METADATA]
    Q -->|No| S[Exclude oldest/least relevant]

    R --> M
    S --> O

    O --> END[Build Final Context]

    style B fill:#4caf50,color:#fff
    style F fill:#4caf50,color:#fff
    style I fill:#4caf50,color:#fff
    style K fill:#ff9800,color:#fff
    style L fill:#9e9e9e,color:#fff
```

### Sliding Window Strategy

```mermaid
gantt
    title Conversation Context Window
    dateFormat X
    axisFormat %s

    section Active Window
    Message N (Current)     :active, 0, 1
    Message N-1             :active, 1, 2
    Message N-2             :active, 2, 3

    section Summary Window
    Message N-3             :done, 3, 4
    Message N-4             :done, 4, 5
    Message N-5             :done, 5, 6

    section Archive
    Messages 1 to N-6       :crit, 6, 10
```

### Token Budget Allocation

```mermaid
pie title Token Budget Distribution (8000 total)
    "System Prompt" : 500
    "Conversation (Active)" : 1500
    "Conversation (Summaries)" : 500
    "Documents (Full)" : 2000
    "Documents (Summaries)" : 800
    "Documents (Metadata)" : 200
    "User Query" : 500
    "Safety Buffer" : 2000
```

### Content Level Comparison

```mermaid
flowchart LR
    subgraph Full["FULL (~1500 tokens)"]
        F1["Complete OCR text<br/>All tables<br/>All details"]
    end

    subgraph Summary["SUMMARY (~150 tokens)"]
        S1["2-3 sentence summary<br/>Key entities<br/>Main amounts"]
    end

    subgraph Meta["METADATA (~40 tokens)"]
        M1["Filename, type<br/>Date, entities list"]
    end

    Full -->|"Budget pressure"| Summary
    Summary -->|"More pressure"| Meta

    style Full fill:#4caf50,color:#fff
    style Summary fill:#ff9800,color:#fff
    style Meta fill:#9e9e9e,color:#fff
```

---

## Data Models

### Entity Relationship Diagram

```mermaid
erDiagram
    SESSION ||--o{ MESSAGE_GROUP : contains
    SESSION ||--o{ DOCUMENT_INDEX : contains
    MESSAGE_GROUP ||--o{ DOCUMENT_INDEX : references
    DOCUMENT_INDEX ||--|| EMBEDDING : has

    SESSION {
        uuid session_id PK
        uuid user_id FK
        uuid tenant_id FK
        datetime created_at
        datetime updated_at
        int message_count
    }

    MESSAGE_GROUP {
        int group_idx PK
        uuid session_id FK
        string user_message
        string assistant_response
        string summary
        int token_count
        datetime created_at
    }

    DOCUMENT_INDEX {
        uuid doc_id PK
        uuid session_id FK
        int message_group_idx FK
        string filename
        string document_type
        string full_content
        string summary
        json metadata
        int token_count
        datetime uploaded_at
    }

    EMBEDDING {
        uuid doc_id PK
        vector embedding
        datetime created_at
    }
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

```mermaid
gantt
    title Implementation Timeline
    dateFormat YYYY-MM-DD

    section Phase 1
    Document Indexing Foundation    :p1, 2024-02-07, 3d

    section Phase 2
    Basic Context Builder           :p2, after p1, 3d

    section Phase 3
    Semantic Search                 :p3, after p1, 4d

    section Phase 4
    Conversation Memory             :p4, after p2, 3d

    section Phase 5
    Optimization & Edge Cases       :p5, after p3 p4, 3d

    section Milestones
    MVP Ready                       :milestone, after p2, 0d
    Full Feature                    :milestone, after p5, 0d
```

### Phase Dependencies

```mermaid
flowchart TD
    P1[Phase 1<br/>Document Indexing]
    P2[Phase 2<br/>Context Builder]
    P3[Phase 3<br/>Semantic Search]
    P4[Phase 4<br/>Conversation Memory]
    P5[Phase 5<br/>Optimization]

    P1 --> P2
    P1 --> P3
    P2 --> P4
    P3 --> P5
    P4 --> P5

    MVP{MVP<br/>Functional}
    FULL{Full<br/>Feature}

    P2 --> MVP
    P5 --> FULL

    style P1 fill:#e3f2fd
    style P2 fill:#e8f5e9
    style P3 fill:#fff3e0
    style P4 fill:#fce4ec
    style P5 fill:#f3e5f5
    style MVP fill:#4caf50,color:#fff
    style FULL fill:#2196f3,color:#fff
```

---

## Phase 1: Document Indexing Foundation

**Goal**: Store document metadata and summaries on upload

**Duration**: 2-3 days

### Sequence Diagram

```mermaid
sequenceDiagram
    participant U as User
    participant FE as Frontend
    participant API as Upload API
    participant OCR as OCR Service
    participant IDX as Document Indexer
    participant SUM as Summarizer
    participant DB as Redis

    U->>FE: Upload file
    FE->>API: POST /upload
    API->>OCR: Process file
    OCR-->>API: Extracted content

    API->>IDX: Index document
    IDX->>SUM: Generate summary
    SUM-->>IDX: Summary text

    IDX->>IDX: Extract metadata
    IDX->>IDX: Count tokens
    IDX->>DB: Store DocumentIndex

    DB-->>IDX: Stored
    IDX-->>API: Index complete
    API-->>FE: Response + doc_index
    FE-->>U: Upload complete
```

### Tasks

- [ ] **1.1** Create data models (`DocumentIndex`, `DocumentMetadata`)
- [ ] **1.2** Create `DocumentIndexService` with CRUD operations
- [ ] **1.3** Implement `Summarizer` service (Gemini API)
- [ ] **1.4** Implement metadata extraction (entities, amounts, dates)
- [ ] **1.5** Create `TokenCounter` utility
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
sequenceDiagram
    participant U as User
    participant API as Chat API
    participant CB as Context Builder
    participant RS as Relevance Scorer
    participant TM as Token Manager
    participant MEM as Memory Service
    participant DB as Redis
    participant LLM as Gemini

    U->>API: Send message
    API->>CB: Build context

    CB->>DB: Get session documents
    DB-->>CB: Document list

    CB->>RS: Score relevance
    RS->>RS: Filename match
    RS->>RS: Keyword match
    RS->>RS: Recency score
    RS-->>CB: Scored documents

    CB->>CB: Sort by relevance
    CB->>CB: Select content levels

    CB->>MEM: Get conversation history
    MEM->>DB: Fetch messages
    DB-->>MEM: Message groups
    MEM-->>CB: History context

    CB->>TM: Check token budget
    TM->>TM: Count all tokens
    TM-->>CB: Budget status

    alt Over budget
        CB->>CB: Downgrade content levels
        CB->>TM: Recheck budget
    end

    CB-->>API: Optimized context
    API->>LLM: Send to Gemini
    LLM-->>API: Response
    API-->>U: Chat response
```

### Tasks

- [ ] **2.1** Create `ContextBuilder` service
- [ ] **2.2** Implement `RelevanceScorer` (filename + keyword matching)
- [ ] **2.3** Implement sliding window for conversation
- [ ] **2.4** Implement `TokenBudgetManager`
- [ ] **2.5** Create context assembly logic (full → summary → metadata)
- [ ] **2.6** Integrate into chat endpoint
- [ ] **2.7** Add context metrics to response (for debugging)

### Files to Create

```
app/services/context_manager/
├── context_builder.py        # Main context assembly
├── relevance_scorer.py       # Document relevance scoring
├── token_budget_manager.py   # Budget allocation & enforcement
└── context_assembler.py      # Assemble final context string
```

---

## Phase 3: Semantic Search

**Goal**: Find relevant documents by meaning

**Duration**: 3-4 days

### Sequence Diagram

```mermaid
sequenceDiagram
    participant IDX as Document Indexer
    participant EMB as Embedder
    participant VS as Vector Store
    participant RS as Relevance Scorer
    participant API as Gemini Embedding API

    Note over IDX,API: On Document Upload
    IDX->>EMB: Generate embedding
    EMB->>API: Embed text
    API-->>EMB: Vector [768 dims]
    EMB->>VS: Store vector
    VS-->>EMB: Stored

    Note over RS,API: On Query
    RS->>EMB: Embed query
    EMB->>API: Embed text
    API-->>EMB: Query vector
    EMB->>VS: Search similar
    VS-->>EMB: Top K results
    EMB-->>RS: Similar documents
```

### Tasks

- [ ] **3.1** Create `Embedder` service (Gemini embedding API)
- [ ] **3.2** Create `VectorStore` service (Redis VSS)
- [ ] **3.3** Generate embeddings on document upload
- [ ] **3.4** Implement semantic similarity search
- [ ] **3.5** Enhance `RelevanceScorer` with semantic signals
- [ ] **3.6** Add query embedding at request time
- [ ] **3.7** Benchmark embedding latency

### Files to Create

```
app/services/embedding/
├── __init__.py
├── embedder.py               # Generate embeddings via Gemini
├── vector_store.py           # Redis VSS operations
└── similarity_search.py      # Search & rank by similarity
```

---

## Phase 4: Conversation Memory Enhancement

**Goal**: Efficient long conversation handling

**Duration**: 2-3 days

### Sequence Diagram

```mermaid
sequenceDiagram
    participant API as Chat API
    participant MEM as Memory Service
    participant SUM as Summarizer
    participant DB as Redis

    Note over API,DB: After each message
    API->>MEM: Save message group
    MEM->>DB: Store full content

    MEM->>MEM: Check window size

    alt Window exceeded
        MEM->>SUM: Summarize old group
        SUM-->>MEM: Summary
        MEM->>DB: Update with summary
        MEM->>DB: Clear full content
    end

    Note over API,DB: On context build
    API->>MEM: Get conversation
    MEM->>DB: Fetch groups
    DB-->>MEM: All groups

    MEM->>MEM: Active: full content
    MEM->>MEM: History: summaries only
    MEM-->>API: Optimized history
```

### Tasks

- [ ] **4.1** Enhance `ConversationMemoryService` with message groups
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

## Phase 5: Optimization & Edge Cases

**Goal**: Production-ready robustness

**Duration**: 2-3 days

### Edge Case Handling Flow

```mermaid
flowchart TD
    subgraph LargeFile["Large File (>10k tokens)"]
        LF1[Detect large file] --> LF2[Chunk into sections]
        LF2 --> LF3[Embed each chunk]
        LF3 --> LF4[Retrieve relevant chunks only]
    end

    subgraph ManyFiles["Many Files (10+)"]
        MF1[Count files] --> MF2[Index all]
        MF2 --> MF3[Include top 3 full]
        MF3 --> MF4[Rest as metadata]
    end

    subgraph Ambiguous["Ambiguous Reference"]
        AR1["'that file'"] --> AR2[No filename match]
        AR2 --> AR3[Use recency + semantic]
        AR3 --> AR4{Confidence > 0.8?}
        AR4 -->|Yes| AR5[Use best match]
        AR4 -->|No| AR6[Ask clarification]
    end

    subgraph CrossRef["Cross-Message Reference"]
        CR1["'file from earlier'"] --> CR2[Search all session docs]
        CR2 --> CR3[Score by relevance]
        CR3 --> CR4[Promote to active]
    end
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

## Edge Cases & Solutions

```mermaid
flowchart LR
    subgraph Cases["Edge Cases"]
        C1[Very Large File]
        C2[10+ Files]
        C3[Ambiguous Reference]
        C4[Cross-Message Ref]
        C5[Conflicting Info]
        C6[Long Conversation]
    end

    subgraph Solutions["Solutions"]
        S1[Chunk + Selective Retrieval]
        S2[Top 3 Full + Rest Metadata]
        S3[Recency + Semantic + Ask]
        S4[Session-wide Search]
        S5[Include All Relevant]
        S6[Aggressive Summarization]
    end

    C1 --> S1
    C2 --> S2
    C3 --> S3
    C4 --> S4
    C5 --> S5
    C6 --> S6
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
sequenceDiagram
    participant Client
    participant ChatAPI
    participant ContextManager
    participant Redis
    participant Gemini

    Client->>ChatAPI: POST /chat {message, session_id}

    ChatAPI->>ContextManager: build_context(session_id, message)

    ContextManager->>Redis: Get documents
    ContextManager->>Redis: Get conversation
    ContextManager->>ContextManager: Score & select
    ContextManager->>ContextManager: Assemble context
    ContextManager-->>ChatAPI: BuiltContext

    ChatAPI->>Gemini: Send optimized context
    Gemini-->>ChatAPI: Response

    ChatAPI->>Redis: Save message group
    ChatAPI-->>Client: Response + context_stats
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
flowchart TD
    subgraph Embedding["Embedding Model"]
        E1[Gemini Embedding API]
        E2[Local: sentence-transformers]
    end

    subgraph Vector["Vector Store"]
        V1[Redis VSS]
        V2[Qdrant]
        V3[Pinecone]
    end

    subgraph Summary["Summarization"]
        S1[Gemini LLM]
        S2[Local: BART/T5]
    end

    E1 -->|"Recommended"| REC1[Consistency with LLM<br/>High quality<br/>~$0.0001/1k tokens]
    V1 -->|"Recommended"| REC2[Already using Redis<br/>No new infra<br/>Good for <100k docs]
    S1 -->|"Recommended"| REC3[Best quality<br/>Already integrated<br/>~$0.00025/summary]

    style REC1 fill:#4caf50,color:#fff
    style REC2 fill:#4caf50,color:#fff
    style REC3 fill:#4caf50,color:#fff
```

### Recommended Stack

| Component | Choice | Rationale |
|-----------|--------|-----------|
| **Embedding Model** | Gemini Embedding API | Consistency, quality, low cost |
| **Vector Store** | Redis VSS | Already in stack, sufficient scale |
| **Summarization** | Gemini LLM | Quality, already integrated |
| **Token Counter** | tiktoken | Accuracy for Gemini tokenizer |

### Cost Estimation

```mermaid
pie title Estimated Monthly Cost (1000 conversations)
    "Embedding Generation" : 5
    "Summarization" : 15
    "LLM Queries (Optimized)" : 80
```

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

```mermaid
xychart-beta
    title "Token Usage: Current vs Target"
    x-axis ["1 doc", "5 docs", "10 docs", "20 docs"]
    y-axis "Tokens per Request" 0 --> 25000
    bar [1500, 7500, 15000, 22500]
    line [1500, 3000, 5000, 7000]
```

| Metric | Current | Target | Improvement |
|--------|---------|--------|-------------|
| Tokens/request (10 docs) | ~15,000 | ~5,000 | 67% ↓ |
| Context build latency | N/A | < 500ms | - |
| Relevant doc accuracy | N/A | > 90% | - |
| Cost/conversation (20 msgs) | ~$0.50 | ~$0.15 | 70% ↓ |
| Max conversation length | ~20 msgs | 100+ msgs | 5x ↑ |

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

1. **Review & Approve** this plan
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
| Filename match | 0.4 | Query: "invoice.pdf" → invoice.pdf |
| Explicit reference | 0.3 | Query: "the first file" → file at index 1 |
| Keyword overlap | 0.2 | Query: "total amount" → docs with "total" |
| Semantic similarity | 0.3 | Query: "how much to pay" → invoice docs |
| Recency | 0.1 | Newer docs score higher |

---

*Document Version: 1.0*
*Status: Ready for Review*
