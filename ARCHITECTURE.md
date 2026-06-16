# System Architecture — TalentMatch-AI

## Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                    INPUT: Job Description (JD)                       │
│                    + 100,000 Candidate Profiles (JSONL, 487MB)       │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│  PHASE 1: LLM-Powered JD Parsing                                    │
│  ─────────────────────────────                                      │
│  Model:   Groq Llama-3.3-70B (JSON mode)                           │
│  Input:   Raw JD text (sample_jd.txt)                               │
│  Process: Strip boilerplate → extract structured signals            │
│  Output:  parsed_jd.json                                            │
│           ├── role_title                                            │
│           ├── seniority_level                                       │
│           ├── must_have_technical_skills[]                          │
│           ├── nice_to_have_technical_skills[]                       │
│           ├── implicit_behavioral_signals[]                         │
│           ├── minimum_years_experience                              │
│           └── core_responsibilities[]                               │
└──────────────────────────────┬──────────────────────────────────────┘
                               │ parsed_jd.json
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│  PHASE 2: Memory-Safe Semantic Vector Search                        │
│  ──────────────────────────────────────────                         │
│  Model:   sentence-transformers/all-MiniLM-L6-v2 (384-dim)         │
│  Index:   FAISS IndexFlatIP (cosine via L2-normalized vectors)      │
│  Process:                                                           │
│    Pass 1: Stream JSONL in batches of 5,000                         │
│            → flatten candidate to rich text                         │
│            → encode to 384-dim vector                               │
│            → add to FAISS index                                     │
│    Search: jd_embedding.search(index, k=100)                        │
│    Pass 2: Re-read only top-100 profiles from JSONL                 │
│            → generate detailed reasoning strings                    │
│  Output:  submission.csv (100 rows: candidate_id, rank, score, ...) │
└──────────────────────────────┬──────────────────────────────────────┘
                               │ submission.csv (100 candidates)
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│  PHASE 3: Multi-Signal Hybrid Scoring Engine                        │
│  ───────────────────────────────────────────                        │
│  Formula:                                                           │
│    FinalScore = 0.45 × S_semantic                                   │
│               + 0.40 × S_metadata                                   │
│               + 0.15 × S_behavioral                                 │
│                                                                     │
│  S_semantic  : FAISS cosine similarity (Phase 2 score, [0,1])       │
│  S_metadata  : YoE compliance (60%) + title relevance (20%)        │
│              + skill overlap score (20%)                            │
│  S_behavioral: Recruiter response rate from Redrob signals          │
│                                                                     │
│  Output:  hybrid_shortlist.csv (Top 20 with all sub-scores)         │
└──────────────────────────────┬──────────────────────────────────────┘
                               │ hybrid_shortlist.csv (Top 20)
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│  PHASE 4: Expert LLM Re-Ranker & Submission Merger                  │
│  ─────────────────────────────────────────────────                  │
│  Model:   Groq Llama-3.3-70B (expert recruiter persona)            │
│  Process:                                                           │
│    • Load evaluation cache (skip already-scored candidates)         │
│    • Fetch full profiles for uncached Top-20                        │
│    • Call Groq API: JD + full profile → {score 0-100, notes}       │
│    • Fallback: if API limit hit → use hybrid score (no zeros)       │
│    • Sort Top-20 by LLM score                                       │
│    • Two-band score normalization:                                  │
│        Top-20  → [0.505, 1.000]  (LLM-driven)                      │
│        Tail-80 → [0.005, 0.500]  (Phase-2 driven)                  │
│    • Monotonicity safety pass                                       │
│  Output:                                                            │
│    submission.csv          (final 100-row ranked output)            │
│    final_ai_recruiter_report.csv (human-readable LLM notes)         │
└─────────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                OUTPUT: submission.csv                                │
│   Columns: candidate_id | rank | score | reasoning                  │
│   100 rows, score monotonically decreasing [0.005 → 1.000]         │
│   Passes official hackathon validator ✅                            │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Data Flow Diagram

```
candidates.jsonl (487MB)
      │
      ├──[Pass 1]──► sentence-transformers ──► FAISS index
      │                                              │
      │                                       FAISS search ◄── JD embedding
      │                                              │
      └──[Pass 2]──► Top-100 profiles ◄─────────────┘
                          │
                          ▼
                    submission.csv (Phase 2 baseline)
                          │
                    hybrid_scorer ◄── parsed_jd.json
                          │            (YoE, skills, title)
                          ▼
                   hybrid_shortlist.csv (Top 20)
                          │
                    llm_reranker ◄── Groq API (Llama-3.3-70B)
                          │
                          ▼
                    submission.csv (FINAL) + final_ai_recruiter_report.csv
```

---

## Technology Stack

| Component | Technology | Why |
|-----------|-----------|-----|
| LLM Backbone | Groq Llama-3.3-70B | State-of-art reasoning, fast inference, JSON mode |
| Embedding Model | all-MiniLM-L6-v2 | Proven quality/speed tradeoff for semantic search |
| Vector Index | FAISS IndexFlatIP | Exact cosine search, handles 100K vectors in RAM |
| Data Format | JSONL streaming | Memory-safe — avoids loading 487MB into RAM at once |
| Framework | Python (standard lib + numpy) | Minimal dependencies, reproducible |
| API Client | groq-python | Official SDK with rate-limit handling |

---

## Design Decisions

### Why Batch Streaming in Phase 2?
Loading 100,000 embeddings of 384 dimensions simultaneously requires ~145MB of float32 memory — manageable. However, loading all 487MB of raw JSONL + the text conversion overhead would exceed typical workstation RAM. The two-pass architecture processes candidates in batches of 5,000, pushing each batch into FAISS before clearing memory.

### Why the Two-Band Score System?
The official hackathon validator requires scores to be **monotonically non-increasing** by rank. A naive approach of mixing LLM scores (0-100 integers) with Phase-2 normalized scores (0.0-1.0) can easily violate this constraint. The two-band system guarantees the invariant mathematically: every LLM-evaluated candidate (top-20) always scores above every tail candidate (bottom-80).

### Why Hybrid Scoring Instead of Pure Semantic?
Semantic search finds conceptually similar candidates, but cannot count years of experience or assess recruiter responsiveness. A candidate mentioning "Python" and "FastAPI" in a personal project description gets the same semantic score as a 10-year veteran. Phase 3's metadata and behavioral signals correct this bias deterministically.

### Why LLM Re-ranking Only on Top 20?
LLMs are expensive and slow. Running Groq API calls on 100 candidates per submission would consume the daily token quota and add significant latency. The hybrid engine in Phase 3 is an efficient gatekeeper that narrows the field to the 20 most promising candidates for deep cognitive evaluation.
