# 🚀 TalentMatch-AI
### Intelligent Candidate Discovery & Ranking Engine
**India Runs Hackathon × Redrob AI — Track 1: The AI & Datathon Arena**

---

> **One command. 100,000 candidates. A ranked shortlist with explainable AI decisions.**

**Offline ranking (no API, no network — used for submission):**
```bash
python rank.py --candidates ./data/raw/candidates.jsonl --out ./submission.csv
```

**Full research pipeline (uses Groq API for LLM re-ranking):**
```bash
python run_all.py
```

---

## 🏆 What This Solves

Traditional recruitment tools miss brilliant candidates because they rely on brittle keyword matching. **TalentMatch-AI** combines three complementary intelligence layers to rank candidates by *true fit*, not just vocabulary overlap:

| Layer | Technology | What It Catches |
|-------|-----------|-----------------|
| **Semantic Search** | FAISS + sentence-transformers | Conceptual alignment regardless of specific keywords |
| **Mathematical Scoring** | Hybrid formula | YoE compliance, title relevance, skill overlap |
| **Cognitive Reasoning** | Groq Llama-3.3-70B | Context mismatches algorithms can't see |

---

## 📋 Hackathon Submission Checklist

| Requirement | Status | Details |
|-------------|--------|---------|
| ✅ Ranked candidate list | **Complete** | `submission.csv` — 100 candidates, monotonically scored |
| ✅ Explainable rankings | **Complete** | Specific reasoning per candidate (skill match, YoE, signals) |
| ✅ Clean, documented code | **Complete** | All scripts fully commented with docstrings |
| ✅ System architecture | **Complete** | `ARCHITECTURE.md` — pipeline diagram + design rationale |
| ✅ Offline ranking (`rank.py`) | **Complete** | No API, no GPU, runs in ~4 min on CPU |
| ✅ Honeypot detection | **Complete** | `rank.py` detects impossible profiles and penalizes them |
| ✅ `submission_metadata.yaml` | **Complete** | Filled template at repo root |
| ✅ Passes validator | **Complete** | Monotonically decreasing scores, auto-validated |
| 📹 Video demo | **Record locally** | See demo script below |

---

## 🏗️ System Architecture

```
Job Description (JD)
      +
100,000 Candidate Profiles (487MB JSONL)
         │
         ▼
┌─────────────────────────────────────────────────────────┐
│  Phase 1: LLM JD Parser                                 │
│  Groq Llama-3.3-70B → parsed_jd.json                   │
│  Extracts: must-have skills, YoE, seniority, goals      │
└──────────────────────────┬──────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│  Phase 2: Semantic Vector Search                        │
│  sentence-transformers (all-MiniLM-L6-v2) + FAISS      │
│  Streams 100K profiles in batches → cosine similarity  │
│  Output: Top 100 candidates (submission.csv)            │
└──────────────────────────┬──────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│  Phase 3: Hybrid Scoring Engine                         │
│  Score = 0.45×semantic + 0.40×metadata + 0.15×behavior │
│  Metadata = 60% YoE + 20% title + 20% skill overlap    │
│  Output: Top 20 shortlist (hybrid_shortlist.csv)        │
└──────────────────────────┬──────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│  Phase 4: Expert LLM Re-Ranker                          │
│  Groq Llama-3.3-70B acts as elite technical recruiter  │
│  Evaluates full career context of Top 20               │
│  Two-band scoring: top-20 → [0.505,1.000]              │
│                    tail-80 → [0.005,0.500]              │
│  Output: Final submission.csv + recruiter report        │
└─────────────────────────────────────────────────────────┘
```

> Full architecture documentation: **[ARCHITECTURE.md](./ARCHITECTURE.md)**

---

## 📁 Project Structure

```
ai-candidate-ranker/
├── run_all.py                          # One-shot pipeline runner ← START HERE
├── ARCHITECTURE.md                     # System architecture diagram + rationale
├── requirements.txt                    # Python dependencies
├── .env                                # GROQ_API_KEY (not committed)
│
├── submission.csv                      # ✅ FINAL hackathon submission
├── final_ai_recruiter_report.csv       # LLM recruiter notes for Top 20
├── hybrid_shortlist.csv                # Phase 3 intermediate (Top 20)
│
├── data/
│   └── raw/
│       ├── candidates.jsonl            # 100K candidate profiles (487MB)
│       ├── parsed_jd.json              # Structured JD (Phase 1 output)
│       └── validate_submission.py      # Official hackathon validator
│
└── src/
    ├── phase1/
    │   ├── jd_parser.py                # LLM-powered JD structured extraction
    │   └── sample_jd.txt              # Input: raw job description
    ├── phase2/
    │   └── real_vector_search.py      # FAISS semantic search (100K profiles)
    ├── phase3/
    │   └── hybrid_scorer.py           # Multi-signal hybrid scoring
    └── phase4/
        └── llm_reranker.py            # LLM deep evaluation + submission merger
```

---

## 💻 Quick Start

### 1. Prerequisites

- Python 3.10+
- `candidates.jsonl` placed at `data/raw/candidates.jsonl`
- Groq API key (free at [console.groq.com](https://console.groq.com))

### 2. Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/TalentMatch-AI.git
cd TalentMatch-AI

# Create virtual environment
python3 -m venv venv
source venv/bin/activate          # Linux/Mac
# venv\Scripts\activate           # Windows

# Install dependencies
pip install -r requirements.txt
```

### 3. Configure API Key

```bash
# Create .env file
echo "GROQ_API_KEY=your_groq_api_key_here" > .env
```

### 4. Run the Offline Ranker (Primary — No API needed)

```bash
python rank.py --candidates ./data/raw/candidates.jsonl --out ./submission.csv
```

**Expected output:**
```
══════════════════════════════════════════════════════════════════════
  TalentMatch-AI — Offline Ranking (rank.py)
  No API calls. No network. No GPU.
══════════════════════════════════════════════════════════════════════

  JD: 'Senior AI Engineer' | Min YoE: 5 | Must-have skills: 4
  Streaming candidates.jsonl...
  Processed 10,000 candidates (12.3s)...
  Processed 20,000 candidates (24.5s)...
  ...
  ✅ Done in ~3-4 min
  100,000 candidates scored | N honeypots detected
  Top 100 written to: submission.csv
```

### 5. Run the Full Research Pipeline (Optional — Uses Groq API)

```bash
python run_all.py
```

**Expected output:**
```
  [1/4] Phase 1: JD Parser (LLM Extraction)     ✅ ~3s
  [2/4] Phase 2: Semantic Vector Search (FAISS)  ✅ ~8 min
  [3/4] Phase 3: Hybrid Scoring Engine           ✅ ~30s
  [4/4] Phase 4: LLM Re-Ranker & Merger         ✅ ~2 min

  ✅ submission.csv       — 100 ranked candidates
  ✅ final_ai_recruiter_report.csv — LLM recruiter notes
```

### 6. Run Phases Individually

```bash
# Offline ranking ONLY (no API, used for submission)
python rank.py --candidates ./data/raw/candidates.jsonl --out ./submission.csv

# Phase 1 — Parse the Job Description (requires GROQ_API_KEY)
python src/phase1/jd_parser.py

# Phase 2 — Semantic search over 100K candidates (~8 min, requires GROQ_API_KEY)
python src/phase2/real_vector_search.py

# Phase 3 — Apply hybrid scoring, select Top 20
python src/phase3/hybrid_scorer.py

# Phase 4 — LLM re-rank Top 20, generate final submission (requires GROQ_API_KEY)
python src/phase4/llm_reranker.py
```

---

## 📊 Sample Output

`submission.csv` (first 5 rows):

| candidate_id | rank | score | reasoning |
|---|---|---|---|
| CAND_XXXXXXX | 1 | 1.0000 | Senior Backend Engineer \| 8yrs XP (meets requirement) \| 5 must-have skills matched: Python, FastAPI, PostgreSQL... |
| CAND_XXXXXXX | 2 | 0.9505 | Backend Engineer \| 6yrs XP (meets requirement) \| 4 matched skills \| response rate 87% |
| ... | ... | ... | ... |

`final_ai_recruiter_report.csv` (LLM recruiter notes):

| final_rank | candidate_id | title | llm_score | recruiter_notes |
|---|---|---|---|---|
| 1 | CAND_XXXXXXX | Senior Backend Engineer | 92 | Exceptional fit — 8 years of core Python/FastAPI development with microservices architecture experience. |
| 2 | CAND_XXXXXXX | Backend Engineer | 85 | Strong match — solid FastAPI and PostgreSQL background; slightly junior on distributed systems. |

---

## 🧠 Why This Pipeline Wins

### Problem: Keyword Matching Fails at Scale

A candidate who writes "developed RESTful services" is semantically equivalent to someone who writes "built FastAPI microservices" — but a keyword filter treats them as completely different. Meanwhile, a QA Engineer who lists "Python" and "PostgreSQL" from their test automation work passes a naive filter for a Senior Backend role.

### Our Solution: Three Complementary Intelligence Layers

**1. Semantic Search (Phase 2) — Scale without precision loss**
- Embeds the entire JD and all 100K candidate profiles into the same 384-dimensional vector space
- Uses cosine similarity to retrieve candidates who are *conceptually aligned* with the role
- Memory-safe batching handles 487MB of data without OOM crashes

**2. Hybrid Scoring (Phase 3) — Deterministic business logic**
- Forces hard YoE requirements (you need X years, not approximately X)
- Penalizes irrelevant role titles (a QA Engineer is not a Backend Engineer)
- Rewards active candidates with high recruiter response rates

**3. LLM Re-ranking (Phase 4) — Human-level contextual judgment**
- Reads the *full* career history of the top 20 candidates like a real recruiter would
- Understands context: a candidate with "Python" experience *entirely in QA automation* is not a backend developer
- Produces specific, actionable recruiter notes explaining each decision

### The Result

A ranked list where:
- **Top candidates** are genuinely the best technical matches — not keyword-stuffers
- **Every ranking is explainable** — judges can read exactly why each candidate was ranked where they were
- **The system is production-ready** — memory-safe, API-resilient, validator-passing

---

## 🎬 Video Demo Script

For the hackathon video submission, demonstrate:

1. **Show the JD** — open `src/phase1/sample_jd.txt`
2. **Run Phase 1** — `python src/phase1/jd_parser.py` → show `parsed_jd.json`
3. **Mention Phase 2** — "This phase takes ~8 minutes to process 100K candidates; here are the results"
4. **Show hybrid scores** — open `hybrid_shortlist.csv`, explain the sub-scores
5. **Run Phase 4** — `python src/phase4/llm_reranker.py` → show live LLM evaluation
6. **Show final output** — open `submission.csv` and `final_ai_recruiter_report.csv`
7. **Run validator** — `python data/raw/validate_submission.py submission.csv` → ✅

---

## 📦 Dependencies

```
groq>=0.4.0                 # Groq API client (Llama-3.3-70B)
sentence-transformers>=2.2.0 # Embedding model (all-MiniLM-L6-v2)
faiss-cpu>=1.7.0            # Vector similarity index
numpy>=1.24.0               # Numerical operations
tqdm>=4.60.0                # Progress bars
python-dotenv>=1.0.0        # .env file loading
pydantic>=2.0.0             # Data validation (Phase 1)
```

---

## 🔑 Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `GROQ_API_KEY` | Yes | Free API key from [console.groq.com](https://console.groq.com) |

---

*Built for India Runs Hackathon × Redrob AI × Hack2Skill*
*Track 1: The AI & Datathon Arena — Intelligent Candidate Discovery & Ranking*
