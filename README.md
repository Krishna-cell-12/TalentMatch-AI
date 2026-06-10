# 🚀 TalentMatch-AI
**From Keyword Matching to Cognitive Recruitment**

TalentMatch-AI is an Intelligent Candidate Discovery & Ranking pipeline built for a high-scale recruitment hackathon. It processes a massive 487MB dataset containing 100,000 candidate profiles in JSONL format and intelligently ranks them against a target Job Description. By combining fast mathematical indexing with deep cognitive reasoning, it moves beyond traditional keyword filters to understand the true semantic fit and hidden context of every candidate.

---

## ✨ Core Features

*   **Memory-Safe Data Streaming:** Built to process 100,000 dense JSONL profiles (487MB) without Out-Of-Memory (OOM) crashes on standard machines using a two-pass batching architecture.
*   **Semantic Vector Search:** Employs `sentence-transformers` and `FAISS` to cast a wide, intelligent net based on meaning, not just keywords.
*   **Mathematical Hybrid Grading Engine:** Layers hard constraints (YoE, titles) and behavioral signals (response rates) over raw semantic scores using a weighted, two-band mathematical formula to guarantee strict score monotonicity.
*   **Deep Cognitive Reasoning:** Utilizes Groq's `llama-3.3-70b-versatile` to act as an expert technical recruiter, catching context mismatches that algorithms miss (e.g., identifying a QA Engineer who happens to know backend keywords but lacks the required backend architecture experience).
*   **API-Resilient Caching:** Features exponential backoff handlers for rate limits and a local evaluation cache to bypass daily token constraints during iterative optimization runs.

---

## 🏗️ System Architecture

Our pipeline is organized into 4 modular phases to guarantee memory efficiency, mathematical rigor, and cognitive reasoning.

| Phase | Component | Technology | Input | Output | Description |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Phase 1** | **JD Parser** | Groq Llama-3.3 | Raw text JD | `data/raw/parsed_jd.json` | Strips corporate boilerplate and extracts structured signals (Must-Haves, YoE, Soft Skills). |
| **Phase 2** | **Vector Search** | `sentence-transformers`, `FAISS` | `parsed_jd.json`, `candidates.jsonl` | `submission.csv` (Baseline Top 100) | Generates dense embeddings (`all-MiniLM-L6-v2`) and performs memory-safe batch indexing for semantic retrieval. |
| **Phase 3** | **Hybrid Scoring** | Python / Math Engine | `submission.csv` | `hybrid_shortlist.csv` | Applies the formula: $FinalScore = w_1 \cdot S_{semantic} + w_2 \cdot S_{metadata} + w_3 \cdot S_{behavioral}$. Creates non-overlapping scoring bands. |
| **Phase 4** | **LLM Re-Ranker** | Groq Llama-3.3 | `hybrid_shortlist.csv` | `submission.csv` (Final), `final_ai_recruiter_report.csv` | Deep LLM evaluation of the Top 20. Caches results, merges with the tail 80, and auto-runs the official validator. |

### 📁 File Structure
```text
ai-candidate-ranker/
├── venv/
├── requirements.txt
├── .gitignore
├── submission.csv                # Official validated output
├── final_ai_recruiter_report.csv # Rich LLM analysis report
├── hybrid_shortlist.csv          # Phase 3 intermediate output
├── data/
│   └── raw/
│       ├── candidates.jsonl      # 100,000 raw candidate profiles (487MB)
│       ├── parsed_jd.json        # Output of Phase 1
│       └── validate_submission.py# Official judge validation script
└── src/
    ├── phase1/
    │   └── jd_parser.py          # Phase 1 script
    ├── phase2/
    │   └── real_vector_search.py # Phase 2 script
    ├── phase3/
    │   └── hybrid_scorer.py      # Phase 3 script
    └── phase4/
        └── llm_reranker.py       # Phase 4 script
```

---

## 💻 Installation Guide

Follow these steps to set up the environment, specifically tailored for WSL Ubuntu users.

1. **Clone the Repository**
   ```bash
   git clone https://github.com/yourusername/TalentMatch-AI.git
   cd TalentMatch-AI
   ```

2. **Create and Activate a Virtual Environment**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Environment Variables**
   Create a `.env` file in the root directory and add your Groq API key:
   ```env
   GROQ_API_KEY=your_groq_api_key_here
   ```

---

## 🚀 Execution Workflow

Run the pipeline sequentially to generate the final validated submission. Ensure your virtual environment is active.

**Step 1: Parse the Job Description**
```bash
python src/phase1/jd_parser.py
```

**Step 2: Execute Scale Vector Search**
*This will stream the 100,000 candidates and output the initial top 100.*
```bash
python src/phase2/real_vector_search.py
```

**Step 3: Apply Hybrid Mathematical Scoring**
*Integrates YoE, title matches, and behavioral signals into the score.*
```bash
python src/phase3/hybrid_scorer.py
```

**Step 4: Run Expert LLM Re-Ranking & Validation**
*Deep evaluates the top 20, merges the results, and automatically validates the final `submission.csv`.*
```bash
python src/phase4/llm_reranker.py
```

---

## 🏆 The Competitive Edge (Why It Wins)

TalentMatch-AI bridges the gap between raw computational speed and human-level discernment. 

Traditional recruitment filters fail because they rely on brittle string matching. A candidate might list "Python" and "PostgreSQL" from a bootcamp project, easily passing a keyword filter for a Senior Backend role, while a seasoned engineer with 10 years of "Distributed Systems Architecture" might be excluded because they didn't explicitly write the exact keywords.

Our approach solves this at scale:
1. **Fast Mathematical Indexing (FAISS):** We cast a wide semantic net. By embedding the entire JD and candidate profiles into dense vector space, we retrieve candidates who are conceptually aligned with the role, regardless of the specific vocabulary they used. It processes 100,000 profiles in seconds, efficiently narrowing the haystack down to a 100-candidate shortlist.
2. **Deterministic Constraint Layering:** Pure semantic search can't count years of experience or gauge responsiveness. Phase 3's hybrid engine forcefully applies hard business logic (YoE, title relevance) to penalize irrelevant matches while mathematically guaranteeing score monotonicity for the validator.
3. **Deep Cognitive Reasoning (Llama 3.3 70B):** The LLM acts as the ultimate gatekeeper. It reads the full career history of the top 20 candidates like a human recruiter would. It understands *context*—catching, for instance, that while a candidate has the required tech stack, their experience was entirely in QA rather than core backend development. 

By utilizing vector search for **scale** and Large Language Models for **precision**, TalentMatch-AI delivers an optimized, human-verified shortlist that pure algorithmic filters simply cannot match.
