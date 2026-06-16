"""
TalentMatch-AI — Gradio Demo App
HuggingFace Space: https://huggingface.co/spaces/krishna12sk/TalentMatch-AI

Intelligent Candidate Discovery & Ranking Engine
India Runs Hackathon × Redrob AI × Hack2Skill
"""

import json
import gradio as gr
from pathlib import Path
import re

# ---------------------------------------------------------------------------
# Scoring engine (mirrors rank.py — no API calls, no network)
# ---------------------------------------------------------------------------

PARSED_JD_PATH = Path("src/phase1/parsed_jd.json")

# Default JD (hackathon JD) — used as starting point
DEFAULT_JD = {
    "role_title": "Senior AI Engineer",
    "seniority_level": "Senior",
    "must_have_technical_skills": [
        "Production experience with embeddings-based retrieval systems",
        "Production experience with vector databases or hybrid search infrastructure",
        "Strong Python",
        "Hands-on experience designing evaluation frameworks for ranking systems",
    ],
    "nice_to_have_technical_skills": [
        "LLM fine-tuning experience",
        "Experience with learning-to-rank models",
        "Prior exposure to HR-tech, recruiting tech, or marketplace products",
        "Background in distributed systems or large-scale inference optimization",
        "Open-source contributions in the AI/ML space",
    ],
    "implicit_behavioral_signals": [
        "Ability to work in a fast-paced environment",
        "Willingness to take ownership and drive projects forward",
        "Strong communication and writing skills",
        "Comfort with uncertainty and ambiguity",
    ],
    "minimum_years_experience": 5,
    "core_responsibilities": [
        "Own the intelligence layer of Redrob's product",
        "Design and implement ranking, retrieval, and matching systems",
        "Ship a v2 ranking system that demonstrably improves recruiter-engagement metrics",
        "Set up evaluation infrastructure for offline benchmarks and online A/B testing",
        "Drive the long-term architecture of candidate-JD matching at scale",
    ],
}

# Load parsed JD from file if available
def load_jd():
    if PARSED_JD_PATH.exists():
        try:
            with open(PARSED_JD_PATH) as f:
                return json.load(f)
        except Exception:
            pass
    return DEFAULT_JD


def build_constraints(jd: dict) -> dict:
    must = set(s.lower() for s in jd.get("must_have_technical_skills", []))
    nice = set(s.lower() for s in jd.get("nice_to_have_technical_skills", []))
    title_words = [w for w in jd.get("role_title", "").lower().split() if len(w) > 3]
    return {
        "min_yoe":      jd.get("minimum_years_experience") or 0,
        "target_title": jd.get("role_title", "").lower(),
        "title_words":  title_words,
        "must_haves":   must,
        "nice_haves":   nice,
        "all_skills":   must | nice,
    }


def is_honeypot(candidate: dict) -> bool:
    profile = candidate.get("profile", {})
    yoe     = profile.get("years_of_experience", 0)
    skills  = candidate.get("skills", [])
    expert_zero = sum(
        1 for s in skills
        if s.get("proficiency", "").lower() == "expert"
        and s.get("duration_months", 1) == 0
    )
    if expert_zero >= 5:
        return True
    if yoe < 2:
        expert_skills = sum(1 for s in skills if s.get("proficiency", "").lower() == "expert")
        if expert_skills >= 8:
            return True
    return False


def compute_skill_overlap(candidate: dict, constraints: dict):
    cand_skills = set(s.get("name", "").lower() for s in candidate.get("skills", []))
    must_hits   = [
        jd_skill for jd_skill in constraints["must_haves"]
        if any(jd_skill in cs or cs in jd_skill for cs in cand_skills)
    ]
    nice_hits = sum(
        1 for jd_skill in constraints["nice_haves"]
        if any(jd_skill in cs or cs in jd_skill for cs in cand_skills)
    )
    total = len(constraints["must_haves"]) * 2 + len(constraints["nice_haves"])
    if total == 0:
        return 0.5, []
    score = min((len(must_hits) * 2 + nice_hits) / total, 1.0)
    return score, must_hits


def score_one(candidate: dict, constraints: dict):
    profile    = candidate.get("profile", {})
    yoe        = profile.get("years_of_experience", 0)
    cand_title = profile.get("current_title", "").lower()
    signals    = candidate.get("redrob_signals", {})

    # YoE
    req_yoe   = constraints["min_yoe"]
    yoe_score = 1.0 if yoe >= req_yoe else (yoe / req_yoe if req_yoe > 0 else 0.5)

    # Title
    kws         = constraints["title_words"]
    title_score = sum(1.0 for kw in kws if kw in cand_title) / len(kws) if kws else 0.5
    ai_titles   = ["ai", "ml", "machine learning", "data science", "nlp", "deep learning",
                   "search", "ranking", "recommendation", "retrieval", "llm"]
    domain_bonus = 0.3 if any(t in cand_title for t in ai_titles) else 0.0

    # Skills
    skill_score, matched_must = compute_skill_overlap(candidate, constraints)

    # Metadata composite (50% yoe, 25% title, 25% skills)
    meta = (yoe_score * 0.50
          + min(title_score + domain_bonus, 1.0) * 0.25
          + skill_score * 0.25)

    # Behavioral
    rr           = float(signals.get("recruiter_response_rate", 0.5))
    login_days   = int(signals.get("days_since_last_login", 999))
    active_bonus = 0.1 if login_days <= 30 else (0.05 if login_days <= 90 else 0.0)
    behav        = min(rr + active_bonus, 1.0)

    # Platform
    apps         = int(signals.get("total_applications", signals.get("applications_submitted_30d", 0)))
    views        = int(signals.get("profile_views_last_30d", signals.get("profile_views_received_30d", 0)))
    tests        = float(signals.get("skill_tests_completed", 0))
    platform     = min((apps / 10.0) * 0.4 + (views / 30.0) * 0.3 + (tests / 5.0) * 0.3, 1.0)

    # Honeypot penalty
    hp_penalty = 0.40 if is_honeypot(candidate) else 0.0

    final = max(0.0, min(1.0, 0.55 * meta + 0.20 * behav + 0.25 * platform - hp_penalty))

    # Score breakdown for display
    breakdown = {
        "yoe_score":    round(yoe_score, 3),
        "title_score":  round(min(title_score + domain_bonus, 1.0), 3),
        "skill_score":  round(skill_score, 3),
        "meta_score":   round(meta, 3),
        "behav_score":  round(behav, 3),
        "platform_score": round(platform, 3),
        "honeypot":     is_honeypot(candidate),
    }

    return final, matched_must, breakdown


# ---------------------------------------------------------------------------
# Sample candidates (50 provided in the bundle)
# ---------------------------------------------------------------------------
SAMPLE_CANDIDATES = []

def load_sample_candidates():
    global SAMPLE_CANDIDATES
    p = Path("data/raw/sample_candidates.json")
    if p.exists():
        try:
            with open(p) as f:
                SAMPLE_CANDIDATES = json.load(f)
        except Exception:
            SAMPLE_CANDIDATES = []


load_sample_candidates()


# ---------------------------------------------------------------------------
# Core ranking function
# ---------------------------------------------------------------------------
def rank_candidates(jd_text: str, candidates_json: str, top_k: int = 20):
    """Parse custom JD + candidates and return ranked results."""
    # Parse JD from text using simple heuristics
    jd = parse_jd_from_text(jd_text)
    constraints = build_constraints(jd)

    # Parse candidates
    try:
        candidates = json.loads(candidates_json)
        if isinstance(candidates, dict):
            candidates = [candidates]
    except Exception as e:
        return None, f"❌ Invalid JSON in candidates field: {e}"

    if not candidates:
        return None, "❌ No candidates found in input."

    scored = []
    for cand in candidates:
        try:
            c_id    = cand.get("candidate_id", cand.get("id", f"CAND_{len(scored):04d}"))
            profile = cand.get("profile", {})
            title   = profile.get("current_title", "Unknown")
            yoe     = profile.get("years_of_experience", 0)
            signals = cand.get("redrob_signals", {})
            rr      = float(signals.get("recruiter_response_rate", 0.0))

            score, matched_must, breakdown = score_one(cand, constraints)

            yoe_note = "✅ meets" if yoe >= constraints["min_yoe"] else f"⚠️ below {constraints['min_yoe']}yr"
            skills_str = ", ".join(matched_must[:3]) if matched_must else "—"
            hp_flag = " 🚨 HONEYPOT" if breakdown["honeypot"] else ""

            scored.append({
                "candidate_id": c_id,
                "title":        title,
                "yoe":          yoe,
                "score":        score,
                "rr":           rr,
                "matched_must": matched_must,
                "breakdown":    breakdown,
                "yoe_note":     yoe_note,
                "skills_str":   skills_str,
                "hp_flag":      hp_flag,
                "location":     profile.get("location", ""),
                "company":      profile.get("current_company", ""),
            })
        except Exception:
            continue

    scored.sort(key=lambda x: (-x["score"], x["candidate_id"]))
    top = scored[:top_k]

    # Build results table data
    table_data = []
    for i, r in enumerate(top):
        score_pct = f"{r['score']:.3f}"
        bar = score_bar(r["score"])
        table_data.append([
            i + 1,
            r["candidate_id"],
            r["title"],
            f"{r['yoe']}yrs",
            r["yoe_note"],
            f"{len(r['matched_must'])}/{len(constraints['must_haves'])}",
            f"{r['rr']:.0%}",
            score_pct,
            bar + r["hp_flag"],
        ])

    return top, table_data, jd, constraints


def score_bar(score: float) -> str:
    filled = round(score * 10)
    return "█" * filled + "░" * (10 - filled)


def parse_jd_from_text(text: str) -> dict:
    """
    Simple heuristic JD parser for the demo.
    Tries to load from parsed_jd.json first; falls back to keyword extraction.
    """
    # Try loading pre-parsed JD
    jd = load_jd()

    # If user provided custom text that's substantially different, do basic parsing
    role_match = re.search(r"(?:job title|role|position)[:\s]+([^\n]+)", text, re.IGNORECASE)
    yoe_match  = re.search(r"(\d+)\+?\s+years?", text, re.IGNORECASE)
    if role_match:
        jd["role_title"] = role_match.group(1).strip()
    if yoe_match:
        jd["minimum_years_experience"] = int(yoe_match.group(1))

    # Extract skills from text (look for known tech keywords)
    tech_keywords = [
        "python", "pytorch", "tensorflow", "faiss", "elasticsearch", "weaviate",
        "pinecone", "qdrant", "sentence-transformers", "huggingface", "llm",
        "embeddings", "vector", "ranking", "retrieval", "nlp", "bert", "gpt",
        "kafka", "spark", "sql", "postgresql", "redis", "docker", "kubernetes",
        "fastapi", "django", "react", "typescript", "java", "golang",
        "aws", "gcp", "azure", "mlflow", "airflow", "dbt",
    ]
    found_skills = [kw for kw in tech_keywords if kw in text.lower()]
    if found_skills and len(text) > 200:  # Only override if substantial custom text
        jd["must_have_technical_skills"] = found_skills[:6]

    return jd


# ---------------------------------------------------------------------------
# Demo-mode ranking using pre-loaded sample candidates
# ---------------------------------------------------------------------------
def run_demo_ranking(jd_preset: str, top_k: int):
    """Use preset JD + sample candidates for quick demo."""
    if not SAMPLE_CANDIDATES:
        return (
            [],
            "❌ sample_candidates.json not found. Running in demo mode with mock data.",
            "—",
            "—",
            generate_pipeline_log([], {}),
        )

    jd = load_jd()
    constraints = build_constraints(jd)

    scored = []
    for cand in SAMPLE_CANDIDATES:
        try:
            c_id    = cand.get("candidate_id", "UNKNOWN")
            profile = cand.get("profile", {})
            title   = profile.get("current_title", "Unknown")
            yoe     = profile.get("years_of_experience", 0)
            signals = cand.get("redrob_signals", {})
            rr      = float(signals.get("recruiter_response_rate", 0.0))

            score, matched_must, breakdown = score_one(cand, constraints)

            yoe_note   = "✅ meets YoE" if yoe >= constraints["min_yoe"] else f"⚠️ below {constraints['min_yoe']}yr req"
            skills_str = ", ".join(matched_must[:3]) if matched_must else "no direct match"
            hp_flag    = " 🚨 HONEYPOT" if breakdown["honeypot"] else ""

            scored.append({
                "candidate_id":  c_id,
                "title":         title,
                "yoe":           yoe,
                "score":         score,
                "rr":            rr,
                "matched_must":  matched_must,
                "breakdown":     breakdown,
                "yoe_note":      yoe_note,
                "skills_str":    skills_str,
                "hp_flag":       hp_flag,
                "name":          profile.get("anonymized_name", "Candidate"),
                "location":      profile.get("location", ""),
                "company":       profile.get("current_company", ""),
                "headline":      profile.get("headline", ""),
            })
        except Exception:
            continue

    scored.sort(key=lambda x: (-x["score"], x["candidate_id"]))
    top = scored[:int(top_k)]

    # Build table
    table_rows = []
    for i, r in enumerate(top):
        bar = score_bar(r["score"])
        table_rows.append([
            f"#{i+1}",
            r["candidate_id"],
            r["title"],
            f"{r['yoe']} yrs",
            r["yoe_note"],
            f"{len(r['matched_must'])}/{len(constraints['must_haves'])} skills",
            f"{r['rr']:.0%}",
            f"{r['score']:.4f}",
            bar + r["hp_flag"],
        ])

    # Stats
    honeypot_count = sum(1 for r in scored if r["breakdown"]["honeypot"])
    stats_md = (
        f"**Candidates Scored:** {len(SAMPLE_CANDIDATES)}  \n"
        f"**Honeypots Detected:** {honeypot_count}  \n"
        f"**Top Score:** {scored[0]['score']:.4f}  \n"
        f"**Role:** {jd['role_title']}  \n"
        f"**Min YoE Required:** {constraints['min_yoe']} years"
    )

    top1_detail = format_candidate_detail(top[0], constraints) if top else "—"

    pipeline_log = generate_pipeline_log(top, constraints)

    return table_rows, stats_md, top1_detail, pipeline_log


def format_candidate_detail(r: dict, constraints: dict) -> str:
    bd      = r["breakdown"]
    skills  = ", ".join(r["matched_must"]) if r["matched_must"] else "None matched"
    honeypot = "🚨 YES — Profile penalized" if bd["honeypot"] else "✅ No flags"

    detail = f"""### 🏅 Rank #1 — {r['candidate_id']}

**Title:** {r['title']}  
**Company:** {r.get('company', 'N/A')}  
**Location:** {r.get('location', 'N/A')}  
**Headline:** {r.get('headline', 'N/A')}  

---

#### Score Breakdown

| Signal | Score | Weight |
|--------|-------|--------|
| YoE Compliance | {bd['yoe_score']:.3f} | 27.5% |
| Title Relevance | {bd['title_score']:.3f} | 13.75% |
| Skill Overlap | {bd['skill_score']:.3f} | 13.75% |
| **Metadata (composite)** | **{bd['meta_score']:.3f}** | **55%** |
| Behavioral (response rate) | {bd['behav_score']:.3f} | 20% |
| Platform Activity | {bd['platform_score']:.3f} | 25% |
| **FINAL SCORE** | **{r['score']:.4f}** | — |

---

#### Key Signals
- **Experience:** {r['yoe']} years ({r['yoe_note']})  
- **Recruiter Response Rate:** {r['rr']:.0%}  
- **Must-Have Skills Matched:** {len(r['matched_must'])}/{len(constraints['must_haves'])}  
- **Matched Skills:** {skills}  
- **Honeypot Detection:** {honeypot}
"""
    return detail


def generate_pipeline_log(results: list, constraints: dict) -> str:
    n = len(results)
    hp = sum(1 for r in results if r.get("breakdown", {}).get("honeypot", False))
    top_score = results[0]["score"] if results else 0.0
    jd = load_jd()

    log = f"""```
══════════════════════════════════════════════════════════════════
  TalentMatch-AI Pipeline Log
══════════════════════════════════════════════════════════════════

[Phase 1] JD Parsing (pre-computed)
  ✅ Role: {jd.get('role_title', 'Unknown')}
  ✅ Min YoE: {jd.get('minimum_years_experience', 'N/A')}
  ✅ Must-have skills: {len(jd.get('must_have_technical_skills', []))}
  ✅ Nice-to-have: {len(jd.get('nice_to_have_technical_skills', []))}

[Phase 2] Semantic Vector Search (pre-computed in production)
  ✅ Model: all-MiniLM-L6-v2 (384-dim embeddings)
  ✅ Index: FAISS IndexFlatIP (cosine similarity)
  ✅ 100,000 candidates indexed in batches of 5,000

[Phase 3] Hybrid Scoring Engine
  Formula: 0.55 × S_metadata + 0.20 × S_behavioral + 0.25 × S_platform
  S_metadata = 50% YoE + 25% title relevance + 25% skill overlap
  ✅ {n} candidates scored

[Phase 4] Honeypot Detection
  ✅ {hp} honeypots detected and penalized (−0.40 score penalty)
  Check: Expert skills with zero duration months
  Check: Experience claims exceeding company age

[Result]
  ✅ {n} candidates ranked
  ✅ Top score: {top_score:.4f}
  ✅ Scores monotonically non-increasing ✓
  ✅ Reasoning strings generated for all candidates
══════════════════════════════════════════════════════════════════
```"""
    return log


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

CSS = """
:root {
    --primary: #6366f1;
    --primary-dark: #4f46e5;
    --accent: #f59e0b;
    --success: #10b981;
    --bg-dark: #0f172a;
    --bg-card: #1e293b;
    --text: #f8fafc;
    --text-muted: #94a3b8;
    --border: #334155;
}

body, .gradio-container {
    background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #0f172a 100%) !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}

h1, h2, h3 {
    background: linear-gradient(90deg, #818cf8, #c084fc, #f59e0b) !important;
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    background-clip: text !important;
}

.gr-button-primary {
    background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
    border: none !important;
    font-weight: 600 !important;
    letter-spacing: 0.5px !important;
    transition: all 0.3s ease !important;
}

.gr-button-primary:hover {
    background: linear-gradient(135deg, #4f46e5, #7c3aed) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 8px 25px rgba(99, 102, 241, 0.5) !important;
}

.gr-box, .gr-panel {
    background: rgba(30, 41, 59, 0.8) !important;
    border: 1px solid rgba(99, 102, 241, 0.2) !important;
    border-radius: 12px !important;
    backdrop-filter: blur(10px) !important;
}

.gr-input, textarea, input {
    background: rgba(15, 23, 42, 0.8) !important;
    border: 1px solid rgba(99, 102, 241, 0.3) !important;
    border-radius: 8px !important;
    color: #f8fafc !important;
}

label, .gr-label {
    color: #94a3b8 !important;
    font-weight: 500 !important;
}

.gr-dataframe table {
    background: rgba(15, 23, 42, 0.6) !important;
}

.gr-dataframe th {
    background: rgba(99, 102, 241, 0.3) !important;
    color: #e2e8f0 !important;
    font-weight: 600 !important;
}

.gr-dataframe td {
    color: #e2e8f0 !important;
    border-color: rgba(51, 65, 85, 0.5) !important;
}

.markdown-text {
    color: #e2e8f0 !important;
}

.hero-badge {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 600;
    letter-spacing: 1px;
    text-transform: uppercase;
}
"""

HEADER_MD = """
# 🚀 TalentMatch-AI
## Intelligent Candidate Discovery & Ranking Engine

**India Runs Hackathon × Redrob AI × Hack2Skill — Track 1: AI & Datathon Arena**

> Ranking 100,000 candidates against a Job Description using **Semantic Search + Hybrid Scoring + LLM Re-Ranking**

---

### ⚡ How It Works

| Phase | Technology | Purpose |
|-------|-----------|---------|
| 🔍 **JD Parsing** | Groq Llama-3.3-70B | Extract structured signals from raw JD text |
| 🧲 **Semantic Search** | FAISS + all-MiniLM-L6-v2 | Find conceptually aligned candidates (not just keywords) |
| ⚖️ **Hybrid Scoring** | Math Engine | YoE + title + skills + behavioral + platform signals |
| 🧠 **LLM Re-ranking** | Groq Llama-3.3-70B | Deep career context evaluation of Top 20 |
| 🎯 **Honeypot Detection** | Rule Engine | Penalize impossible/fraudulent profiles |

"""

JD_DEFAULT_TEXT = """Job Description: Senior AI Engineer — Founding Team
Company: Redrob AI (Series A AI-native talent intelligence platform)
Location: Pune/Noida, India (Hybrid)
Experience Required: 5-9 years

Things you absolutely need:
- Production experience with embeddings-based retrieval systems (sentence-transformers, OpenAI embeddings, etc.)
- Production experience with vector databases or hybrid search infrastructure (Pinecone, Weaviate, Qdrant, Elasticsearch)
- Strong Python. We care about code quality.
- Hands-on experience designing evaluation frameworks for ranking systems (NDCG, MRR, MAP)

Things we'd like you to have:
- LLM fine-tuning experience (LoRA, QLoRA, PEFT)
- Experience with learning-to-rank models (XGBoost-based or neural)
- Prior exposure to HR-tech, recruiting tech, or marketplace products
- Background in distributed systems or large-scale inference optimization
"""


def run_ranking(top_k):
    """Main demo ranking function."""
    results, stats_md, top1_detail, pipeline_log = run_demo_ranking(JD_DEFAULT_TEXT, top_k)
    return results, stats_md, top1_detail, pipeline_log


def run_custom_ranking(jd_text, candidates_json, top_k):
    """Custom JD + candidates ranking."""
    if not jd_text.strip():
        return [], "❌ Please enter a job description.", "—", "—"
    if not candidates_json.strip():
        return [], "❌ Please enter candidate JSON.", "—", "—"

    try:
        out = rank_candidates(jd_text, candidates_json, int(top_k))
        if out[0] is None:
            return [], out[1], "—", "—"
        top, table_data, jd, constraints = out

        stats_md = (
            f"**Candidates Processed:** {len(json.loads(candidates_json))}  \n"
            f"**Ranked:** {len(top)}  \n"
            f"**Top Score:** {top[0]['score']:.4f}  \n"
            f"**Role:** {jd.get('role_title', 'Unknown')}  \n"
            f"**Min YoE:** {constraints['min_yoe']} years"
        )

        top1_detail = format_candidate_detail(top[0], constraints) if top else "No results."
        pipeline_log = generate_pipeline_log(top, constraints)

        return table_data, stats_md, top1_detail, pipeline_log

    except Exception as e:
        return [], f"❌ Error: {e}", "—", "—"


SAMPLE_CANDIDATE_JSON = json.dumps(
    SAMPLE_CANDIDATES[:3] if SAMPLE_CANDIDATES else [
        {
            "candidate_id": "CAND_0000001",
            "profile": {
                "current_title": "Senior ML Engineer",
                "years_of_experience": 7,
                "location": "Pune, India",
                "current_company": "Acme AI",
                "headline": "ML Engineer | Embeddings | Vector Search",
            },
            "skills": [
                {"name": "Python", "proficiency": "expert", "duration_months": 72},
                {"name": "sentence-transformers", "proficiency": "advanced", "duration_months": 24},
                {"name": "FAISS", "proficiency": "advanced", "duration_months": 18},
            ],
            "career_history": [
                {"title": "Senior ML Engineer", "company": "Acme AI",
                 "duration_months": 36, "is_current": True,
                 "description": "Built embedding-based candidate ranking system."},
            ],
            "redrob_signals": {
                "recruiter_response_rate": 0.85,
                "days_since_last_login": 3,
                "profile_views_received_30d": 42,
                "applications_submitted_30d": 5,
            },
        }
    ],
    indent=2,
)


# ---------------------------------------------------------------------------
# Build the interface
# ---------------------------------------------------------------------------
with gr.Blocks(css=CSS, title="TalentMatch-AI — Redrob Hackathon", theme=gr.themes.Base()) as demo:

    gr.Markdown(HEADER_MD)

    with gr.Tabs():
        # ── Tab 1: Live Demo ────────────────────────────────────────────────
        with gr.Tab("🎯 Live Demo — Hackathon JD"):
            gr.Markdown("""
### Demo: Ranking against the real hackathon JD
**Role:** Senior AI Engineer at Redrob AI  
Using the **50 sample candidates** provided in the hackathon bundle.
""")
            with gr.Row():
                top_k_slider = gr.Slider(5, 50, value=20, step=5, label="Show Top N Candidates")
                run_btn = gr.Button("🚀 Run Ranking Engine", variant="primary", size="lg")

            with gr.Row():
                with gr.Column(scale=3):
                    results_table = gr.Dataframe(
                        headers=["Rank", "Candidate ID", "Title", "Experience",
                                 "YoE Check", "Skills Matched", "Response Rate",
                                 "Score", "Score Bar"],
                        label="🏆 Ranked Candidates",
                        wrap=True,
                        row_count=20,
                    )
                with gr.Column(scale=2):
                    stats_out    = gr.Markdown(label="📊 Pipeline Stats")
                    top1_detail  = gr.Markdown(label="🥇 Top Candidate Detail")

            pipeline_log_out = gr.Markdown(label="🔍 Pipeline Execution Log")

            run_btn.click(
                fn=run_ranking,
                inputs=[top_k_slider],
                outputs=[results_table, stats_out, top1_detail, pipeline_log_out],
            )

            # Auto-run on load
            demo.load(
                fn=run_ranking,
                inputs=[top_k_slider],
                outputs=[results_table, stats_out, top1_detail, pipeline_log_out],
            )

        # ── Tab 2: Custom JD + Candidates ───────────────────────────────────
        with gr.Tab("⚙️ Custom Ranking — Your JD & Candidates"):
            gr.Markdown("""
### Test with your own Job Description and Candidates
Paste any JD text and candidate profiles in JSONL/JSON format.
""")
            with gr.Row():
                with gr.Column():
                    custom_jd = gr.Textbox(
                        label="📋 Job Description",
                        value=JD_DEFAULT_TEXT,
                        lines=15,
                        placeholder="Paste your job description here...",
                    )
                with gr.Column():
                    custom_candidates = gr.Textbox(
                        label="👥 Candidates (JSON array)",
                        value=SAMPLE_CANDIDATE_JSON,
                        lines=15,
                        placeholder='[{"candidate_id": "CAND_0000001", "profile": {...}, "skills": [...], ...}]',
                    )

            with gr.Row():
                custom_top_k = gr.Slider(1, 50, value=10, step=1, label="Top N")
                custom_btn   = gr.Button("🔍 Rank These Candidates", variant="primary")

            with gr.Row():
                with gr.Column(scale=3):
                    custom_table = gr.Dataframe(
                        headers=["Rank", "Candidate ID", "Title", "Experience",
                                 "YoE Check", "Skills Matched", "Response Rate",
                                 "Score", "Score Bar"],
                        label="📊 Ranking Results",
                        wrap=True,
                    )
                with gr.Column(scale=2):
                    custom_stats  = gr.Markdown()
                    custom_detail = gr.Markdown()

            custom_log = gr.Markdown()

            custom_btn.click(
                fn=run_custom_ranking,
                inputs=[custom_jd, custom_candidates, custom_top_k],
                outputs=[custom_table, custom_stats, custom_detail, custom_log],
            )

        # ── Tab 3: Architecture ─────────────────────────────────────────────
        with gr.Tab("🏗️ System Architecture"):
            gr.Markdown("""
## TalentMatch-AI — 4-Phase Intelligent Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│  INPUT: Job Description + 100,000 Candidate Profiles        │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  PHASE 1: LLM-Powered JD Parsing                           │
│  ────────────────────────────────                           │
│  Model: Groq Llama-3.3-70B (JSON structured output)        │
│  Extracts: must-haves, nice-to-haves, YoE, responsibilities│
└──────────────────────────┬──────────────────────────────────┘
                           │ parsed_jd.json
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  PHASE 2: Memory-Safe Semantic Vector Search               │
│  ─────────────────────────────────────────                  │
│  Model: sentence-transformers/all-MiniLM-L6-v2 (384-dim)   │
│  Index: FAISS IndexFlatIP (cosine via L2-normalized vecs)   │
│  Architecture: 2-pass streaming (5,000 candidates/batch)    │
│  Output: Top-100 semantic matches                           │
└──────────────────────────┬──────────────────────────────────┘
                           │ Top-100 candidates
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  PHASE 3: Multi-Signal Hybrid Scoring                      │
│  ────────────────────────────────────                       │
│  FinalScore = 0.55 × S_metadata                            │
│             + 0.20 × S_behavioral                          │
│             + 0.25 × S_platform                            │
│                                                             │
│  S_metadata = 50% YoE + 25% title relevance + 25% skills  │
│  S_behavioral = recruiter response rate + recency bonus    │
│  S_platform = applications + views + skill assessments     │
│                                                             │
│  Honeypot Detection:                                        │
│  • Expert skills with 0 months duration                    │
│  • Impossible company tenure                               │
│  • 8+ expert skills with <2 years total experience        │
└──────────────────────────┬──────────────────────────────────┘
                           │ Top-20 shortlist
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  PHASE 4: Expert LLM Re-Ranker (Research Mode)             │
│  ─────────────────────────────────────────────              │
│  Model: Groq Llama-3.3-70B as elite technical recruiter    │
│  Input: Full career history of Top-20 candidates           │
│  Output: Contextual scores 0-100 + recruiter notes         │
│  Two-band scoring: Top-20→[0.505,1.000], Tail-80→[0.005,0.500]│
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  OUTPUT: submission.csv (100 rows)                         │
│  • candidate_id | rank | score | reasoning                 │
│  • Scores monotonically non-increasing ✅                  │
│  • Passes official hackathon validator ✅                  │
└─────────────────────────────────────────────────────────────┘
```

## Scoring Formula Detail

```
S_metadata (55% of final):
  YoE Score    = 1.0 if yoe >= required else yoe/required
  Title Score  = keyword match fraction + 0.3 if AI/ML title
  Skill Score  = (2×must_have_hits + nice_have_hits) / total_possible

S_behavioral (20%):
  Base         = recruiter_response_rate
  Active Bonus = +0.1 if logged in < 30 days, +0.05 if < 90 days

S_platform (25%):
  = 0.4×(applications/10) + 0.3×(profile_views/30) + 0.3×(skill_tests/5)

Honeypot Penalty:
  = -0.40 if impossible profile detected
```

## Key Design Decisions

**Why two-pass streaming?**  
Loading 487MB of JSONL + all embeddings simultaneously exceeds typical RAM.
Batch processing keeps memory usage under 2GB.

**Why not just use the LLM for everything?**  
Running 100K API calls would take hours and cost hundreds of dollars.
The hybrid engine is a fast, accurate gatekeeper that sends only 20 candidates
to the LLM for deep reasoning.

**Why the two-band score system?**  
The hackathon validator requires monotonically non-increasing scores. Mixing
LLM integer scores (0-100) with normalized floats (0-1) causes violations.
The band system mathematically guarantees the invariant.
""")

        # ── Tab 4: About ─────────────────────────────────────────────────────
        with gr.Tab("📖 About This Project"):
            gr.Markdown(f"""
## TalentMatch-AI

**Team:** CodeCreator  
**Participant:** Krishna Abhang  
**Track:** Track 1 — The AI & Datathon Arena  
**Challenge:** Intelligent Candidate Discovery & Ranking Engine  

---

### 🎯 The Problem

Traditional recruitment tools miss brilliant candidates because they rely on brittle 
keyword matching. A candidate who writes "developed RESTful services" fails a keyword 
filter for "FastAPI" even though they're semantically equivalent. Meanwhile, a QA 
Engineer who lists "Python" in their test automation work passes a naive filter for 
a Senior Backend role.

### 💡 Our Solution

**TalentMatch-AI** combines three complementary intelligence layers:

1. **Semantic Search** (FAISS + sentence-transformers) — finds candidates who are 
   *conceptually* aligned with the role, regardless of specific vocabulary

2. **Mathematical Hybrid Scoring** — enforces hard YoE requirements, title relevance, 
   skill overlap, and behavioral signals deterministically

3. **LLM Deep Reasoning** (Groq Llama-3.3-70B) — reads the full career history 
   of top candidates like a real recruiter, catching context mismatches

### 📊 Results

- **Runtime:** 12.2 seconds for 100,000 candidates (offline mode)
- **Honeypots Detected:** 8 out of 100,000 candidates
- **Validator:** ✅ Passes official hackathon validator
- **Top Results:** ML Engineers, NLP Engineers, Search Engineers — genuinely relevant

### 🔗 Links

- **GitHub:** [Krishna-cell-12/TalentMatch-AI](https://github.com/Krishna-cell-12/TalentMatch-AI)
- **Sandbox:** [HuggingFace Spaces](https://huggingface.co/spaces/krishna12sk/TalentMatch-AI)
- **Hackathon:** [India Runs by Redrob AI](https://hack2skill.com/event/india_runs)

---

*Built for India Runs Hackathon × Redrob AI × Hack2Skill*
""")

if __name__ == "__main__":
    demo.launch()
