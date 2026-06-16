"""
Phase 2: Memory-Safe Semantic Vector Search
==========================================
Embeds all 100,000 candidate profiles using sentence-transformers
(all-MiniLM-L6-v2) and indexes them in FAISS. Performs a semantic
similarity search against the parsed Job Description to retrieve the
Top 100 most relevant candidates.

Architecture:
  Pass 1 — Stream JSONL → flatten text → encode → push batches to FAISS
  Pass 2 — Re-read only the Top-100 profiles for reasoning generation

Output: submission.csv (candidate_id, rank, score, reasoning)
"""
import json
import csv
import time
import os
import sys
import subprocess
from pathlib import Path

import numpy as np
import faiss
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PROJECT_ROOT          = Path(__file__).resolve().parent.parent.parent
PARSED_JD_PATH        = PROJECT_ROOT / "src" / "phase1" / "parsed_jd.json"
CANDIDATES_JSONL_PATH = PROJECT_ROOT / "data" / "raw" / "candidates.jsonl"
OUTPUT_CSV_PATH       = PROJECT_ROOT / "submission.csv"
VALIDATOR_SCRIPT      = PROJECT_ROOT / "data" / "raw" / "validate_submission.py"

MODEL_NAME = "all-MiniLM-L6-v2"
TOP_K      = 100
BATCH_SIZE = 5000   # candidates encoded per batch before pushing to FAISS


# ===================================================================
# Stage 1: Load & Flatten Job Description into a rich query string
# ===================================================================
def load_and_flatten_jd(jd_path: Path) -> str:
    with open(jd_path, "r", encoding="utf-8") as f:
        jd = json.load(f)

    parts = [
        f"Role: {jd.get('role_title', '')}",
        f"Seniority: {jd.get('seniority_level', '')}",
    ]
    must_have = jd.get("must_have_technical_skills", [])
    if must_have:
        parts.append(f"Required skills: {', '.join(must_have)}")

    nice_to_have = jd.get("nice_to_have_technical_skills", [])
    if nice_to_have:
        parts.append(f"Preferred skills: {', '.join(nice_to_have)}")

    behavioral = jd.get("implicit_behavioral_signals", [])
    if behavioral:
        parts.append(f"Soft skills: {', '.join(behavioral)}")

    min_exp = jd.get("minimum_years_experience")
    if min_exp is not None:
        parts.append(f"Minimum {min_exp} years of professional experience required")

    responsibilities = jd.get("core_responsibilities", [])
    if responsibilities:
        parts.append(f"Key responsibilities: {'; '.join(responsibilities)}")

    return ". ".join(parts) + "."


# ===================================================================
# Stage 2: Flatten a Candidate Profile into a rich text representation
# ===================================================================
def flatten_candidate(candidate: dict) -> str:
    parts = []

    # ── Profile ──────────────────────────────────────────────────────
    profile          = candidate.get("profile", {})
    headline         = profile.get("headline", "")
    summary          = profile.get("summary", "")
    current_title    = profile.get("current_title", "")
    current_company  = profile.get("current_company", "")
    current_industry = profile.get("current_industry", "")
    company_size     = profile.get("current_company_size", "")
    yoe              = profile.get("years_of_experience", 0)
    location         = profile.get("location", "")
    country          = profile.get("country", "")

    if headline:      parts.append(f"Headline: {headline}")
    if summary:       parts.append(f"Summary: {summary}")
    if current_title:
        role_str = f"Current role: {current_title}"
        if current_company:  role_str += f" at {current_company}"
        if current_industry:
            role_str += f" ({current_industry}"
            if company_size: role_str += f", {company_size} employees"
            role_str += ")"
        parts.append(role_str)
    if yoe:           parts.append(f"Total experience: {yoe} years")
    if location:
        loc_str = location
        if country: loc_str += f", {country}"
        parts.append(f"Location: {loc_str}")

    # ── Career History ────────────────────────────────────────────────
    career = candidate.get("career_history", [])
    if career:
        career_parts = []
        for job in career:
            title    = job.get("title", "")
            company  = job.get("company", "")
            duration = job.get("duration_months", 0)
            desc     = job.get("description", "")
            industry = job.get("industry", "")
            is_curr  = job.get("is_current", False)

            job_str = f"{title} at {company}"
            if duration: job_str += f" ({duration} months)"
            if industry: job_str += f" [{industry}]"
            if is_curr:  job_str += " [CURRENT]"
            if desc:     job_str += f" — {desc}"
            career_parts.append(job_str)
        parts.append("Career: " + " | ".join(career_parts))

    # ── Skills (name + proficiency + duration) ────────────────────────
    skills = candidate.get("skills", [])
    if skills:
        skill_parts = []
        for skill in skills:
            name        = skill.get("name", "")
            proficiency = skill.get("proficiency", "")
            duration    = skill.get("duration_months", 0)
            s = f"{name} ({proficiency}"
            if duration: s += f", {duration}mo"
            s += ")"
            skill_parts.append(s)
        parts.append("Skills: " + ", ".join(skill_parts))

    return ". ".join(parts) + "."


# ===================================================================
# Stage 3: Score Normalization & Rank Assignment
# ===================================================================
def normalize_and_rank(
    scores: np.ndarray,
    indices: np.ndarray,
    candidate_map: dict,
) -> list[dict]:
    results = []
    for score, idx in zip(scores, indices):
        cid = candidate_map[int(idx)].get(
            "candidate_id", candidate_map[int(idx)].get("id", "UNKNOWN")
        )
        results.append({
            "candidate_id": cid,
            "raw_score":    float(score),
            "index":        int(idx),
        })

    raw_scores = [r["raw_score"] for r in results]
    min_score, max_score = min(raw_scores), max(raw_scores)
    score_range = (max_score - min_score) if max_score != min_score else 1.0

    for r in results:
        r["normalized_score"] = (r["raw_score"] - min_score) / score_range

    results.sort(key=lambda r: (-r["normalized_score"], r["candidate_id"]))

    for i, r in enumerate(results):
        r["rank"]        = i + 1
        r["final_score"] = round(1.0 - (i * 0.0095), 4)   # 1.0 → ~0.05 over 100 ranks

    return results


# ===================================================================
# Stage 4: Rich Reasoning Generation for submission CSV
# ===================================================================
def count_matching_skills(candidate: dict, jd: dict) -> tuple[int, list[str]]:
    """Returns (count, list_of_matched_skill_names)."""
    jd_skills = set()
    for skill in jd.get("must_have_technical_skills", []) + jd.get("nice_to_have_technical_skills", []):
        jd_skills.add(skill.lower())

    cand_skills = {s.get("name", "").lower(): s.get("name", "") for s in candidate.get("skills", [])}
    matched = [
        orig_name
        for jd_skill in jd_skills
        for cs_lower, orig_name in cand_skills.items()
        if jd_skill in cs_lower or cs_lower in jd_skill
    ]
    seen = set()
    unique_matched = [x for x in matched if not (x in seen or seen.add(x))]
    return len(unique_matched), unique_matched[:4]   # cap list at 4 for readability


def generate_reasoning(candidate: dict, jd: dict, rank: int) -> str:
    profile       = candidate.get("profile", {})
    title         = profile.get("current_title", "Unknown")
    yoe           = profile.get("years_of_experience", 0)
    signals       = candidate.get("redrob_signals", {})
    response_rate = signals.get("recruiter_response_rate", 0.0)
    n_matched, matched_skills = count_matching_skills(candidate, jd)

    min_yoe    = jd.get("minimum_years_experience", 0) or 0
    yoe_note   = "meets YoE requirement" if yoe >= min_yoe else f"below {min_yoe}yr requirement"
    skills_str = (f"matched skills: {', '.join(matched_skills)}"
                  if matched_skills else "no direct skill matches")

    return (
        f"{title} | {yoe}yrs XP ({yoe_note}) | "
        f"{n_matched} {skills_str} | "
        f"response rate {response_rate:.0%} | semantic rank #{rank}"
    )


def export_csv(results: list[dict], candidate_map: dict, jd: dict, output_path: Path):
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        for r in results:
            candidate = candidate_map[r["index"]]
            reasoning = generate_reasoning(candidate, jd, r["rank"])
            writer.writerow([
                r["candidate_id"],
                r["rank"],
                f"{r['final_score']:.4f}",
                reasoning,
            ])


# ===================================================================
# Main Pipeline
# ===================================================================
def main():
    start_time = time.time()
    print("=" * 70)
    print("  PHASE 2: Memory-Safe Semantic Vector Search Pipeline")
    print("=" * 70)

    if not PARSED_JD_PATH.exists():
        print(f"  ERROR: {PARSED_JD_PATH} not found. Run Phase 1 first.")
        sys.exit(1)

    with open(PARSED_JD_PATH, "r", encoding="utf-8") as f:
        jd = json.load(f)

    jd_text = load_and_flatten_jd(PARSED_JD_PATH)
    print(f"\n  JD query text: \"{jd_text[:120]}...\"")

    print("\n[Stage 1] Loading sentence-transformers model (all-MiniLM-L6-v2)...")
    model         = SentenceTransformer(MODEL_NAME)
    jd_embedding  = model.encode([jd_text], normalize_embeddings=True, convert_to_numpy=True)
    dim           = model.get_sentence_embedding_dimension()
    index         = faiss.IndexFlatIP(dim)   # Inner-product on L2-normalized vectors = cosine

    # ── PASS 1: Stream, encode, and index ───────────────────────────────────
    print("\n[Stage 2] Pass 1: Streaming JSONL → generating vectors → building FAISS index...")
    total_lines = sum(1 for _ in open(CANDIDATES_JSONL_PATH, "r", encoding="utf-8"))
    print(f"           {total_lines:,} candidate records detected.")

    batch_texts = []
    n_indexed   = 0

    with open(CANDIDATES_JSONL_PATH, "r", encoding="utf-8") as f:
        for line in tqdm(f, total=total_lines, desc="  Indexing"):
            if not line.strip():
                continue
            try:
                cand = json.loads(line)
                batch_texts.append(flatten_candidate(cand))
            except Exception:
                continue

            if len(batch_texts) >= BATCH_SIZE:
                vecs = model.encode(
                    batch_texts, batch_size=512,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                )
                index.add(vecs)
                n_indexed  += len(batch_texts)
                batch_texts = []   # free memory

    if batch_texts:
        vecs = model.encode(
            batch_texts, batch_size=512,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        index.add(vecs)
        n_indexed += len(batch_texts)

    print(f"\n           FAISS index built: {n_indexed:,} vectors (dim={dim})")

    # ── FAISS Search ─────────────────────────────────────────────────────────
    print(f"\n[Stage 3] Executing FAISS Top-{TOP_K} semantic search...")
    scores, indices = index.search(jd_embedding, TOP_K)
    scores, indices = scores[0], indices[0]
    print(f"           Score range: [{scores.min():.4f}, {scores.max():.4f}]")

    # ── PASS 2: Retrieve only Top-100 profiles ───────────────────────────────
    print(f"\n[Stage 4] Pass 2: Retrieving full profiles for Top {TOP_K} matches...")
    top_indices_set  = set(indices)
    top_candidate_map = {}

    with open(CANDIDATES_JSONL_PATH, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i in top_indices_set:
                try:
                    top_candidate_map[i] = json.loads(line)
                except Exception:
                    pass

    print(f"           {len(top_candidate_map)} profiles retrieved.")

    print("\n[Stage 5] Normalizing scores and exporting submission.csv...")
    results = normalize_and_rank(scores, indices, top_candidate_map)
    export_csv(results, top_candidate_map, jd, OUTPUT_CSV_PATH)

    elapsed = time.time() - start_time
    print(f"\n✅ Phase 2 complete in {elapsed:.1f}s — {TOP_K} candidates ranked → {OUTPUT_CSV_PATH}")

    # ── Auto-validate ────────────────────────────────────────────────────────
    if VALIDATOR_SCRIPT.exists():
        print(f"\n  Running official validator...")
        result = subprocess.run(
            [sys.executable, str(VALIDATOR_SCRIPT), str(OUTPUT_CSV_PATH)],
            capture_output=True, text=True,
        )
        if result.stdout.strip():
            print(f"  {result.stdout.strip()}")
        if result.returncode == 0:
            print("  ✅ Submission is VALID!")
        else:
            print(f"  ❌ Submission FAILED. {result.stderr.strip()}")


if __name__ == "__main__":
    main()