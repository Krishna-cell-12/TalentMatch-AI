import json
import csv
import time
import os
import sys
from pathlib import Path

import numpy as np
import faiss
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PARSED_JD_PATH = PROJECT_ROOT / "src" / "phase1" / "parsed_jd.json"
CANDIDATES_JSONL_PATH = PROJECT_ROOT / "data" / "raw" / "candidates.jsonl"
OUTPUT_CSV_PATH = PROJECT_ROOT / "submission.csv"
VALIDATOR_SCRIPT = PROJECT_ROOT / "data" / "raw" / "validate_submission.py"

MODEL_NAME = "all-MiniLM-L6-v2"
TOP_K = 100
BATCH_SIZE = 5000  # How many embeddings to hold in memory before pushing to FAISS


# ===================================================================
# Stage 1: Load & Flatten Job Description
# ===================================================================
def load_and_flatten_jd(jd_path: Path) -> str:
    with open(jd_path, "r", encoding="utf-8") as f:
        jd = json.load(f)

    parts = []
    parts.append(f"Role: {jd.get('role_title', '')}")
    parts.append(f"Seniority: {jd.get('seniority_level', '')}")

    must_have = jd.get("must_have_technical_skills", [])
    if must_have:
        parts.append(f"Must-have skills: {', '.join(must_have)}")

    nice_to_have = jd.get("nice_to_have_technical_skills", [])
    if nice_to_have:
        parts.append(f"Nice-to-have skills: {', '.join(nice_to_have)}")

    behavioral = jd.get("implicit_behavioral_signals", [])
    if behavioral:
        parts.append(f"Behavioral signals: {', '.join(behavioral)}")

    min_exp = jd.get("minimum_years_experience")
    if min_exp is not None:
        parts.append(f"Minimum experience: {min_exp} years")

    responsibilities = jd.get("core_responsibilities", [])
    if responsibilities:
        parts.append(f"Core responsibilities: {'; '.join(responsibilities)}")

    return ". ".join(parts) + "."

# ===================================================================
# Stage 2: Flatten Candidate Profile to Rich Text
# ===================================================================
def flatten_candidate(candidate: dict) -> str:
    parts = []
    
    # --- Profile ---
    profile = candidate.get("profile", {})
    headline = profile.get("headline", "")
    summary = profile.get("summary", "")
    current_title = profile.get("current_title", "")
    current_company = profile.get("current_company", "")
    current_industry = profile.get("current_industry", "")
    company_size = profile.get("current_company_size", "")
    yoe = profile.get("years_of_experience", 0)
    location = profile.get("location", "")
    country = profile.get("country", "")

    if headline: parts.append(f"Headline: {headline}")
    if summary: parts.append(f"Summary: {summary}")
    if current_title:
        role_str = f"Current role: {current_title}"
        if current_company: role_str += f" at {current_company}"
        if current_industry:
            role_str += f" ({current_industry}"
            if company_size: role_str += f", {company_size} employees"
            role_str += ")"
        parts.append(role_str)
    if yoe: parts.append(f"Total experience: {yoe} years")
    if location:
        loc_str = location
        if country: loc_str += f", {country}"
        parts.append(f"Location: {loc_str}")

    # --- Career History ---
    career = candidate.get("career_history", [])
    if career:
        career_parts = []
        for job in career:
            title = job.get("title", "")
            company = job.get("company", "")
            duration = job.get("duration_months", 0)
            desc = job.get("description", "")
            industry = job.get("industry", "")
            is_current = job.get("is_current", False)

            job_str = f"{title} at {company}"
            if duration: job_str += f" ({duration} months)"
            if industry: job_str += f" [{industry}]"
            if is_current: job_str += " [CURRENT]"
            if desc: job_str += f" — {desc}"
            career_parts.append(job_str)
        parts.append("Career: " + " | ".join(career_parts))

    # --- Skills ---
    skills = candidate.get("skills", [])
    if skills:
        skill_parts = []
        for skill in skills:
            name = skill.get("name", "")
            proficiency = skill.get("proficiency", "")
            duration = skill.get("duration_months", 0)
            s = f"{name} ({proficiency}"
            if duration: s += f", {duration}mo"
            s += ")"
            skill_parts.append(s)
        parts.append("Skills: " + ", ".join(skill_parts))

    return ". ".join(parts) + "."

# ===================================================================
# Stage 3: Score Normalization & Tie-Breaking
# ===================================================================
def normalize_and_rank(scores: np.ndarray, indices: np.ndarray, candidate_map: dict) -> list[dict]:
    results = []
    for score, idx in zip(scores, indices):
        cid = candidate_map[int(idx)].get("candidate_id", candidate_map[int(idx)].get("id", "UNKNOWN"))
        results.append({
            "candidate_id": cid,
            "raw_score": float(score),
            "index": int(idx),
        })

    raw_scores = [r["raw_score"] for r in results]
    min_score, max_score = min(raw_scores), max(raw_scores)
    score_range = max_score - min_score if max_score != min_score else 1.0

    for r in results:
        r["normalized_score"] = (r["raw_score"] - min_score) / score_range

    results.sort(key=lambda r: (-r["normalized_score"], r["candidate_id"]))

    for i, r in enumerate(results):
        r["rank"] = i + 1
        r["final_score"] = round(1.0 - (i * 0.008), 4) 

    return results

# ===================================================================
# Stage 4: Generate Reasoning & Export CSV
# ===================================================================
def count_matching_skills(candidate: dict, jd: dict) -> int:
    jd_skills = set()
    for skill in jd.get("must_have_technical_skills", []) + jd.get("nice_to_have_technical_skills", []):
        jd_skills.add(skill.lower())

    candidate_skills = set(skill.get("name", "").lower() for skill in candidate.get("skills", []))
    return len(jd_skills & candidate_skills)

def generate_reasoning(candidate: dict, jd: dict) -> str:
    profile = candidate.get("profile", {})
    title = profile.get("current_title", "Unknown")
    yoe = profile.get("years_of_experience", 0)
    signals = candidate.get("redrob_signals", {})
    response_rate = signals.get("recruiter_response_rate", 0.0)
    matching_skills = count_matching_skills(candidate, jd)

    return f"{title} with {yoe} yrs; {matching_skills} matching skills; response rate {response_rate:.2f}."

def export_csv(results: list[dict], candidate_map: dict, jd: dict, output_path: Path):
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])

        for r in results:
            candidate = candidate_map[r["index"]]
            reasoning = generate_reasoning(candidate, jd)
            writer.writerow([r["candidate_id"], r["rank"], f"{r['final_score']:.4f}", reasoning])


# ===================================================================
# Main Pipeline
# ===================================================================
def main():
    start_time = time.time()
    print("=" * 70)
    print("  PHASE 2: Memory-Safe Vector Search Pipeline")
    print("=" * 70)

    if not PARSED_JD_PATH.exists():
        print(f"  ERROR: {PARSED_JD_PATH} not found.")
        sys.exit(1)

    with open(PARSED_JD_PATH, "r", encoding="utf-8") as f:
        jd = json.load(f)
        
    jd_text = load_and_flatten_jd(PARSED_JD_PATH)
    
    print("\n[Stage 1] Loading sentence-transformers model...")
    model = SentenceTransformer(MODEL_NAME)
    jd_embedding = model.encode([jd_text], normalize_embeddings=True, convert_to_numpy=True)
    
    dim = model.get_sentence_embedding_dimension()
    index = faiss.IndexFlatIP(dim)

    # --- PASS 1: Stream, Encode, and Index ---
    print("\n[Stage 2] Pass 1: Streaming JSONL, generating vectors, and building FAISS...")
    total_lines = sum(1 for _ in open(CANDIDATES_JSONL_PATH, "r", encoding="utf-8"))
    
    batch_texts = []
    
    with open(CANDIDATES_JSONL_PATH, "r", encoding="utf-8") as f:
        for line in tqdm(f, total=total_lines, desc="  Indexing"):
            if not line.strip(): continue
            try:
                cand = json.loads(line)
                batch_texts.append(flatten_candidate(cand))
            except:
                continue
            
            if len(batch_texts) >= BATCH_SIZE:
                vecs = model.encode(batch_texts, batch_size=512, normalize_embeddings=True, show_progress_bar=False, convert_to_numpy=True)
                index.add(vecs)
                batch_texts = [] # Dump memory!
                
        # Final partial batch
        if batch_texts:
            vecs = model.encode(batch_texts, batch_size=512, normalize_embeddings=True, show_progress_bar=False, convert_to_numpy=True)
            index.add(vecs)

    print(f"\n[Stage 3] Executing FAISS Search...")
    scores, indices = index.search(jd_embedding, TOP_K)
    scores, indices = scores[0], indices[0]
    
    # --- PASS 2: Retrieve only the Top 100 Profiles ---
    print("\n[Stage 4] Pass 2: Retrieving full profiles for only the Top 100 matches...")
    top_indices_set = set(indices)
    top_candidate_map = {}
    
    with open(CANDIDATES_JSONL_PATH, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i in top_indices_set:
                top_candidate_map[i] = json.loads(line)

    print("\n[Stage 5] Normalizing scores and exporting CSV...")
    results = normalize_and_rank(scores, indices, top_candidate_map)
    export_csv(results, top_candidate_map, jd, OUTPUT_CSV_PATH)
    
    elapsed = time.time() - start_time
    print(f"\n✅ Pipeline completed safely in {elapsed:.1f} seconds. Output saved to {OUTPUT_CSV_PATH}")

    # --- Auto-validate ---
    if VALIDATOR_SCRIPT.exists():
        print(f"\n  Running validator...")
        import subprocess
        result = subprocess.run([sys.executable, str(VALIDATOR_SCRIPT), str(OUTPUT_CSV_PATH)], capture_output=True, text=True)
        print(f"  Validator stdout: {result.stdout.strip()}")
        if result.returncode == 0:
            print("  ✅ Submission is VALID!")
        else:
            print(f"  ❌ Submission FAILED. {result.stderr.strip()}")

if __name__ == "__main__":
    main()