import json
import csv
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration & Weights
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PHASE2_CSV_PATH = PROJECT_ROOT / "submission.csv"
CANDIDATES_JSONL_PATH = PROJECT_ROOT / "data" / "raw" / "candidates.jsonl"
PARSED_JD_PATH = PROJECT_ROOT / "src" / "phase1" / "parsed_jd.json"
OUTPUT_CSV_PATH = PROJECT_ROOT / "hybrid_shortlist.csv"

# The Hybrid Formula Weights (must sum to 1.0)
W_SEMANTIC = 0.45
W_METADATA = 0.40
W_BEHAVIORAL = 0.15

# ===================================================================
# Helper Functions
# ===================================================================
def load_jd_constraints() -> dict:
    """Extracts hard constraints from the parsed JD."""
    with open(PARSED_JD_PATH, "r", encoding="utf-8") as f:
        jd = json.load(f)
    return {
        "min_yoe": jd.get("minimum_years_experience") or 0,
        "target_title": jd.get("role_title", "").lower()
    }

def fetch_phase2_results() -> dict:
    """Loads Phase 2 results and normalizes semantic scores to [0, 1]."""
    candidates = {}
    with open(PHASE2_CSV_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Assume Phase 2 score is out of 1.0 or 100. Normalize to 1.0.
            raw_score = float(row["score"])
            norm_score = raw_score if raw_score <= 1.0 else raw_score / 100.0
            candidates[row["candidate_id"]] = {"s_semantic": norm_score}
    return candidates

def compute_metadata_score(candidate: dict, jd_constraints: dict) -> float:
    """S_metadata: Calculates hard-skill fit based on YoE and Role Title."""
    profile = candidate.get("profile", {})
    cand_yoe = profile.get("years_of_experience", 0)
    cand_title = profile.get("current_title", "").lower()
    
    # 1. YoE Sub-score (Caps at 1.0 if they meet or exceed JD)
    req_yoe = jd_constraints["min_yoe"]
    yoe_score = 1.0 if cand_yoe >= req_yoe else (cand_yoe / req_yoe if req_yoe > 0 else 0)
    
    # 2. Title Sub-score (Are they actually a backend engineer?)
    # Extract core keywords from JD title (e.g., "Backend", "Engineer")
    target_keywords = [w for w in jd_constraints["target_title"].split() if len(w) > 3]
    title_score = 0.0
    for kw in target_keywords:
        if kw in cand_title:
            title_score += (1.0 / len(target_keywords))
            
    # Metadata is a 70/30 split between having the right experience and having the right title
    return (yoe_score * 0.7) + (title_score * 0.3)

def compute_behavioral_score(candidate: dict) -> float:
    """S_behavioral: Calculates platform activity and responsiveness."""
    signals = candidate.get("redrob_signals", {})
    # Default to 0.5 (neutral) if no data exists
    return float(signals.get("recruiter_response_rate", 0.5))

# ===================================================================
# Main Pipeline
# ===================================================================
def main():
    print("=" * 70)
    print("  PHASE 3: Hybrid Scoring (Semantic + Metadata + Behavioral)")
    print("=" * 70)

    jd_constraints = load_jd_constraints()
    print(f"🎯 Target Constraints: Min {jd_constraints['min_yoe']} YoE | Title: {jd_constraints['target_title']}")

    p2_candidates = fetch_phase2_results()
    target_ids = set(p2_candidates.keys())
    
    print("\n[1/3] Extracting top 100 full profiles from JSONL...")
    scored_candidates = []
    
    with open(CANDIDATES_JSONL_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            try:
                cand = json.loads(line)
                c_id = cand.get("candidate_id", cand.get("id"))
                if c_id in target_ids:
                    s_semantic = p2_candidates[c_id]["s_semantic"]
                    s_meta = compute_metadata_score(cand, jd_constraints)
                    s_behav = compute_behavioral_score(cand)
                    
                    # The Hybrid Formula
                    final_score = (W_SEMANTIC * s_semantic) + (W_METADATA * s_meta) + (W_BEHAVIORAL * s_behav)
                    
                    scored_candidates.append({
                        "candidate_id": c_id,
                        "title": cand.get("profile", {}).get("current_title", "Unknown"),
                        "yoe": cand.get("profile", {}).get("years_of_experience", 0),
                        "s_semantic": round(s_semantic, 3),
                        "s_metadata": round(s_meta, 3),
                        "s_behavioral": round(s_behav, 3),
                        "hybrid_score": round(final_score, 4)
                    })
            except Exception as e:
                continue

    print("\n[2/3] Ranking candidates by Hybrid Score...")
    # Sort by Hybrid Score (Descending)
    scored_candidates.sort(key=lambda x: x["hybrid_score"], reverse=True)
    
    # We only want to pass the absolute best 20 to the LLM
    top_20 = scored_candidates[:20]

    print(f"\n[3/3] Exporting Top 20 to {OUTPUT_CSV_PATH.name}...")
    with open(OUTPUT_CSV_PATH, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["hybrid_rank", "candidate_id", "title", "yoe", "s_semantic", "s_metadata", "s_behavioral", "hybrid_score"])
        for i, r in enumerate(top_20):
            writer.writerow([i+1, r["candidate_id"], r["title"], r["yoe"], r["s_semantic"], r["s_metadata"], r["s_behavioral"], r["hybrid_score"]])

    print("\n✅ Phase 3 Complete! The QA Engineers have been filtered out.")
    print("\n🏆 Top 3 Hybrid Candidates:")
    for i in range(min(3, len(top_20))):
        c = top_20[i]
        print(f"  {i+1}. {c['title']} ({c['yoe']} yrs) | Hybrid: {c['hybrid_score']:.3f} | Meta: {c['s_metadata']:.2f} | Sem: {c['s_semantic']:.2f}")

if __name__ == "__main__":
    main()