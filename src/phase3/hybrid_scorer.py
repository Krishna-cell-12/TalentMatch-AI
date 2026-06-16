"""
Phase 3: Hybrid Scoring Engine
================================
Combines three independent signals into a single weighted score:
  S_final = 0.45 * S_semantic + 0.40 * S_metadata + 0.15 * S_behavioral

S_semantic  : FAISS cosine similarity score from Phase 2 (meaning-level fit)
S_metadata  : YoE compliance + title relevance + skill overlap
S_behavioral: Platform engagement signals (recruiter response rate)

Outputs:
  hybrid_shortlist.csv — Top 20 candidates for LLM deep evaluation
"""
import json
import csv
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration & Weights
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PHASE2_CSV_PATH       = PROJECT_ROOT / "submission.csv"
CANDIDATES_JSONL_PATH = PROJECT_ROOT / "data" / "raw" / "candidates.jsonl"
PARSED_JD_PATH        = PROJECT_ROOT / "src" / "phase1" / "parsed_jd.json"
OUTPUT_CSV_PATH       = PROJECT_ROOT / "hybrid_shortlist.csv"

# Hybrid Formula Weights (must sum to 1.0)
W_SEMANTIC   = 0.45
W_METADATA   = 0.40
W_BEHAVIORAL = 0.15


# ===================================================================
# Helper Functions
# ===================================================================
def load_jd() -> dict:
    """Load the full parsed JD."""
    with open(PARSED_JD_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def load_jd_constraints(jd: dict) -> dict:
    """Extract hard constraints and skill lists from the parsed JD."""
    must_haves = set(s.lower() for s in jd.get("must_have_technical_skills", []))
    nice_haves = set(s.lower() for s in jd.get("nice_to_have_technical_skills", []))
    return {
        "min_yoe":       jd.get("minimum_years_experience") or 0,
        "target_title":  jd.get("role_title", "").lower(),
        "must_haves":    must_haves,
        "nice_haves":    nice_haves,
        "all_jd_skills": must_haves | nice_haves,
    }


def fetch_phase2_results() -> dict:
    """Load Phase 2 results with normalized semantic scores [0, 1]."""
    candidates = {}
    with open(PHASE2_CSV_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw_score = float(row["score"])
            norm_score = raw_score if raw_score <= 1.0 else raw_score / 100.0
            candidates[row["candidate_id"]] = {
                "s_semantic": norm_score,
                "phase2_rank": int(row["rank"]),
            }
    return candidates


def compute_skill_overlap(candidate: dict, jd_constraints: dict) -> float:
    """
    Compute how many JD skills the candidate has.
    Must-have matches are weighted 2x over nice-to-have matches.
    Returns a normalized score in [0, 1].
    """
    cand_skills = set(
        s.get("name", "").lower()
        for s in candidate.get("skills", [])
    )
    # Partial matching: check if any JD skill keyword appears in any candidate skill
    must_have_hits = sum(
        1 for jd_skill in jd_constraints["must_haves"]
        if any(jd_skill in cs or cs in jd_skill for cs in cand_skills)
    )
    nice_have_hits = sum(
        1 for jd_skill in jd_constraints["nice_haves"]
        if any(jd_skill in cs or cs in jd_skill for cs in cand_skills)
    )
    total_jd_skills = len(jd_constraints["must_haves"]) * 2 + len(jd_constraints["nice_haves"])
    if total_jd_skills == 0:
        return 0.5
    weighted_hits = (must_have_hits * 2 + nice_have_hits)
    return min(weighted_hits / total_jd_skills, 1.0)


def compute_metadata_score(candidate: dict, jd_constraints: dict) -> float:
    """
    S_metadata: Calculates hard-skill fit (60% YoE + 20% title + 20% skill overlap).
    """
    profile   = candidate.get("profile", {})
    cand_yoe  = profile.get("years_of_experience", 0)
    cand_title = profile.get("current_title", "").lower()

    # 1. YoE sub-score — caps at 1.0 when candidate meets or exceeds requirement
    req_yoe = jd_constraints["min_yoe"]
    yoe_score = 1.0 if cand_yoe >= req_yoe else (cand_yoe / req_yoe if req_yoe > 0 else 0.5)

    # 2. Title sub-score — how many core JD title keywords appear in candidate title?
    target_keywords = [w for w in jd_constraints["target_title"].split() if len(w) > 3]
    if target_keywords:
        title_score = sum(1.0 for kw in target_keywords if kw in cand_title) / len(target_keywords)
    else:
        title_score = 0.5  # neutral if no keywords to match

    # 3. Skill overlap sub-score
    skill_score = compute_skill_overlap(candidate, jd_constraints)

    # Weighted combination: YoE is the strongest signal
    return (yoe_score * 0.60) + (title_score * 0.20) + (skill_score * 0.20)


def compute_behavioral_score(candidate: dict) -> float:
    """
    S_behavioral: Platform activity and recruiter responsiveness.
    Uses recruiter_response_rate as primary signal; defaults to 0.5 (neutral).
    """
    signals = candidate.get("redrob_signals", {})
    response_rate = float(signals.get("recruiter_response_rate", 0.5))
    # Clamp to [0, 1]
    return max(0.0, min(1.0, response_rate))


def generate_hybrid_reasoning(candidate: dict, jd_constraints: dict,
                               s_semantic: float, s_meta: float, s_behav: float,
                               final_score: float) -> str:
    """Generate a rich, judge-readable reasoning string for the submission CSV."""
    profile = candidate.get("profile", {})
    title   = profile.get("current_title", "Unknown")
    yoe     = profile.get("years_of_experience", 0)
    signals = candidate.get("redrob_signals", {})
    rr      = signals.get("recruiter_response_rate", 0.0)

    # Skill overlap
    cand_skills = set(s.get("name", "").lower() for s in candidate.get("skills", []))
    matched_must = [
        jd_skill for jd_skill in jd_constraints["must_haves"]
        if any(jd_skill in cs or cs in jd_skill for cs in cand_skills)
    ]
    matched_count = len(matched_must)

    yoe_status = "meets" if yoe >= jd_constraints["min_yoe"] else "below"
    skills_str  = f"{matched_count} of {len(jd_constraints['must_haves'])} must-have skills"
    return (
        f"{title} | {yoe} yrs XP ({yoe_status} requirement) | "
        f"{skills_str} matched | response rate {rr:.0%} | "
        f"hybrid score {final_score:.3f} "
        f"[sem={s_semantic:.2f} meta={s_meta:.2f} beh={s_behav:.2f}]"
    )


# ===================================================================
# Main Pipeline
# ===================================================================
def main():
    start = time.time()
    print("=" * 70)
    print("  PHASE 3: Hybrid Scoring (Semantic + Metadata + Behavioral)")
    print("=" * 70)

    jd = load_jd()
    jd_constraints = load_jd_constraints(jd)
    print(f"\n🎯 JD: '{jd.get('role_title', '')}' | "
          f"Min YoE: {jd_constraints['min_yoe']} | "
          f"Must-have skills: {len(jd_constraints['must_haves'])}")

    p2_candidates = fetch_phase2_results()
    target_ids    = set(p2_candidates.keys())
    print(f"   Loaded {len(target_ids)} Phase 2 candidates.")

    print("\n[1/3] Streaming JSONL and computing hybrid scores...")
    scored_candidates = []

    with open(CANDIDATES_JSONL_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                cand = json.loads(line)
                c_id = cand.get("candidate_id", cand.get("id"))
                if c_id not in target_ids:
                    continue

                s_semantic = p2_candidates[c_id]["s_semantic"]
                s_meta     = compute_metadata_score(cand, jd_constraints)
                s_behav    = compute_behavioral_score(cand)

                # Weighted Hybrid Formula
                final_score = (W_SEMANTIC   * s_semantic
                             + W_METADATA   * s_meta
                             + W_BEHAVIORAL * s_behav)

                reasoning = generate_hybrid_reasoning(
                    cand, jd_constraints,
                    s_semantic, s_meta, s_behav, final_score
                )

                scored_candidates.append({
                    "candidate_id":  c_id,
                    "title":         cand.get("profile", {}).get("current_title", "Unknown"),
                    "yoe":           cand.get("profile", {}).get("years_of_experience", 0),
                    "s_semantic":    round(s_semantic, 4),
                    "s_metadata":    round(s_meta, 4),
                    "s_behavioral":  round(s_behav, 4),
                    "hybrid_score":  round(final_score, 4),
                    "reasoning":     reasoning,
                })
            except Exception:
                continue

    print("\n[2/3] Ranking candidates by Hybrid Score...")
    scored_candidates.sort(key=lambda x: x["hybrid_score"], reverse=True)
    top_20 = scored_candidates[:TOP_N_TO_ANALYZE]

    print(f"\n[3/3] Exporting Top {len(top_20)} to {OUTPUT_CSV_PATH.name}...")
    with open(OUTPUT_CSV_PATH, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "hybrid_rank", "candidate_id", "title", "yoe",
            "s_semantic", "s_metadata", "s_behavioral", "hybrid_score", "reasoning"
        ])
        for i, r in enumerate(top_20):
            writer.writerow([
                i + 1, r["candidate_id"], r["title"], r["yoe"],
                r["s_semantic"], r["s_metadata"], r["s_behavioral"],
                r["hybrid_score"], r["reasoning"],
            ])

    elapsed = time.time() - start
    print(f"\n✅ Phase 3 complete in {elapsed:.1f}s — "
          f"{len(scored_candidates)} candidates scored, top {len(top_20)} selected.\n")

    print("🏆 Top 5 Hybrid Candidates:")
    for i, c in enumerate(top_20[:5]):
        print(f"  #{i+1}  {c['candidate_id']}  [{c['title']}]  "
              f"YoE={c['yoe']}  Hybrid={c['hybrid_score']:.4f}  "
              f"[S={c['s_semantic']:.2f} M={c['s_metadata']:.2f} B={c['s_behavioral']:.2f}]")


TOP_N_TO_ANALYZE = 20

if __name__ == "__main__":
    main()