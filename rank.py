"""
rank.py — Offline Ranking Script (Submission-Ready)
=====================================================
Produces submission.csv from candidates.jsonl with NO external API calls.
Designed to run within 5 minutes on CPU with 16GB RAM.

Architecture:
  1. Load parsed JD signals (pre-computed by Phase 1, cached in parsed_jd.json)
  2. Stream candidates.jsonl and compute a fast hybrid score for each
  3. Output top-100 ranked candidates to submission.csv

Pre-computation (run once):
  python run_all.py          # Full pipeline (Phases 1-4, uses API for dev/research)
  python precompute.py       # Only precomputes embeddings if needed

Offline ranking (no network):
  python rank.py --candidates ./data/raw/candidates.jsonl --out ./submission.csv

This script intentionally avoids:
  - Any LLM API calls
  - Any network requests
  - GPU usage

Usage:
  python rank.py --candidates path/to/candidates.jsonl --out path/to/submission.csv
  python rank.py  # uses defaults: data/raw/candidates.jsonl → submission.csv
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths & Configuration
# ---------------------------------------------------------------------------
PROJECT_ROOT   = Path(__file__).resolve().parent
PARSED_JD_PATH = PROJECT_ROOT / "src" / "phase1" / "parsed_jd.json"

# Scoring weights (must sum to 1.0)
W_METADATA   = 0.55   # YoE + title + skill overlap — primary offline signal
W_BEHAVIORAL = 0.20   # Recruiter response rate (behavioral signal)
W_PLATFORM   = 0.25   # Platform activity signals from Redrob

TOP_K = 100


# ---------------------------------------------------------------------------
# JD Loading
# ---------------------------------------------------------------------------
def load_jd(jd_path: Path) -> dict:
    """Load the pre-computed JD structure from Phase 1."""
    if not jd_path.exists():
        print(f"ERROR: {jd_path} not found.")
        print("Run 'python src/phase1/jd_parser.py' first (requires GROQ_API_KEY).")
        sys.exit(1)
    with open(jd_path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_jd_constraints(jd: dict) -> dict:
    """Build normalized constraint lookup from parsed JD."""
    must_haves  = set(s.lower() for s in jd.get("must_have_technical_skills", []))
    nice_haves  = set(s.lower() for s in jd.get("nice_to_have_technical_skills", []))
    title_words = [w for w in jd.get("role_title", "").lower().split() if len(w) > 3]
    return {
        "min_yoe":       jd.get("minimum_years_experience") or 0,
        "target_title":  jd.get("role_title", "").lower(),
        "title_words":   title_words,
        "must_haves":    must_haves,
        "nice_haves":    nice_haves,
        "all_skills":    must_haves | nice_haves,
        "responsibilities": " ".join(jd.get("core_responsibilities", [])).lower(),
    }


# ---------------------------------------------------------------------------
# Honeypot Detection
# ---------------------------------------------------------------------------
def is_honeypot(candidate: dict) -> bool:
    """
    Detect subtly impossible profiles.
    Returns True if the candidate shows honeypot signals.
    """
    profile = candidate.get("profile", {})
    yoe     = profile.get("years_of_experience", 0)
    skills  = candidate.get("skills", [])

    # Signal 1: Expert in many skills but 0 years each (expertise without time)
    expert_zero_yoe = sum(
        1 for s in skills
        if s.get("proficiency", "").lower() == "expert"
        and s.get("duration_months", 1) == 0
    )
    if expert_zero_yoe >= 5:
        return True

    # Signal 2: Company tenure exceeds company age (impossible timeline)
    career = candidate.get("career_history", [])
    for job in career:
        company_founded = job.get("company_founded_year")
        duration_months = job.get("duration_months", 0)
        if company_founded and duration_months:
            years_at_company = duration_months / 12
            company_age = 2025 - int(company_founded)
            if years_at_company > company_age + 1:   # +1 for rounding
                return True

    # Signal 3: Claims 10+ skills at expert level in < 2 years total experience
    if yoe < 2:
        expert_skills = sum(
            1 for s in skills if s.get("proficiency", "").lower() == "expert"
        )
        if expert_skills >= 8:
            return True

    return False


# ---------------------------------------------------------------------------
# Skill Overlap Scoring
# ---------------------------------------------------------------------------
def compute_skill_overlap(candidate: dict, constraints: dict) -> tuple[float, list[str]]:
    """
    Returns (score [0,1], list_of_matched_must_have_skills).
    Must-have matches weighted 2x over nice-to-have.
    """
    cand_skills_raw = [s.get("name", "") for s in candidate.get("skills", [])]
    cand_skills = set(s.lower() for s in cand_skills_raw)

    must_hits = []
    for jd_skill in constraints["must_haves"]:
        # Partial bidirectional matching
        if any(jd_skill in cs or cs in jd_skill for cs in cand_skills):
            must_hits.append(jd_skill)

    nice_hits = sum(
        1 for jd_skill in constraints["nice_haves"]
        if any(jd_skill in cs or cs in jd_skill for cs in cand_skills)
    )

    total_possible = len(constraints["must_haves"]) * 2 + len(constraints["nice_haves"])
    if total_possible == 0:
        return 0.5, []

    weighted = len(must_hits) * 2 + nice_hits
    return min(weighted / total_possible, 1.0), must_hits


# ---------------------------------------------------------------------------
# Main Scoring Function
# ---------------------------------------------------------------------------
def score_candidate(candidate: dict, constraints: dict) -> tuple[float, str]:
    """
    Compute a multi-signal score for a single candidate.
    Returns (score [0,1], reasoning_string).
    """
    profile       = candidate.get("profile", {})
    yoe           = profile.get("years_of_experience", 0)
    cand_title    = profile.get("current_title", "").lower()
    signals       = candidate.get("redrob_signals", {})
    skills_list   = candidate.get("skills", [])

    # ── 1. YoE score (hard constraint) ───────────────────────────────────────
    req_yoe  = constraints["min_yoe"]
    yoe_score = 1.0 if yoe >= req_yoe else (yoe / req_yoe if req_yoe > 0 else 0.5)

    # ── 2. Title relevance ────────────────────────────────────────────────────
    title_score = 0.0
    kws = constraints["title_words"]
    if kws:
        title_score = sum(1.0 for kw in kws if kw in cand_title) / len(kws)
    # Bonus: exact domain match (AI, ML, data, backend, engineer)
    ai_titles   = ["ai", "ml", "machine learning", "data science", "nlp", "deep learning",
                   "search", "ranking", "recommendation", "retrieval"]
    domain_bonus = 0.3 if any(t in cand_title for t in ai_titles) else 0.0

    # ── 3. Skill overlap ──────────────────────────────────────────────────────
    skill_score, matched_must = compute_skill_overlap(candidate, constraints)

    # ── 4. Metadata composite ─────────────────────────────────────────────────
    # YoE 50%, title 25%, skills 25%
    meta_score = (yoe_score * 0.50
                + min(title_score + domain_bonus, 1.0) * 0.25
                + skill_score * 0.25)

    # ── 5. Behavioral signals ─────────────────────────────────────────────────
    rr           = float(signals.get("recruiter_response_rate", 0.5))
    login_days   = int(signals.get("days_since_last_login", 999))
    active_bonus = 0.1 if login_days <= 30 else (0.05 if login_days <= 90 else 0.0)
    behav_score  = min(rr + active_bonus, 1.0)

    # ── 6. Platform activity ─────────────────────────────────────────────────
    applications    = int(signals.get("total_applications", 0))
    profile_views   = int(signals.get("profile_views_last_30d", 0))
    skill_tests     = float(signals.get("skill_tests_completed", 0))
    platform_score  = min(
        (applications / 50.0) * 0.4
        + (profile_views / 100.0) * 0.3
        + (skill_tests / 5.0) * 0.3,
        1.0,
    )

    # ── 7. Honeypot penalty ──────────────────────────────────────────────────
    honeypot = is_honeypot(candidate)
    hp_penalty = 0.40 if honeypot else 0.0

    # ── Final weighted score ──────────────────────────────────────────────────
    raw = (W_METADATA   * meta_score
         + W_BEHAVIORAL * behav_score
         + W_PLATFORM   * platform_score
         - hp_penalty)
    final = max(0.0, min(1.0, raw))

    # ── Reasoning string ──────────────────────────────────────────────────────
    display_title = profile.get("current_title", "Unknown")
    yoe_note      = "meets YoE req" if yoe >= req_yoe else f"below {req_yoe}yr req"
    skills_str    = (f"matched: {', '.join(matched_must[:3])}"
                     if matched_must else "no direct must-have skill match")
    hp_note       = " [HONEYPOT DETECTED — penalized]" if honeypot else ""
    reasoning = (
        f"{display_title} | {yoe}yrs XP ({yoe_note}) | "
        f"{len(matched_must)} must-have skills {skills_str} | "
        f"response rate {rr:.0%} | "
        f"score {final:.3f}{hp_note}"
    )

    return final, reasoning


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------
def main(candidates_path: Path, output_path: Path):
    t0 = time.time()
    print("=" * 70)
    print("  TalentMatch-AI — Offline Ranking (rank.py)")
    print("  No API calls. No network. No GPU.")
    print("=" * 70)

    jd          = load_jd(PARSED_JD_PATH)
    constraints = build_jd_constraints(jd)
    print(f"\n  JD: '{jd.get('role_title', '')}' | Min YoE: {constraints['min_yoe']} "
          f"| Must-have skills: {len(constraints['must_haves'])}")

    if not candidates_path.exists():
        print(f"\n  ERROR: {candidates_path} not found.")
        sys.exit(1)

    print(f"\n  Streaming {candidates_path.name}...")
    all_scored: list[dict] = []
    n_read = 0
    n_honeypots = 0

    with open(candidates_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                cand = json.loads(line)
                c_id = cand.get("candidate_id", cand.get("id", "UNKNOWN"))
                score, reasoning = score_candidate(cand, constraints)

                if is_honeypot(cand):
                    n_honeypots += 1

                all_scored.append({
                    "candidate_id": c_id,
                    "score":        score,
                    "reasoning":    reasoning,
                })
                n_read += 1

                if n_read % 10000 == 0:
                    elapsed = time.time() - t0
                    print(f"  Processed {n_read:,} candidates ({elapsed:.1f}s)...")

            except Exception:
                continue

    # Sort by score descending, break ties by candidate_id ascending
    all_scored.sort(key=lambda x: (-x["score"], x["candidate_id"]))
    top_100 = all_scored[:TOP_K]

    # Write output
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        for rank, row in enumerate(top_100, start=1):
            writer.writerow([
                row["candidate_id"],
                rank,
                f"{row['score']:.4f}",
                row["reasoning"],
            ])

    elapsed = time.time() - t0
    print(f"\n  ✅ Done in {elapsed:.1f}s")
    print(f"  📊 {n_read:,} candidates scored | {n_honeypots} honeypots detected")
    print(f"  📄 Top 100 written to: {output_path}")
    print(f"\n  Top 5:")
    for i, r in enumerate(top_100[:5]):
        print(f"    #{i+1}  {r['candidate_id']}  score={r['score']:.4f}")
        print(f"         {r['reasoning'][:100]}...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="TalentMatch-AI offline ranker — no API calls required"
    )
    parser.add_argument(
        "--candidates",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw" / "candidates.jsonl",
        help="Path to candidates.jsonl",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=PROJECT_ROOT / "submission.csv",
        help="Output path for submission.csv",
    )
    args = parser.parse_args()
    main(args.candidates, args.out)
