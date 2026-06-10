import os
import csv
import json
import sys
import time
import subprocess
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

PHASE2_CSV_PATH       = PROJECT_ROOT / "submission.csv"
CANDIDATES_JSONL_PATH = PROJECT_ROOT / "data" / "raw" / "candidates.jsonl"
PARSED_JD_PATH        = PROJECT_ROOT / "src" / "phase1" / "parsed_jd.json"
FINAL_REPORT_PATH     = PROJECT_ROOT / "final_ai_recruiter_report.csv"
VALIDATOR_SCRIPT      = PROJECT_ROOT / "data" / "raw" / "validate_submission.py"

TOP_N_TO_ANALYZE = 20

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))


# ===========================================================================
# Stage 1 — Load Phase 2 submission.csv in full
# ===========================================================================
def load_submission_csv(csv_path: Path) -> list[dict]:
    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


# ===========================================================================
# Stage 2 — Cache: load previously computed LLM scores (avoids re-burning
#            daily API token quota when the report already exists)
# ===========================================================================
def load_cached_evaluations(report_path: Path) -> dict:
    """
    Read final_ai_recruiter_report.csv written by any previous run.
    Returns {candidate_id: {llm_score, recruiter_notes, title}} for every
    row whose notes are NOT an error string.
    """
    cache: dict = {}
    if not report_path.exists():
        return cache
    try:
        with open(report_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                c_id  = row.get("candidate_id", "").strip()
                notes = row.get("recruiter_notes", "")
                # Skip rows that failed in a prior run
                if not c_id or notes.startswith("Evaluation failed"):
                    continue
                try:
                    cache[c_id] = {
                        "llm_score":      int(float(row.get("llm_score", 0))),
                        "recruiter_notes": notes,
                        "title":           row.get("title", ""),
                    }
                except (ValueError, TypeError):
                    pass
    except Exception as exc:
        print(f"  ⚠️  Could not read cache ({exc}). Will call API for all candidates.")
    return cache


# ===========================================================================
# Stage 3 — Fetch full profiles for candidates not already cached
# ===========================================================================
def fetch_full_profiles(jsonl_path: Path, target_ids: list[str]) -> dict:
    profiles: dict = {}
    target_set = set(target_ids)
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                cand  = json.loads(line)
                c_id  = cand.get("candidate_id", cand.get("id"))
                if c_id in target_set:
                    profiles[c_id] = {
                        "title":            cand.get("profile", {}).get("current_title", ""),
                        "experience_years": cand.get("profile", {}).get("years_of_experience", 0),
                        "skills":           [s.get("name") for s in cand.get("skills", [])],
                        "career_history":   cand.get("career_history", []),
                        "signals":          cand.get("redrob_signals", {}),
                    }
                    if len(profiles) == len(target_set):
                        break
            except Exception:
                continue
    return profiles


# ===========================================================================
# Stage 4 — LLM evaluation with smart retry / daily-limit detection
# ===========================================================================
def evaluate_candidate_with_llm(jd: dict, candidate: dict) -> dict:
    """
    Call Groq Llama-3.  Returns {"final_score": int, "recruiter_notes": str}.
    • Per-minute rate limit  → wait 30 s and retry (up to 3 times).
    • Daily token limit      → return score=0 immediately (retrying is useless).
    • Any other error        → return score=0 with note.
    """
    system_prompt = """
You are an elite, highly critical technical recruiter.
Evaluate the candidate against the job description.
Return ONLY a valid JSON object with exactly two keys:
1. "final_score": An integer from 0 to 100 representing true fit.
2. "recruiter_notes": A single, punchy sentence explaining exactly why they
   are or aren't a perfect fit.
"""
    user_prompt = f"""
JOB DESCRIPTION REQUIREMENTS:
{json.dumps(jd, indent=2)}

CANDIDATE PROFILE:
{json.dumps(candidate, indent=2)}
"""
    MAX_RETRIES = 3
    for attempt in range(MAX_RETRIES):
        try:
            resp = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_prompt},
                ],
                temperature=0.1,
                response_format={"type": "json_object"},
            )
            return json.loads(resp.choices[0].message.content)
        except Exception as exc:
            err = str(exc)
            # Daily token quota (TPD) — no point retrying in the same run
            if "tokens per day" in err or "TPD" in err:
                return {
                    "final_score":     0,
                    "recruiter_notes": "Evaluation skipped: daily API token limit reached.",
                }
            # Per-minute rate limit — wait and retry
            if "rate_limit_exceeded" in err and attempt < MAX_RETRIES - 1:
                wait = 30 * (attempt + 1)
                print(f"\n  ⏳ Per-minute rate limit — waiting {wait}s before retry {attempt+2}/{MAX_RETRIES}...")
                time.sleep(wait)
                continue
            # Anything else
            return {"final_score": 0, "recruiter_notes": f"Evaluation failed: {exc}"}
    return {"final_score": 0, "recruiter_notes": "Evaluation failed after max retries."}


# ===========================================================================
# Stage 5 — Two-band score assignment (guarantees validator monotonicity)
# ===========================================================================
def compute_banded_scores(
    reranked_top20: list[dict],
    remaining_80:   list[dict],
) -> tuple[list[dict], list[dict]]:
    """
    Assign final submission scores in two non-overlapping bands:
      • Top-20  → [0.5050, 1.0000]   (LLM scores drive relative ordering)
      • Tail-80 → [0.0050, 0.5000]   (Phase-2 scores drive relative ordering)

    This guarantees min(top-20 score) = 0.505 > max(tail-80 score) = 0.500,
    so the file is always monotonically non-increasing across all 100 rows.
    """
    # ── Top-20 band ─────────────────────────────────────────────────────────
    top_raw  = [r["llm_score"] for r in reranked_top20]
    top_max  = max(top_raw) if top_raw else 1
    top_min  = min(top_raw) if top_raw else 0
    top_span = (top_max - top_min) or 1

    for r in reranked_top20:
        norm = (r["llm_score"] - top_min) / top_span       # 0.0 → 1.0
        r["final_score"] = round(0.505 + norm * 0.495, 4)  # 0.505 → 1.000

    # ── Tail-80 band ─────────────────────────────────────────────────────────
    tail_raw  = [float(r["score"]) for r in remaining_80]
    tail_max  = max(tail_raw) if tail_raw else 1
    tail_min  = min(tail_raw) if tail_raw else 0
    tail_span = (tail_max - tail_min) or 1

    for r in remaining_80:
        norm = (float(r["score"]) - tail_min) / tail_span   # 0.0 → 1.0
        r["_final_score"] = round(0.005 + norm * 0.495, 4)  # 0.005 → 0.500

    return reranked_top20, remaining_80


# ===========================================================================
# Stage 6 — Safety net: forward-pass monotonicity enforcement
# ===========================================================================
def enforce_monotonicity(rows: list[dict]) -> list[dict]:
    """
    Walk ranks in order and clamp each score to be ≤ the previous score.
    This is a belt-and-suspenders guard; with proper banding it should never
    actually change anything.
    """
    prev = float(rows[0]["_out_score"])
    for row in rows[1:]:
        s = float(row["_out_score"])
        if s > prev:
            row["_out_score"] = prev
        else:
            prev = s
    return rows


# ===========================================================================
# Stage 7 — Save human-readable side report
# ===========================================================================
def save_final_report(results: list[dict], report_path: Path):
    with open(report_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["final_rank", "candidate_id", "title", "llm_score", "recruiter_notes"])
        for i, r in enumerate(results, start=1):
            writer.writerow([i, r["candidate_id"], r["title"],
                             r["llm_score"], r["recruiter_notes"]])


# ===========================================================================
# Stage 8 — Merge & overwrite submission.csv
# ===========================================================================
def merge_and_overwrite_submission(
    reranked_top20: list[dict],
    remaining_80:   list[dict],
    output_path:    Path,
):
    # Attach a unified _out_score field to every row before monotonicity pass
    all_rows: list[dict] = []
    for r in reranked_top20:
        all_rows.append({
            "candidate_id": r["candidate_id"],
            "reasoning":    r["recruiter_notes"],
            "_out_score":   r["final_score"],
        })
    for r in remaining_80:
        all_rows.append({
            "candidate_id": r["candidate_id"],
            "reasoning":    r["reasoning"],
            "_out_score":   r["_final_score"],
        })

    all_rows = enforce_monotonicity(all_rows)

    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        for rank, row in enumerate(all_rows, start=1):
            writer.writerow([
                row["candidate_id"],
                rank,
                f"{float(row['_out_score']):.4f}",
                row["reasoning"],
            ])

    print(f"\n  ✅ submission.csv overwritten — "
          f"{len(reranked_top20)} LLM-ranked + {len(remaining_80)} Phase-2 tail "
          f"= {len(all_rows)} total rows.")


# ===========================================================================
# Main Pipeline
# ===========================================================================
def main():
    print("=" * 70)
    print("  PHASE 4: Expert LLM Re-Ranker  →  Official submission.csv Merger")
    print("=" * 70)

    # ── 1. Load JD ──────────────────────────────────────────────────────────
    if not PARSED_JD_PATH.exists():
        print(f"  ERROR: {PARSED_JD_PATH} not found. Run Phase 1 first.")
        sys.exit(1)
    with open(PARSED_JD_PATH, "r", encoding="utf-8") as f:
        jd = json.load(f)

    # ── 2. Load Phase 2 submission.csv ──────────────────────────────────────
    if not PHASE2_CSV_PATH.exists():
        print(f"  ERROR: {PHASE2_CSV_PATH} not found. Run Phase 2 first.")
        sys.exit(1)

    print(f"\n[1/6] Loading Phase 2 submission.csv...")
    phase2_rows = load_submission_csv(PHASE2_CSV_PATH)
    print(f"       → {len(phase2_rows)} candidates loaded.")

    top_ids   = [row["candidate_id"] for row in phase2_rows[:TOP_N_TO_ANALYZE]]
    top_id_set = set(top_ids)

    # ── 3. Load evaluation cache (previous runs) ─────────────────────────────
    print(f"\n[2/6] Checking evaluation cache ({FINAL_REPORT_PATH.name})...")
    cache = load_cached_evaluations(FINAL_REPORT_PATH)
    cached_ids   = [cid for cid in top_ids if cid in cache]
    uncached_ids = [cid for cid in top_ids if cid not in cache]
    print(f"       → {len(cached_ids)}/{TOP_N_TO_ANALYZE} evaluations found in cache.")
    if uncached_ids:
        print(f"       → {len(uncached_ids)} candidates need fresh API calls.")

    # ── 4. Fetch profiles only for uncached candidates ───────────────────────
    raw_results: list[dict] = []

    # Pre-populate from cache
    for c_id in cached_ids:
        c = cache[c_id]
        raw_results.append({
            "candidate_id":    c_id,
            "title":           c["title"],
            "llm_score":       c["llm_score"],
            "recruiter_notes": c["recruiter_notes"],
        })

    if uncached_ids:
        print(f"\n[3/6] Fetching full profiles for {len(uncached_ids)} uncached candidates...")
        profiles = fetch_full_profiles(CANDIDATES_JSONL_PATH, uncached_ids)
        print(f"       → {len(profiles)} profiles retrieved.")

        print(f"\n[4/6] Llama-3.3-70B evaluating {len(uncached_ids)} uncached candidates...")
        for c_id in tqdm(uncached_ids, desc="  Evaluating"):
            candidate_data = profiles.get(c_id, {})
            evaluation = evaluate_candidate_with_llm(jd, candidate_data)
            raw_results.append({
                "candidate_id":    c_id,
                "title":           candidate_data.get("title", "Unknown"),
                "llm_score":       evaluation.get("final_score", 0),
                "recruiter_notes": evaluation.get("recruiter_notes", "Evaluation error."),
            })
            time.sleep(0.5)
    else:
        print(f"\n[3/6] Skipping profile fetch — all candidates served from cache.")
        print(f"\n[4/6] Skipping API calls — all evaluations loaded from cache.")

    # ── 5. Sort Top-20 by LLM score (desc), ties broken by candidate_id ─────
    raw_results.sort(key=lambda x: (-x["llm_score"], x["candidate_id"]))

    # ── 6. Build tail-80 (Phase-2 rows not in top-20) ───────────────────────
    remaining_80 = [
        row for row in phase2_rows
        if row["candidate_id"] not in top_id_set
    ]
    remaining_80.sort(key=lambda r: int(r["rank"]))

    # ── 7. Compute two-band scores ───────────────────────────────────────────
    print(f"\n[5/6] Computing banded scores (top-20 → [0.505–1.000], tail-80 → [0.005–0.500])...")
    reranked_top20, remaining_80 = compute_banded_scores(raw_results, remaining_80)

    # ── 8. Save side report then overwrite submission.csv ────────────────────
    save_final_report(reranked_top20, FINAL_REPORT_PATH)
    print(f"       → Human-readable report saved to {FINAL_REPORT_PATH.name}")

    print(f"\n[6/6] Merging and overwriting submission.csv...")
    merge_and_overwrite_submission(reranked_top20, remaining_80, PHASE2_CSV_PATH)

    # ── Leaderboard preview ───────────────────────────────────────────────────
    print("\n🏆 Top 5 Candidates (LLM Re-Ranked):")
    for i, r in enumerate(reranked_top20[:5], start=1):
        print(f"   #{i}  {r['candidate_id']}  [{r['title']}]  "
              f"LLM={r['llm_score']}/100  FinalScore={r['final_score']:.4f}")
        print(f"       ↳ {r['recruiter_notes']}")

    # ── Boundary check ────────────────────────────────────────────────────────
    min_top  = min(r["final_score"]   for r in reranked_top20)
    max_tail = max(float(r["_final_score"]) for r in remaining_80) if remaining_80 else 0
    print(f"\n  Score boundary: min(top-20)={min_top:.4f}  max(tail-80)={max_tail:.4f}  "
          f"→ {'✅ clean gap' if min_top > max_tail else '⚠️  overlap!'}")

    # ── 9. Auto-validate ──────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("  Running official hackathon validator on submission.csv...")
    print(f"{'='*70}")

    if VALIDATOR_SCRIPT.exists():
        result = subprocess.run(
            [sys.executable, str(VALIDATOR_SCRIPT), str(PHASE2_CSV_PATH)],
            capture_output=True, text=True,
        )
        if result.stdout.strip():
            print(result.stdout.strip())
        if result.returncode == 0:
            print("\n  ✅ Submission is VALID and ready for hackathon submission!")
        else:
            print(f"\n  ❌ Validation FAILED:\n{result.stderr.strip()}")
    else:
        print(f"  ⚠️  Validator not found at {VALIDATOR_SCRIPT}. Skipping.")


if __name__ == "__main__":
    main()