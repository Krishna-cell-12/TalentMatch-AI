"""
run_all.py — One-Shot TalentMatch-AI Pipeline Runner
======================================================
Executes all 4 phases in sequence and reports timing + status.

Usage:
    python run_all.py

Phases:
    1. JD Parser        — Groq LLM extracts structured JD signals
    2. Vector Search    — FAISS semantic search over 100K candidates
    3. Hybrid Scorer    — Multi-signal weighted scoring (Top 20)
    4. LLM Re-Ranker    — Groq LLM deep evaluation + submission merge
"""
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

PHASES = [
    {
        "name":   "Phase 1: JD Parser (LLM Extraction)",
        "script": PROJECT_ROOT / "src" / "phase1" / "jd_parser.py",
        "desc":   "Parsing Job Description with Groq Llama-3.3-70B...",
    },
    {
        "name":   "Phase 2: Semantic Vector Search (FAISS)",
        "script": PROJECT_ROOT / "src" / "phase2" / "real_vector_search.py",
        "desc":   "Embedding 100K candidates & running semantic search...",
    },
    {
        "name":   "Phase 3: Hybrid Scoring Engine",
        "script": PROJECT_ROOT / "src" / "phase3" / "hybrid_scorer.py",
        "desc":   "Applying multi-signal hybrid scoring to select Top 20...",
    },
    {
        "name":   "Phase 4: LLM Re-Ranker & Submission Merger",
        "script": PROJECT_ROOT / "src" / "phase4" / "llm_reranker.py",
        "desc":   "Deep LLM evaluation of Top 20 + merging final submission.csv...",
    },
]


def run_phase(phase: dict, phase_num: int, total: int) -> bool:
    """Run a single pipeline phase. Returns True if successful."""
    bar    = "─" * 68
    header = f"  [{phase_num}/{total}] {phase['name']}"
    print(f"\n{'═' * 70}")
    print(header)
    print(f"  {phase['desc']}")
    print(f"{'─' * 70}")

    if not phase["script"].exists():
        print(f"  ❌ ERROR: Script not found at {phase['script']}")
        return False

    t0 = time.time()
    result = subprocess.run(
        [sys.executable, str(phase["script"])],
        cwd=str(PROJECT_ROOT),
    )
    elapsed = time.time() - t0

    if result.returncode == 0:
        print(f"\n  ✅ {phase['name']} completed in {elapsed:.1f}s")
        return True
    else:
        print(f"\n  ❌ {phase['name']} FAILED (exit code {result.returncode})")
        return False


def main():
    print("=" * 70)
    print("  🚀 TalentMatch-AI — Full Pipeline Runner")
    print("     Intelligent Candidate Discovery & Ranking Engine")
    print("=" * 70)
    print(f"\n  Running {len(PHASES)} phases. This will take several minutes.")
    print("  Phase 2 (vector search) takes the longest — please be patient.\n")

    pipeline_start = time.time()
    results = []

    for i, phase in enumerate(PHASES, start=1):
        ok = run_phase(phase, i, len(PHASES))
        results.append((phase["name"], ok))
        if not ok:
            print(f"\n  ⚠️  Pipeline stopped at Phase {i}. Fix the error above and re-run.")
            break

    # ── Summary ────────────────────────────────────────────────────────────
    total_elapsed = time.time() - pipeline_start
    print(f"\n{'═' * 70}")
    print("  PIPELINE SUMMARY")
    print(f"{'─' * 70}")
    all_ok = True
    for name, ok in results:
        status = "✅ PASS" if ok else "❌ FAIL"
        print(f"  {status}  {name}")
        if not ok:
            all_ok = False

    print(f"{'─' * 70}")
    print(f"  Total runtime: {total_elapsed:.1f}s")

    if all_ok:
        print("\n  🏆 All phases complete!")
        print("  📄 Output files:")
        outputs = [
            ("submission.csv",              "Official ranked output (submit this)"),
            ("final_ai_recruiter_report.csv","Human-readable LLM recruiter notes"),
            ("hybrid_shortlist.csv",         "Phase 3 top-20 hybrid scores"),
            ("src/phase1/parsed_jd.json",    "Structured JD extracted by LLM"),
        ]
        for fname, desc in outputs:
            fpath = PROJECT_ROOT / fname
            size  = fpath.stat().st_size if fpath.exists() else 0
            exists = "✅" if fpath.exists() else "❌ missing"
            print(f"    {exists}  {fname:40s}  {desc}  ({size:,} bytes)")
    print(f"{'═' * 70}\n")

    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
