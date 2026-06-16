---
title: TalentMatch-AI
emoji: 🚀
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: true
license: mit
short_description: Intelligent Candidate Discovery & Ranking — India Runs Hackathon
---

# TalentMatch-AI 🚀

**Intelligent Candidate Discovery & Ranking Engine**  
*India Runs Hackathon × Redrob AI × Hack2Skill — Track 1: AI & Datathon Arena*

## What This Does

Ranks 100,000 candidate profiles against a Job Description using:
- 🧲 **Semantic Vector Search** (FAISS + sentence-transformers)
- ⚖️ **Hybrid Scoring** (YoE + skills + behavioral + platform signals)
- 🧠 **LLM Re-Ranking** (Groq Llama-3.3-70B)
- 🎯 **Honeypot Detection** (penalizes impossible profiles)

## Quick Start

The demo auto-runs on load using the hackathon JD and sample candidates.

- **Tab 1:** Live demo with the real hackathon JD (Senior AI Engineer at Redrob AI)
- **Tab 2:** Paste your own JD + candidates and rank them instantly
- **Tab 3:** Full system architecture diagram
- **Tab 4:** About the project

## Run Locally

```bash
git clone https://github.com/Krishna-cell-12/TalentMatch-AI
cd TalentMatch-AI
pip install -r requirements.txt
python rank.py --candidates ./data/raw/candidates.jsonl --out ./submission.csv
```

## Links

- [GitHub Repository](https://github.com/Krishna-cell-12/TalentMatch-AI)
- [Hackathon: India Runs by Redrob AI](https://hack2skill.com/event/india_runs)
