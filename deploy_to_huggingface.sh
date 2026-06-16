#!/bin/bash
# deploy_to_huggingface.sh
# Deploys the Gradio app to HuggingFace Spaces
# Run: bash deploy_to_huggingface.sh

set -e

HF_SPACE="https://huggingface.co/spaces/krishna12sk/TalentMatch-AI"
HF_REPO="krishna12sk/TalentMatch-AI"
CLONE_DIR="/tmp/hf_space_deploy"

echo "==================================================================="
echo "  TalentMatch-AI — HuggingFace Space Deployment"
echo "==================================================================="
echo ""
echo "  Space URL: $HF_SPACE"
echo ""
echo "  NOTE: You will be prompted for HuggingFace credentials."
echo "  Use a HF token with WRITE permissions as the password."
echo "  Get one at: https://huggingface.co/settings/tokens"
echo ""

# Clone the HuggingFace Space
echo "[1/4] Cloning HuggingFace Space..."
rm -rf "$CLONE_DIR"
git clone "https://huggingface.co/spaces/$HF_REPO" "$CLONE_DIR"

echo "[2/4] Copying app files..."

# Copy app.py
cp app.py "$CLONE_DIR/app.py"

# Copy requirements (gradio only for Space)
echo "gradio>=4.0.0" > "$CLONE_DIR/requirements.txt"

# Copy the Space README (with HF metadata frontmatter)
cp SPACE_README.md "$CLONE_DIR/README.md"

# Copy pre-computed JD
mkdir -p "$CLONE_DIR/src/phase1"
cp src/phase1/parsed_jd.json "$CLONE_DIR/src/phase1/parsed_jd.json"

# Copy sample candidates (used for live demo)
mkdir -p "$CLONE_DIR/data/raw"
cp data/raw/sample_candidates.json "$CLONE_DIR/data/raw/sample_candidates.json"

echo "[3/4] Committing and pushing to HuggingFace..."
cd "$CLONE_DIR"
git add -A
git commit -m "Deploy TalentMatch-AI Gradio app

- 4-tab interface: Live Demo, Custom Ranking, Architecture, About
- Auto-runs ranking on load using hackathon JD + sample candidates
- Offline scoring: no API keys needed
- Honeypot detection
- Full system architecture explanation

India Runs Hackathon x Redrob AI x Hack2Skill"
git push

echo ""
echo "[4/4] Done!"
echo ""
echo "  ✅ Your Space should be live in ~2 minutes at:"
echo "  $HF_SPACE"
echo ""
echo "  Note: First build takes a moment as HF installs gradio."
echo "==================================================================="
