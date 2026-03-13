#!/bin/bash
#
# Fetch an arXiv paper: PDF, Markdown, Kimi summary, and BibTeX.
#
# Usage: bash fetch_paper.sh <arxiv_id>
# Example: bash fetch_paper.sh 2501.11120v1
#

ARXIV_ID=$1

if [ -z "$ARXIV_ID" ]; then
  echo "Usage: bash fetch_paper.sh <arxiv_id>" >&2
  exit 1
fi

# Run from project root (parent of scripts/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT" || exit 1

# Load environment variables
if [ -f ".env" ]; then
  source ".env"
fi
PAPER_ROOT="${PAPER_ROOT:-storage/papers}"

# Activate Python venv if exists
if [ -f ".venv/bin/activate" ]; then
  source ".venv/bin/activate"
fi

mkdir -p "$PAPER_ROOT/$ARXIV_ID"

# Fetch paper to pdf
echo ">>> Fetching paper PDF for $ARXIV_ID..."
wget -O "$PAPER_ROOT/$ARXIV_ID/paper.pdf" "https://arxiv.org/pdf/$ARXIV_ID"

# Fetch paper to markdown
echo ">>> Converting paper to markdown..."
arxiv2md "$ARXIV_ID" -o "$PAPER_ROOT/$ARXIV_ID/paper.md"

# Fetch paper to kimi summary
echo -e "\n>>> Generating Kimi summary..."
python "skills/kimi_review/kimi_review.py" "$ARXIV_ID" -o "$PAPER_ROOT/$ARXIV_ID/kimi_review.md"

# Fetch bibtex
echo -e "\n>>> Fetching bibtex..."
bash "skills/arxiv2bibtex/arxiv2bibtex.sh" "$ARXIV_ID" -o "$PAPER_ROOT/$ARXIV_ID/paper.bib"

# Add to metadata (paper_meta.json) and sync symlinks
if [ -f "$PAPER_ROOT/$ARXIV_ID/paper.bib" ]; then
  echo -e "\n>>> Adding to metadata..."
  uv run python -m app add "$ARXIV_ID" || echo "Warning: failed to add to metadata"
fi