#!/bin/bash
#
# Set up the Python environment for PaperSkills.
#
# Usage: bash scripts/env_setup.sh
#

set -e

if ! command -v uv &> /dev/null; then
    echo ">>> Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    echo "uv installed. You may need to restart your shell."
fi

echo ">>> Syncing dependencies..."
uv sync

echo ""
echo "Setup complete. Venv: .venv"
