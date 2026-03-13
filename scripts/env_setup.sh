#!/bin/bash
#
# Set up the Python environment for PaperLib skills.
#
# Usage: bash env_setup.sh
#
# This script:
#   1. Installs uv (Python package manager) if not present
#   2. Creates a virtual environment (uv_paperskills)
#   3. Installs required Python packages (arxiv2markdown, requests)
#

set -e

# Install uv if not present
if ! command -v uv &> /dev/null; then
    echo ">>> Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    echo "uv installed. You may need to restart your shell or source your profile."
fi

# Create virtual environment
if [ ! -d "uv_paperskills" ]; then
    echo ">>> Creating virtual environment at uv_paperskills..."
    uv venv "uv_paperskills"
else
    echo ">>> Virtual environment already exists at uv_paperskills"
fi

# Activate and install dependencies
source "uv_paperskills/bin/activate"

echo ">>> Installing Python packages..."
uv pip install arxiv2markdown requests

echo ""
echo "Setup complete."
echo "  Virtual environment: uv_paperskills"
echo "  Activate with: source uv_paperskills/bin/activate"
