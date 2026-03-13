# PaperSkills App Documentation

PaperSkills is a web application for managing academic papers from arXiv. It provides import, organization, and browsing of papers with folders and tags.

## Contents

- [Overview](overview.md) — Project description, features, and data model
- [Architecture](architecture.md) — Tech stack, project structure, and data flow
- [API Reference](api.md) — REST API endpoints

## Quick Start

```bash
# Environment setup
bash scripts/env_setup.sh

# Run the app
uv run python -m app
```

The app serves at http://127.0.0.1:5001 (port 5001 to avoid conflict with macOS AirPlay on 5000).
