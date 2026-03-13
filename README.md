# PaperSkills

A web app for managing academic papers from arXiv. Import papers, organize with folders and tags, and browse via symlinked directories.

## Quick Start

Environment setup:

```bash
bash scripts/env_setup.sh
cp .env.example .env
```

Launch the UI:

```bash
# Launch the UI
uv run python -m app
```

Serves at http://127.0.0.1:5001.

## Import Paper from CLI

This is a convenient way to import papers without the need to use the UI. Exactly the same as importing via the UI, but without the need to open the browser.

To import a paper from CLI, run:

```bash
bash scripts/fetch_paper.sh <arxiv_id>
```

This will download the paper to `PAPER_ROOT` (from `.env`, default `storage/papers`):

- `paper.pdf` — PDF from arXiv
- `paper.md` — Markdown via arxiv2md
- `kimi_review.md` — Kimi summary
- `paper.bib` — BibTeX

After fetching, the paper is added to metadata and symlinks are synced. No UI required.


## Paper Management UI

Features: import via arXiv ID, hierarchical folders, tags, folder drag-and-drop, multi-select bulk actions.

**Environment variables** (`.env`):

| Variable        | Default                  |
|-----------------|--------------------------|
| PAPER_ROOT      | storage/papers           |
| PAPER_META_FILE | storage/paper_meta.json  |
| FOLDERS_ROOT    | storage/folders          |
| TAGS_ROOT       | storage/tags             |

## Documentation

See [doc/](doc/) for full documentation: overview, architecture, API reference.
