# PaperSkills Architecture

## Tech Stack

- **Backend** — Python 3, Flask
- **Frontend** — Vanilla JavaScript, no framework
- **Templates** — Jinja2 (single `index.html`)
- **Dependencies** — arxiv2markdown, bibtexparser, flask, requests (see `pyproject.toml`)

## Project Structure

```
PaperSkills/
├── app/
│   ├── __init__.py
│   ├── __main__.py       # Entry point: python -m app
│   ├── main.py           # Flask app, routes, symlink sync
│   ├── paper_meta.py     # Metadata load/save, BibTeX, folder tree helpers
│   ├── static/
│   │   ├── app.js        # Frontend logic
│   │   └── style.css     # Styles
│   └── templates/
│       └── index.html    # Single-page UI
├── doc/                  # Documentation
├── scripts/
│   ├── env_setup.sh      # Environment setup
│   └── fetch_paper.sh    # Paper fetch (PDF, MD, BibTeX, Kimi)
├── storage/              # Data (configurable via .env)
├── .env                  # Path configuration
└── pyproject.toml
```

## Data Flow

### Import Flow

1. User enters arXiv ID and clicks Import
2. Frontend calls `GET /api/import/stream?arxiv_id=...`
3. Backend runs `fetch_paper.sh` in a PTY for unbuffered output
4. Backend streams log lines as SSE (`data: <line>\n\n`)
5. On success: parses BibTeX, saves metadata, syncs symlinks, sends `data: [DONE]\n\n`
6. On error: sends `data: [ERROR] <message>\n\n`
7. Frontend shows log in modal; auto-closes on `[DONE]`, keeps open on `[ERROR]`

### Symlink Sync

After any metadata change (import, folder add/move/rename/delete, paper update/delete), `sync_symlinks()`:

1. Reads papers and folder tree from `paper_meta.json`
2. Builds maps: folder path → arxiv_ids, tag → arxiv_ids
3. Clears `FOLDERS_ROOT` and `TAGS_ROOT`
4. Recreates symlink trees: each folder/tag directory contains symlinks to paper directories under `PAPER_ROOT`

### Folder Tree Operations

`paper_meta.py` provides helpers for the hierarchical folder tree:

- `flatten_folder_tree()` — All folder paths
- `insert_into_tree()` — Add folder at parent
- `remove_from_tree()` — Remove folder node
- `move_folder()` — Move folder (and subfolders) to new parent; updates papers' folder paths
- `rename_folder()` — Rename folder; updates papers' folder paths
- `delete_folder()` — Remove folder from tree; papers keep their other folder assignments

## Frontend Structure

- **Single-page app** — All UI in `index.html`; `app.js` handles routing, modals, and API calls
- **State** — Papers, folder tree, tags loaded via API; no persistent client state
- **Modals** — Import, New Folder, Rename Folder; shown/hidden via CSS classes
- **Import log** — Fixed-size log area (560×360px), scrollable, auto-closes on success
