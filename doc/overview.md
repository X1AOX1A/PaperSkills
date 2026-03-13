# PaperSkills Overview

## Description

PaperSkills is a paper management web application that helps you import, organize, and browse academic papers from arXiv. It stores papers locally and maintains a metadata index for folders and tags. The UI provides a single-page interface for importing papers by arXiv ID, organizing them into hierarchical folders, tagging them, and viewing or filtering the collection.

## Features

### Paper Import

- **Import via arXiv ID** — Enter an arXiv ID (e.g. `2512.18832v2`) to fetch and import a paper
- **Streaming import** — Real-time log output during import via Server-Sent Events (SSE)
- **Auto-close on success** — Import modal closes automatically when import completes; stays open on error for manual inspection
- **CLI import** — Run `bash scripts/fetch_paper.sh <arxiv_id>` directly; it adds the paper to metadata and syncs symlinks. No UI required.

The import process runs `scripts/fetch_paper.sh`, which downloads:

- `paper.pdf` — PDF from arXiv
- `paper.md` — Markdown conversion via arxiv2md
- `kimi_review.md` — AI summary (Kimi from CoolPaper)
- `paper.bib` — BibTeX metadata

### Organization

- **Hierarchical folders** — Create nested folders (e.g. `ML/NLP`, `ML/CV`)
- **Folder drag-and-drop** — Move folders into other folders; drag papers into folders
- **Inline tags** — Edit tags directly in the paper table
- **Tags sidebar** — Filter papers by tag

### Paper Management

- **Multi-select** — Checkboxes for bulk actions
- **Bulk actions** — Add to folder, Remove from folder, Delete
- **Context menus** — Right-click on papers (e.g. Remove from current folder) and folders (New subfolder, Rename, Delete)

### Symlink Sync

Papers are stored once under `PAPER_ROOT`. Folders and tags are implemented as symlink trees under `FOLDERS_ROOT` and `TAGS_ROOT`. Each folder/tag points to the paper directories, so you can browse papers by folder or tag via the filesystem.

## Data Model

### Paper

| Field       | Type     | Description                          |
|------------|----------|--------------------------------------|
| arxiv_id   | string   | arXiv identifier (e.g. 2512.18832v2) |
| title      | string   | Paper title from BibTeX              |
| published  | string   | Publication date (YYYY-MM)           |
| imported_at| string   | ISO timestamp of import               |
| folders    | string[] | Folder paths the paper belongs to    |
| tags       | string[] | Tags assigned to the paper           |

### Folder Tree

Folders form a tree. Each node has:

- `path` — Full path (e.g. `ML/NLP`)
- `children` — List of child folder nodes

### Storage Layout

```
storage/
├── paper_meta.json     # Metadata (papers, folder_tree)
├── papers/             # Paper files by arxiv_id
│   └── <arxiv_id>/
│       ├── paper.pdf
│       ├── paper.md
│       ├── kimi_review.md
│       └── paper.bib
├── folders/            # Symlinks by folder path
│   └── ML/
│       └── NLP/
│           └── <arxiv_id> -> ../../papers/<arxiv_id>
└── tags/               # Symlinks by tag
    └── <tag>/
        └── <arxiv_id> -> ../../papers/<arxiv_id>
```

## Configuration

Paths are configured via `.env`:

| Variable        | Default              | Description                    |
|-----------------|----------------------|--------------------------------|
| PAPER_META_FILE | storage/paper_meta.json | Metadata JSON path          |
| PAPER_ROOT      | storage/papers        | Paper files directory         |
| FOLDERS_ROOT    | storage/folders       | Folder symlink tree            |
| TAGS_ROOT       | storage/tags          | Tag symlink tree               |
