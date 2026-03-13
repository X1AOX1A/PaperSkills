# Paper Skills

## Env Setup

```bash
bash scripts/env_setup.sh
```

## Fetch Paper

```bash
bash scripts/fetch_paper.sh <arxiv_id>
```

Example:
```bash
bash scripts/fetch_paper.sh 2512.18832v2
```

This will download

- storage/
  - <arxiv_id>/
    - paper.pdf: The paper in PDF format.
    - paper.md: The paper in markdown format.
    - kimi_review.md: The paper reviewed by Kimi from CoolPaper.
    - paper.bib: The bibtex file of the paper.

## Paper Management UI

Setup environment variables:

```
cp .env.example .env
```

Launch the UI:

```bash
uv run python -m app
```

Serves at http://127.0.0.1:5001.