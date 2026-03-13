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
    - kimi_review.md: The paper reviewed by Kimi for CoolPaper.
    - paper.bib: The paper in bibtex format.