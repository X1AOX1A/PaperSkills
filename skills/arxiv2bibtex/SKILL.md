# arxiv2bibtex

Fetch BibTeX citation from arXiv.org by paper ID.

## Dependencies

- `curl`

## Usage

```bash
bash arxiv2bibtex.sh <arxiv_id> [-o OUTPUT]
```

### Examples

```bash
# Print BibTeX to stdout
bash arxiv2bibtex.sh 2501.11120

# Save to file
bash arxiv2bibtex.sh 2501.11120v1 -o paper.bib
```

## How It Works

Fetches the BibTeX entry from `https://arxiv.org/bibtex/<arxiv_id>` via `curl`.
Supports version suffixes (e.g. `2501.11120v1`).
