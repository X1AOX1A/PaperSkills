# arxiv2md

Convert arXiv papers to Markdown using the `arxiv2markdown` CLI tool.

## Installation

```bash
pip install arxiv2markdown
```

Or with uv:

```bash
uv pip install arxiv2markdown
```

## Usage

```bash
arxiv2md <arxiv_id> -o <output_path>
```

### Examples

```bash
arxiv2md 2501.11120 -o paper.md
arxiv2md 2602.10999 -o /path/to/output.md
```

## Notes

- The tool downloads the paper source from arXiv and converts it to Markdown
- Output includes sections, abstracts, equations, and references
- Version suffixes (v1, v2) in arXiv IDs are handled automatically
