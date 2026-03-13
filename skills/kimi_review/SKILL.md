# kimi_review

Fetch Kimi AI-generated paper summary from papers.cool and save as Markdown.

## Dependencies

- Python 3
- `requests` (`pip install requests`)

## Usage

```bash
python kimi_review.py <arxiv_id> [-o OUTPUT]
```

### Examples

```bash
# Save with default filename (<arxiv_id>.md)
python kimi_review.py 2602.10999

# Save to specific file
python kimi_review.py 2602.10999 -o kimi_review.md
```

## How It Works

1. Fetches the paper title from `https://papers.cool/arxiv/<arxiv_id>`
2. POSTs to `https://papers.cool/arxiv/kimi?paper=<arxiv_id>` to get the Kimi-generated HTML blog
3. Converts the HTML Q&A format to clean Markdown
4. Assembles a final Markdown file with title, source links, and the Q&A content

## Notes

- Version suffixes (v1, v2) are automatically stripped for the API call
- If Kimi has not yet generated a blog for the paper, visit the papers.cool page and click the [Kimi] button to trigger generation, then retry
