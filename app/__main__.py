"""Entry point for python -m app."""

import sys

from .main import add_paper_to_meta, app

if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "add":
        arxiv_id = sys.argv[2].strip()
        if not arxiv_id:
            print("Usage: python -m app add <arxiv_id>", file=sys.stderr)
            sys.exit(1)
        err = add_paper_to_meta(arxiv_id)
        if err:
            print(f"Error: {err}", file=sys.stderr)
            sys.exit(1)
        print(f"Added {arxiv_id} to metadata")
    else:
        app.run(host="127.0.0.1", port=5001, debug=True)
