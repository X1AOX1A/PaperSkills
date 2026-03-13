#!/usr/bin/env sh
#
# Fetch BibTeX from arXiv.org
#
# Usage: arxiv2bibtex.sh <arxiv_id> [-o OUTPUT]
# Example: arxiv2bibtex.sh 2501.11120
# Example: arxiv2bibtex.sh 2501.11120v1 -o paper.bib
#

id=""
output=""

while [ $# -gt 0 ]; do
  case "$1" in
    -o|--output)
      output="$2"
      shift 2
      ;;
    *)
      id="$1"
      shift
      ;;
  esac
done

if [ -z "$id" ]; then
  echo "Usage: arxiv2bibtex.sh <arxiv_id> [-o OUTPUT]" >&2
  exit 1
fi

url="https://arxiv.org/bibtex/${id}"
if [ -n "$output" ]; then
  curl -sL "$url" -o "$output"
  echo "Saved to: $output"
else
  curl -sL "$url"
fi
