"""
Paper metadata: load/save JSON, parse BibTeX, folder tree helpers.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import bibtexparser


@dataclass
class Paper:
    arxiv_id: str
    title: str
    published: str  # year-month, e.g. "2025-01"
    imported_at: str  # ISO timestamp
    folders: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "arxiv_id": self.arxiv_id,
            "title": self.title,
            "published": self.published,
            "imported_at": self.imported_at,
            "folders": self.folders,
            "tags": self.tags,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Paper:
        return cls(
            arxiv_id=d["arxiv_id"],
            title=d.get("title", ""),
            published=d.get("published", ""),
            imported_at=d.get("imported_at", ""),
            folders=d.get("folders", []),
            tags=d.get("tags", []),
        )


FolderNode = dict[str, Any]  # {"path": str, "children": list[FolderNode]}


def load_meta(meta_path: Path) -> tuple[dict[str, Paper], list[FolderNode]]:
    """Load meta from JSON. Returns (papers, folder_tree)."""
    if not meta_path.exists():
        return {}, []

    with open(meta_path, encoding="utf-8") as f:
        data = json.load(f)

    papers = {
        k: Paper.from_dict(v)
        for k, v in data.get("papers", {}).items()
    }
    folder_tree = data.get("folder_tree", [])
    return papers, folder_tree


def save_meta(
    meta_path: Path,
    papers: dict[str, Paper],
    folder_tree: list[FolderNode],
) -> None:
    """Save meta to JSON."""
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "papers": {k: v.to_dict() for k, v in papers.items()},
        "folder_tree": folder_tree,
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def parse_bibtex(bib_path: Path) -> tuple[str, str]:
    """
    Parse BibTeX file for title and year-month.
    Returns (title, published) where published is "YYYY-MM" or empty.
    """
    if not bib_path.exists():
        return "", ""

    with open(bib_path, encoding="utf-8") as f:
        db = bibtexparser.load(f)

    if not db.entries:
        return "", ""

    entry = db.entries[0]
    title = entry.get("title", "").strip()
    if title.startswith("{") and title.endswith("}"):
        title = title[1:-1]
    year = entry.get("year", "")
    month = entry.get("month", "")

    # BibTeX month can be "jan", "1", "01", etc.
    month_map = {
        "jan": "01", "feb": "02", "mar": "03", "apr": "04",
        "may": "05", "jun": "06", "jul": "07", "aug": "08",
        "sep": "09", "oct": "10", "nov": "11", "dec": "12",
    }
    month = str(month).strip() if month else ""
    if month and month.lower() in month_map:
        month = month_map[month.lower()]
    elif month and len(month) == 1:
        month = f"0{month}"
    elif month and month.isdigit() and len(month) == 2:
        pass
    elif not month:
        month = "01"

    published = f"{year}-{month}" if year else ""
    return title, published


def flatten_folder_tree(root: list[FolderNode]) -> list[str]:
    """Return all folder paths from the tree (depth-first)."""
    paths: list[str] = []

    def walk(nodes: list[FolderNode]) -> None:
        for node in nodes:
            paths.append(node["path"])
            walk(node.get("children", []))

    walk(root)
    return paths


def find_path_in_tree(tree: list[FolderNode], path: str) -> tuple[list[FolderNode], int] | None:
    """
    Find a node by path. Returns (parent_list, index) or None.
    Nodes store full path (e.g. "ML/Vision").
    """
    if not path:
        return None

    parts = path.split("/")

    def search(nodes: list[FolderNode], depth: int) -> tuple[list[FolderNode], int] | None:
        if depth >= len(parts):
            return None
        target = "/".join(parts[: depth + 1])
        for i, node in enumerate(nodes):
            if node["path"] == target:
                if depth == len(parts) - 1:
                    return (nodes, i)
                return search(node.get("children", []), depth + 1)
        return None

    return search(tree, 0)


def remove_from_tree(tree: list[FolderNode], path: str) -> FolderNode | None:
    """Remove node at path from tree. Returns the removed node or None."""
    if not path:
        return None

    parts = path.split("/")

    def search(nodes: list[FolderNode], depth: int) -> FolderNode | None:
        if depth >= len(parts):
            return None
        target = "/".join(parts[: depth + 1])
        for i, node in enumerate(nodes):
            if node["path"] == target:
                if depth == len(parts) - 1:
                    return nodes.pop(i)
                return search(node.get("children", []), depth + 1)
        return None

    return search(tree, 0)


def _ensure_parent_path(tree: list[FolderNode], path: str) -> list[FolderNode]:
    """Ensure path exists in tree, creating parents if needed. Returns the children list of the final node."""
    if not path:
        return tree
    parts = path.split("/")

    def ensure(nodes: list[FolderNode], depth: int) -> list[FolderNode]:
        if depth >= len(parts):
            return nodes
        target = "/".join(parts[: depth + 1])
        for n in nodes:
            if n["path"] == target:
                if depth == len(parts) - 1:
                    return n.setdefault("children", [])
                return ensure(n.get("children", []), depth + 1)
        parent_node = {"path": target, "children": []}
        nodes.append(parent_node)
        if depth == len(parts) - 1:
            return parent_node["children"]
        return ensure(parent_node["children"], depth + 1)

    return ensure(tree, 0)


def insert_into_tree(tree: list[FolderNode], parent_path: str, node: FolderNode) -> bool:
    """Insert node as child of parent_path. Returns True if inserted. Creates parent chain if missing."""
    if not parent_path:
        tree.append(node)
        return True

    children = _ensure_parent_path(tree, parent_path)
    children.append(node)
    return True


def is_descendant(ancestor: str, path: str) -> bool:
    """True if path is ancestor or a descendant of ancestor."""
    if path == ancestor:
        return True
    return path.startswith(ancestor + "/")


def _rewrite_paths_in_node(node: FolderNode, old_prefix: str, new_prefix: str) -> None:
    """Recursively rewrite paths in node and children."""
    if node["path"].startswith(old_prefix):
        node["path"] = new_prefix + node["path"][len(old_prefix):]
    for child in node.get("children", []):
        _rewrite_paths_in_node(child, old_prefix, new_prefix)


def move_folder(
    folder_tree: list[FolderNode],
    papers: dict[str, Paper],
    source_path: str,
    target_path: str,
) -> tuple[bool, str]:
    """
    Move folder source_path to be child of target_path.
    Returns (success, error_message).
    Rejects if target is source or descendant of source.
    """
    if not source_path:
        return False, "Invalid paths"

    if target_path and is_descendant(source_path, target_path):
        return False, "Cannot move folder into itself or its descendant"

    node = remove_from_tree(folder_tree, source_path)
    if node is None:
        return False, "Source folder not found"

    leaf = source_path.split("/")[-1]
    new_path = f"{target_path}/{leaf}" if target_path else leaf

    _rewrite_paths_in_node(node, source_path, new_path)

    if not insert_into_tree(folder_tree, target_path, node):
        # Restore on failure
        parent = "/".join(source_path.split("/")[:-1]) if "/" in source_path else ""
        insert_into_tree(folder_tree, parent, node)
        return False, "Invalid target path"

    # Update papers that reference old path or descendants
    prefix = source_path + "/"
    new_prefix = new_path + "/"
    for paper in papers.values():
        new_folders = []
        for f in paper.folders:
            if f == source_path:
                new_folders.append(new_path)
            elif f.startswith(prefix):
                new_folders.append(new_prefix + f[len(prefix):])
            else:
                new_folders.append(f)
        paper.folders = new_folders

    return True, ""


def rename_folder(
    folder_tree: list[FolderNode],
    papers: dict[str, Paper],
    old_path: str,
    new_name: str,
) -> tuple[bool, str]:
    """
    Rename folder at old_path to new_name (leaf name only).
    Returns (success, error_message).
    """
    if not old_path or not new_name:
        return False, "Invalid path or name"
    new_name = new_name.strip()
    if not new_name:
        return False, "Invalid name"
    parent = "/".join(old_path.split("/")[:-1]) if "/" in old_path else ""
    new_path = f"{parent}/{new_name}" if parent else new_name
    if new_path == old_path:
        return True, ""
    all_paths = set(flatten_folder_tree(folder_tree))
    if new_path in all_paths:
        return False, "Folder already exists"
    node = remove_from_tree(folder_tree, old_path)
    if node is None:
        return False, "Folder not found"
    prefix = old_path + "/"
    new_prefix = new_path + "/"
    _rewrite_paths_in_node(node, old_path, new_path)
    insert_into_tree(folder_tree, parent, node)
    for paper in papers.values():
        new_folders = []
        for f in paper.folders:
            if f == old_path:
                new_folders.append(new_path)
            elif f.startswith(prefix):
                new_folders.append(new_prefix + f[len(prefix):])
            else:
                new_folders.append(f)
        paper.folders = new_folders
    return True, ""


def delete_folder(
    folder_tree: list[FolderNode],
    papers: dict[str, Paper],
    path: str,
) -> tuple[bool, str]:
    """
    Remove folder from tree. Remove this path and descendants from all papers
    (papers stay in library). Returns (success, error_message).
    """
    if not path:
        return False, "Invalid path"
    node = remove_from_tree(folder_tree, path)
    if node is None:
        return False, "Folder not found"
    prefix = path + "/"
    for paper in papers.values():
        paper.folders = [f for f in paper.folders if f != path and not f.startswith(prefix)]
    return True, ""
