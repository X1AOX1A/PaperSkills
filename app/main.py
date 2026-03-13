"""
Paper management Flask app: API routes and symlink sync.
"""

from __future__ import annotations

import os
import pty
import re
import select
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from flask import Flask, Response, jsonify, render_template, request, send_file

from . import paper_meta

# Project root (parent of app/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_env() -> None:
    """Load .env from project root into os.environ."""
    env_path = PROJECT_ROOT / ".env"
    if not env_path.exists():
        return
    with open(env_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                k, v = k.strip(), v.strip()
                if k and v and k not in os.environ:
                    os.environ[k] = v


def get_paths() -> tuple[Path, Path, Path, Path]:
    """Return (meta_path, paper_root, folders_root, tags_root)."""
    load_env()
    meta = Path(os.environ.get("PAPER_META_FILE", "storage/paper_meta.json"))
    paper_root = Path(os.environ.get("PAPER_ROOT", "storage/papers"))
    folders_root = Path(os.environ.get("FOLDERS_ROOT", "storage/folders"))
    tags_root = Path(os.environ.get("TAGS_ROOT", "storage/tags"))
    if not meta.is_absolute():
        meta = PROJECT_ROOT / meta
    if not paper_root.is_absolute():
        paper_root = PROJECT_ROOT / paper_root
    if not folders_root.is_absolute():
        folders_root = PROJECT_ROOT / folders_root
    if not tags_root.is_absolute():
        tags_root = PROJECT_ROOT / tags_root
    return meta, paper_root, folders_root, tags_root


def sanitize_path_component(s: str) -> str:
    """Sanitize for use in filesystem path."""
    return re.sub(r'[/\\]', "_", s)


def sync_symlinks(
    meta_path: Path,
    paper_root: Path,
    folders_root: Path,
    tags_root: Path,
) -> None:
    """Rebuild FOLDERS_ROOT and TAGS_ROOT symlinks from meta."""
    papers, _ = paper_meta.load_meta(meta_path)

    folder_to_papers: dict[str, set[str]] = {}
    tag_to_papers: dict[str, set[str]] = {}

    for arxiv_id, paper in papers.items():
        for f in paper.folders:
            folder_to_papers.setdefault(f, set()).add(arxiv_id)
        for t in paper.tags:
            tag_to_papers.setdefault(t, set()).add(arxiv_id)

    def create_symlink(dir_path: Path, target: Path, name: str) -> None:
        link_path = dir_path / sanitize_path_component(name)
        target_abs = target.resolve()
        dir_path.mkdir(parents=True, exist_ok=True)
        if link_path.exists():
            link_path.unlink(missing_ok=True)
        try:
            link_path.symlink_to(target_abs, target_is_directory=True)
        except OSError:
            link_path.symlink_to(target_abs)

    # Clear and rebuild
    if folders_root.exists():
        shutil.rmtree(folders_root)
    if tags_root.exists():
        shutil.rmtree(tags_root)

    for folder_path, arxiv_ids in folder_to_papers.items():
        parts = [sanitize_path_component(p) for p in folder_path.split("/")]
        dir_path = folders_root.joinpath(*parts)
        for aid in arxiv_ids:
            paper_dir = paper_root / aid
            if paper_dir.exists():
                create_symlink(dir_path, paper_dir, aid)

    for tag, arxiv_ids in tag_to_papers.items():
        dir_path = tags_root / sanitize_path_component(tag)
        for aid in arxiv_ids:
            paper_dir = paper_root / aid
            if paper_dir.exists():
                create_symlink(dir_path, paper_dir, aid)


def create_app() -> Flask:
    app = Flask(
        __name__,
        template_folder=str(PROJECT_ROOT / "app" / "templates"),
        static_folder=str(PROJECT_ROOT / "app" / "static"),
        static_url_path="/static",
    )

    @app.route("/")
    def index():
        return render_template("index.html")

    @app.route("/api/papers")
    def list_papers():
        meta_path, _, _, _ = get_paths()
        papers, _ = paper_meta.load_meta(meta_path)
        folder_filter = request.args.get("folder")
        tag_filter = request.args.get("tag")
        result = [p.to_dict() for p in papers.values()]
        if folder_filter and folder_filter != "all":
            result = [
                p for p in result
                if folder_filter in p.get("folders", [])
                or any(f.startswith(folder_filter + "/") for f in p.get("folders", []))
            ]
        if tag_filter:
            result = [p for p in result if tag_filter in p.get("tags", [])]
        result.sort(key=lambda x: (x.get("published", "") or "", x.get("imported_at", "") or ""), reverse=True)
        return jsonify(result)

    @app.route("/api/tags")
    def list_tags():
        meta_path, _, _, _ = get_paths()
        papers, _ = paper_meta.load_meta(meta_path)
        tags = set()
        for p in papers.values():
            tags.update(p.tags or [])
        return jsonify(sorted(tags))

    @app.route("/api/papers/<arxiv_id>")
    def get_paper(arxiv_id: str):
        meta_path, _, _, _ = get_paths()
        papers, _ = paper_meta.load_meta(meta_path)
        if arxiv_id not in papers:
            return jsonify({"error": "Paper not found"}), 404
        return jsonify(papers[arxiv_id].to_dict())

    @app.route("/api/papers/<arxiv_id>", methods=["POST"])
    def update_paper(arxiv_id: str):
        meta_path, paper_root, folders_root, tags_root = get_paths()
        papers, folder_tree = paper_meta.load_meta(meta_path)
        if arxiv_id not in papers:
            return jsonify({"error": "Paper not found"}), 404
        data = request.get_json() or {}
        if "folders" in data:
            papers[arxiv_id].folders = data["folders"]
        if "tags" in data:
            papers[arxiv_id].tags = data["tags"]
        paper_meta.save_meta(meta_path, papers, folder_tree)
        sync_symlinks(meta_path, paper_root, folders_root, tags_root)
        return jsonify(papers[arxiv_id].to_dict())

    @app.route("/api/papers/<arxiv_id>", methods=["DELETE"])
    def delete_paper(arxiv_id: str):
        meta_path, paper_root, folders_root, tags_root = get_paths()
        papers, folder_tree = paper_meta.load_meta(meta_path)
        if arxiv_id not in papers:
            return jsonify({"error": "Paper not found"}), 404
        del papers[arxiv_id]
        paper_meta.save_meta(meta_path, papers, folder_tree)
        sync_symlinks(meta_path, paper_root, folders_root, tags_root)
        return jsonify({"ok": True})

    @app.route("/api/import", methods=["POST"])
    def import_paper():
        data = request.get_json() or {}
        arxiv_id = (data.get("arxiv_id") or "").strip()
        if not arxiv_id:
            return jsonify({"error": "arxiv_id required"}), 400

        meta_path, paper_root, folders_root, tags_root = get_paths()
        papers, folder_tree = paper_meta.load_meta(meta_path)
        if arxiv_id in papers:
            return jsonify({"error": "Paper already imported", "paper": papers[arxiv_id].to_dict()}), 409

        script = PROJECT_ROOT / "scripts" / "fetch_paper.sh"
        try:
            proc = subprocess.run(
                ["bash", str(script), arxiv_id],
                cwd=str(PROJECT_ROOT),
                capture_output=True,
                text=True,
                timeout=300,
            )
            if proc.returncode != 0:
                return jsonify({
                    "error": "Import failed",
                    "stderr": proc.stderr or proc.stdout or "",
                }), 500
        except subprocess.TimeoutExpired:
            return jsonify({"error": "Import timed out"}), 500
        except Exception as e:
            return jsonify({"error": str(e)}), 500

        bib_path = paper_root / arxiv_id / "paper.bib"
        title, published = paper_meta.parse_bibtex(bib_path)
        if not title:
            title = arxiv_id

        paper = paper_meta.Paper(
            arxiv_id=arxiv_id,
            title=title,
            published=published,
            imported_at=datetime.now(timezone.utc).isoformat(),
            folders=[],
            tags=[],
        )
        papers[arxiv_id] = paper
        paper_meta.save_meta(meta_path, papers, folder_tree)
        sync_symlinks(meta_path, paper_root, folders_root, tags_root)
        return jsonify(paper.to_dict())

    @app.route("/api/import/stream")
    def import_paper_stream():
        """Stream import logs via Server-Sent Events. Uses PTY for unbuffered output."""
        arxiv_id = request.args.get("arxiv_id", "").strip()
        if not arxiv_id:
            return jsonify({"error": "arxiv_id required"}), 400

        def generate(arxiv_id: str):
            meta_path, paper_root, folders_root, tags_root = get_paths()
            papers, folder_tree = paper_meta.load_meta(meta_path)
            if arxiv_id in papers:
                yield "data: [ERROR] Paper already imported\n\n"
                return

            yield "data: >>> Starting import...\n\n"

            script = PROJECT_ROOT / "scripts" / "fetch_paper.sh"
            master_fd = None
            proc = None
            try:
                master_fd, slave_fd = pty.openpty()
                proc = subprocess.Popen(
                    ["bash", str(script), arxiv_id],
                    cwd=str(PROJECT_ROOT),
                    stdout=slave_fd,
                    stderr=slave_fd,
                    stdin=slave_fd,
                )
                os.close(slave_fd)

                buf = ""
                while True:
                    ready, _, _ = select.select([master_fd], [], [], 0.5)
                    if ready:
                        try:
                            data = os.read(master_fd, 4096).decode("utf-8", errors="replace")
                            if not data:
                                break
                            buf += data
                            parts = buf.replace("\r\n", "\n").replace("\r", "\n").split("\n")
                            buf = parts.pop()
                            for line in parts:
                                line = line.strip()
                                if line:
                                    yield f"data: {line}\n\n"
                        except OSError:
                            break
                    if proc.poll() is not None:
                        break

                if buf.strip():
                    yield f"data: {buf.strip()}\n\n"
                proc.wait()
            finally:
                if master_fd is not None:
                    try:
                        os.close(master_fd)
                    except OSError:
                        pass

            if proc is not None and proc.returncode != 0:
                yield "data: [ERROR] Import failed\n\n"
                return

            try:
                bib_path = paper_root / arxiv_id / "paper.bib"
                title, published = paper_meta.parse_bibtex(bib_path)
                if not title:
                    title = arxiv_id
                paper = paper_meta.Paper(
                    arxiv_id=arxiv_id,
                    title=title,
                    published=published,
                    imported_at=datetime.now(timezone.utc).isoformat(),
                    folders=[],
                    tags=[],
                )
                papers[arxiv_id] = paper
                paper_meta.save_meta(meta_path, papers, folder_tree)
                sync_symlinks(meta_path, paper_root, folders_root, tags_root)
            except Exception as e:
                yield f"data: [ERROR] {e}\n\n"
                return
            yield "data: [DONE]\n\n"

        return Response(
            generate(arxiv_id),
            mimetype="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
                "Connection": "keep-alive",
            },
        )

    @app.route("/api/folders")
    def list_folders():
        meta_path, _, _, _ = get_paths()
        _, folder_tree = paper_meta.load_meta(meta_path)
        return jsonify(folder_tree)

    @app.route("/api/folders", methods=["POST"])
    def add_folder():
        data = request.get_json() or {}
        path = (data.get("path") or "").strip()
        parent = (data.get("parent") or "").strip()
        if not path:
            return jsonify({"error": "path required"}), 400
        meta_path, paper_root, folders_root, tags_root = get_paths()
        papers, folder_tree = paper_meta.load_meta(meta_path)
        full_path = f"{parent}/{path}" if parent else path
        all_paths = set(paper_meta.flatten_folder_tree(folder_tree))
        if full_path in all_paths:
            return jsonify({"error": "Folder already exists"}), 409
        if not paper_meta.insert_into_tree(folder_tree, parent, {"path": full_path, "children": []}):
            return jsonify({"error": "Invalid parent path"}), 400
        paper_meta.save_meta(meta_path, papers, folder_tree)
        sync_symlinks(meta_path, paper_root, folders_root, tags_root)
        return jsonify({"path": full_path})

    @app.route("/api/folders/move", methods=["PATCH"])
    def move_folder():
        data = request.get_json() or {}
        source = (data.get("source_path") or "").strip()
        target = (data.get("target_path") or "").strip()
        if not source:
            return jsonify({"error": "source_path required"}), 400
        meta_path, paper_root, folders_root, tags_root = get_paths()
        papers, folder_tree = paper_meta.load_meta(meta_path)
        ok, err = paper_meta.move_folder(folder_tree, papers, source, target)
        if not ok:
            return jsonify({"error": err}), 400
        paper_meta.save_meta(meta_path, papers, folder_tree)
        sync_symlinks(meta_path, paper_root, folders_root, tags_root)
        return jsonify({"folder_tree": folder_tree})

    @app.route("/api/folders/rename", methods=["PATCH"])
    def rename_folder():
        data = request.get_json() or {}
        path = (data.get("path") or "").strip()
        new_name = (data.get("new_name") or "").strip()
        if not path or not new_name:
            return jsonify({"error": "path and new_name required"}), 400
        meta_path, paper_root, folders_root, tags_root = get_paths()
        papers, folder_tree = paper_meta.load_meta(meta_path)
        ok, err = paper_meta.rename_folder(folder_tree, papers, path, new_name)
        if not ok:
            return jsonify({"error": err}), 400
        paper_meta.save_meta(meta_path, papers, folder_tree)
        sync_symlinks(meta_path, paper_root, folders_root, tags_root)
        return jsonify({"folder_tree": folder_tree})

    @app.route("/api/folders/delete", methods=["POST"])
    def delete_folder():
        data = request.get_json() or {}
        path = (data.get("path") or "").strip()
        if not path:
            return jsonify({"error": "path required"}), 400
        meta_path, paper_root, folders_root, tags_root = get_paths()
        papers, folder_tree = paper_meta.load_meta(meta_path)
        ok, err = paper_meta.delete_folder(folder_tree, papers, path)
        if not ok:
            return jsonify({"error": err}), 400
        paper_meta.save_meta(meta_path, papers, folder_tree)
        sync_symlinks(meta_path, paper_root, folders_root, tags_root)
        return jsonify({"folder_tree": folder_tree})

    @app.route("/papers/<path:subpath>")
    def serve_paper_file(subpath: str):
        """Serve paper files (e.g. PDF) from PAPER_ROOT."""
        _, paper_root, _, _ = get_paths()
        path = (paper_root / subpath).resolve()
        if not path.is_relative_to(paper_root.resolve()) or not path.exists() or not path.is_file():
            return jsonify({"error": "Not found"}), 404
        return send_file(path, as_attachment=False)

    return app


app = create_app()


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5001, debug=True)
