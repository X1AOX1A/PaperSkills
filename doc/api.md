# PaperSkills API Reference

Base URL: `http://127.0.0.1:5001`

All JSON APIs use `Content-Type: application/json` for request bodies.

---

## Papers

### List papers

```
GET /api/papers
```

**Query parameters:**

| Name   | Type   | Description                          |
|--------|--------|--------------------------------------|
| folder | string | Filter by folder path (omit for all) |
| tag    | string | Filter by tag                        |

**Response:** `200 OK` — Array of paper objects, sorted by `published` and `imported_at` (descending).

---

### Get paper

```
GET /api/papers/<arxiv_id>
```

**Response:** `200 OK` — Paper object.  
**Errors:** `404` — Paper not found.

---

### Update paper

```
POST /api/papers/<arxiv_id>
```

**Body:**

```json
{
  "folders": ["ML/NLP", "Reading"],
  "tags": ["transformer", "attention"]
}
```

**Response:** `200 OK` — Updated paper object.  
**Errors:** `404` — Paper not found.

---

### Delete paper

```
DELETE /api/papers/<arxiv_id>
```

**Response:** `200 OK` — `{"ok": true}`.  
**Errors:** `404` — Paper not found.

---

## Import

### Import paper (streaming)

```
GET /api/import/stream?arxiv_id=<arxiv_id>
```

Streams import logs as Server-Sent Events (SSE). Each event is `data: <line>\n\n`.

**Special events:**

- `data: [DONE]\n\n` — Import completed successfully
- `data: [ERROR] <message>\n\n` — Import failed

**Response:** `200 OK` — SSE stream.  
**Errors:** `400` — Missing or invalid `arxiv_id`.

---

### Import paper (non-streaming)

```
POST /api/import
```

**Body:**

```json
{
  "arxiv_id": "2512.18832v2"
}
```

**Response:** `200 OK` — Imported paper object.  
**Errors:**

- `400` — Missing `arxiv_id`
- `409` — Paper already imported
- `500` — Import failed or timed out

---

## Folders

### List folders

```
GET /api/folders
```

**Response:** `200 OK` — Folder tree (array of `{path, children}` nodes).

---

### Create folder

```
POST /api/folders
```

**Body:**

```json
{
  "path": "NLP",
  "parent": "ML"
}
```

Use `parent: ""` for a top-level folder.

**Response:** `200 OK` — `{"path": "ML/NLP"}`.  
**Errors:**

- `400` — Missing `path` or invalid parent
- `409` — Folder already exists

---

### Move folder

```
PATCH /api/folders/move
```

**Body:**

```json
{
  "source_path": "ML/NLP",
  "target_path": "Reading"
}
```

Moves folder and all subfolders. Papers in the folder are updated to the new path.

**Response:** `200 OK` — `{"folder_tree": [...]}`.  
**Errors:** `400` — Invalid source/target (e.g. moving into self or descendant).

---

### Rename folder

```
PATCH /api/folders/rename
```

**Body:**

```json
{
  "path": "ML/NLP",
  "new_name": "NLP2"
}
```

**Response:** `200 OK` — `{"folder_tree": [...]}`.  
**Errors:** `400` — Invalid path or new_name.

---

### Delete folder

```
POST /api/folders/delete
```

**Body:**

```json
{
  "path": "ML/NLP"
}
```

Removes the folder from the tree. Papers keep their other folder assignments.

**Response:** `200 OK` — `{"folder_tree": [...]}`.  
**Errors:** `400` — Invalid path.

---

## Tags

### List tags

```
GET /api/tags
```

**Response:** `200 OK` — Sorted array of tag strings.

---

## Static Files

### Serve paper file

```
GET /papers/<path>
```

Serves files from `PAPER_ROOT` (e.g. PDF, Markdown). Path must be under `PAPER_ROOT`.

**Response:** `200 OK` — File content.  
**Errors:** `404` — Not found or path escape.
