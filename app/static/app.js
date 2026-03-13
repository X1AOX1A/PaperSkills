let currentView = "all";  // "all" | "folder" | "tag"
let currentFolder = "all";
let currentTag = null;
let folderTree = [];
let tagList = [];
let selectedPapers = new Set();

function showToast(msg) {
  const el = document.createElement("div");
  el.className = "toast";
  el.textContent = msg;
  el.style.cssText = "position:fixed;bottom:20px;right:20px;background:#333;color:#fff;padding:12px 20px;border-radius:6px;z-index:2000;font-size:14px;";
  document.body.appendChild(el);
  setTimeout(() => el.remove(), 3000);
}

async function api(path, options = {}) {
  const res = await fetch(path, {
    headers: { "Content-Type": "application/json", ...options.headers },
    ...options,
  });
  const data = res.ok ? await res.json().catch(() => ({})) : await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.error || `HTTP ${res.status}`);
  return data;
}

async function loadPapers() {
  let url = "/api/papers";
  if (currentView === "folder" && currentFolder !== "all") {
    url += `?folder=${encodeURIComponent(currentFolder)}`;
  } else if (currentView === "tag" && currentTag) {
    url += `?tag=${encodeURIComponent(currentTag)}`;
  }
  const papers = await api(url);
  renderPapers(papers);
  loadTags();
}

function renderPapers(papers) {
  const tbody = document.getElementById("paper-list");
  tbody.innerHTML = papers.map((p) => {
    const imported = p.imported_at ? new Date(p.imported_at).toLocaleDateString() : "";
    const checked = selectedPapers.has(p.arxiv_id) ? "checked" : "";
    const tagChips = (p.tags || []).map((t) =>
      `<span class="tag-chip" data-tag="${escapeHtml(t)}"><span class="tag-text">${escapeHtml(t)}</span><span class="tag-remove" draggable="false">×</span></span>`
    ).join("");
    return `
      <tr class="paper-row" data-arxiv="${escapeHtml(p.arxiv_id)}" draggable="true">
        <td class="col-checkbox"><input type="checkbox" class="paper-checkbox" data-arxiv="${escapeHtml(p.arxiv_id)}" ${checked} draggable="false"></td>
        <td><a href="#" data-pdf="${escapeHtml(p.arxiv_id)}" draggable="false">${escapeHtml(p.title || p.arxiv_id)}</a></td>
        <td>${escapeHtml(p.published || "")}</td>
        <td>${imported}</td>
        <td class="paper-tags-cell" data-arxiv="${escapeHtml(p.arxiv_id)}">
          <div class="tags-cell-inner">
            ${tagChips}
            <input type="text" class="tag-input" placeholder="+ tag" draggable="false">
          </div>
        </td>
      </tr>
    `;
  }).join("") || "<tr><td colspan='5'>No papers</td></tr>";
  attachPaperDrag();
  attachTagInputs();
  attachCheckboxes();
  updateSelectionBar();
}

function escapeHtml(s) {
  const div = document.createElement("div");
  div.textContent = s;
  return div.innerHTML;
}

async function loadFolders() {
  folderTree = await api("/api/folders");
  renderFolderTree();
}

async function loadTags() {
  tagList = await api("/api/tags");
  renderTagList();
}

function renderTagList() {
  const container = document.getElementById("tag-list");
  container.innerHTML = tagList.map((tag) =>
    `<div class="tag-node ${currentView === "tag" && currentTag === tag ? "active" : ""}" data-tag="${escapeHtml(tag)}">${escapeHtml(tag)}</div>`
  ).join("");
}

function renderFolderTree() {
  const container = document.getElementById("folder-tree");
  container.innerHTML = renderFolderNodes(folderTree, 0);
  attachFolderDragDrop();
}

function renderFolderNodes(nodes, depth) {
  return nodes.map((node) => {
    const hasChildren = node.children && node.children.length > 0;
    const expand = hasChildren ? '<span class="folder-expand">▼</span>' : '<span class="folder-expand"></span>';
    const childrenHtml = hasChildren
      ? `<div class="folder-children">${renderFolderNodes(node.children, depth + 1)}</div>`
      : "";
    return `
      <div class="folder-node" data-path="${escapeHtml(node.path)}" draggable="true">
        ${expand}
        <span>${escapeHtml(node.path.split("/").pop())}</span>
      </div>
      ${childrenHtml}
    `;
  }).join("");
}

async function savePaperTags(arxivId, tags) {
  const paper = await api(`/api/papers/${arxivId}`);
  await api(`/api/papers/${arxivId}`, {
    method: "POST",
    body: JSON.stringify({ folders: paper.folders || [], tags }),
  });
}

function attachTagInputs() {
  document.querySelectorAll(".paper-tags-cell").forEach((cell) => {
    const arxivId = cell.dataset.arxiv;
    const input = cell.querySelector(".tag-input");
    input.addEventListener("keydown", async (e) => {
      if (e.key === "Enter" || e.key === ",") {
        e.preventDefault();
        const raw = input.value.trim();
        const toAdd = raw.split(",").map((s) => s.trim()).filter(Boolean);
        if (toAdd.length === 0) return;
        const chips = cell.querySelectorAll(".tag-chip");
        const existing = [...chips].map((c) => c.dataset.tag);
        const newTags = [...existing];
        for (const t of toAdd) {
          if (!newTags.includes(t)) newTags.push(t);
        }
        if (newTags.length === existing.length) {
          input.value = "";
          return;
        }
        try {
          await savePaperTags(arxivId, newTags);
          input.value = "";
          loadPapers();
        } catch (err) {
          showToast(err.message);
        }
      }
    });
  });
  document.querySelectorAll(".tag-remove").forEach((btn) => {
    btn.addEventListener("click", async (e) => {
      e.stopPropagation();
      const chip = btn.closest(".tag-chip");
      const cell = chip.closest(".paper-tags-cell");
      const arxivId = cell.dataset.arxiv;
      const removed = chip.dataset.tag;
      const chips = cell.querySelectorAll(".tag-chip");
      const tags = [...chips].map((c) => c.dataset.tag).filter((t) => t !== removed);
      try {
        await savePaperTags(arxivId, tags);
        loadPapers();
      } catch (err) {
        showToast(err.message);
      }
    });
  });
}

function updateSelectionBar() {
  const bar = document.getElementById("selection-bar");
  const countEl = document.getElementById("selection-count");
  const removeBtn = document.getElementById("remove-from-folder-btn");
  const selectAll = document.getElementById("select-all");
  const checkboxes = document.querySelectorAll(".paper-checkbox");
  if (selectAll && checkboxes.length) {
    const allChecked = checkboxes.length && [...checkboxes].every((cb) => cb.checked);
    selectAll.checked = allChecked;
    selectAll.indeterminate = selectedPapers.size > 0 && !allChecked;
  }
  if (selectedPapers.size === 0) {
    bar.classList.add("hidden");
    document.getElementById("add-to-dropdown").classList.add("hidden");
    if (selectAll) selectAll.indeterminate = false;
    return;
  }
  bar.classList.remove("hidden");
  countEl.textContent = `${selectedPapers.size} selected`;
  removeBtn.style.display = currentView === "folder" && currentFolder !== "all" ? "" : "none";
}

function attachCheckboxes() {
  document.querySelectorAll(".paper-checkbox").forEach((cb) => {
    cb.addEventListener("change", (e) => {
      e.stopPropagation();
      if (cb.checked) selectedPapers.add(cb.dataset.arxiv);
      else selectedPapers.delete(cb.dataset.arxiv);
      updateSelectionBar();
    });
    cb.addEventListener("click", (e) => e.stopPropagation());
  });
}

function attachPaperDrag() {
  document.querySelectorAll(".paper-row").forEach((row) => {
    row.addEventListener("dragstart", (e) => {
      const ids = selectedPapers.has(row.dataset.arxiv) && selectedPapers.size > 0
        ? [...selectedPapers]
        : [row.dataset.arxiv];
      e.dataTransfer.setData("text/plain", ids.length > 1 ? "papers:" + ids.join(",") : "paper:" + ids[0]);
      e.dataTransfer.effectAllowed = "copy";
      row.classList.add("dragging");
    });
    row.addEventListener("dragend", () => row.classList.remove("dragging"));
  });
}

function attachFolderDragDrop() {
  document.querySelectorAll(".folder-node").forEach((el) => {
    el.addEventListener("dragstart", (e) => {
      e.dataTransfer.setData("text/plain", "folder:" + el.dataset.path);
      el.classList.add("dragging");
    });
    el.addEventListener("dragend", () => el.classList.remove("dragging"));
    el.addEventListener("dragover", (e) => {
      e.preventDefault();
      e.dataTransfer.dropEffect = "move";
      const data = e.dataTransfer.getData("text/plain");
      const target = el.dataset.path;
      if (data.startsWith("paper:") || data.startsWith("papers:")) {
        el.classList.add("paper-drag-over");
      } else if (data.startsWith("folder:")) {
        const src = data.slice(7);
        if (src && src !== target && !target.startsWith(src + "/")) {
          el.classList.add("drag-over");
        }
      }
    });
    el.addEventListener("dragleave", () => {
      el.classList.remove("drag-over", "paper-drag-over");
    });
    el.addEventListener("drop", async (e) => {
      e.preventDefault();
      el.classList.remove("drag-over", "paper-drag-over");
      const data = e.dataTransfer.getData("text/plain");
      const target = el.dataset.path;
      if (data.startsWith("papers:")) {
        const ids = data.slice(7).split(",").filter(Boolean);
        try {
          let added = 0;
          for (const arxivId of ids) {
            const paper = await api(`/api/papers/${arxivId}`);
            const folders = paper.folders || [];
            if (!folders.includes(target)) {
              await api(`/api/papers/${arxivId}`, {
                method: "POST",
                body: JSON.stringify({ folders: [...folders, target], tags: paper.tags || [] }),
              });
              added++;
            }
          }
          if (added > 0) {
            showToast(`${added} paper(s) added to folder`);
            selectedPapers.clear();
            loadPapers();
          }
        } catch (err) {
          showToast(err.message);
        }
      } else if (data.startsWith("paper:")) {
        const arxivId = data.slice(6);
        try {
          const paper = await api(`/api/papers/${arxivId}`);
          const folders = paper.folders || [];
          if (!folders.includes(target)) {
            await api(`/api/papers/${arxivId}`, {
              method: "POST",
              body: JSON.stringify({ folders: [...folders, target], tags: paper.tags || [] }),
            });
            showToast("Paper added to folder");
            loadPapers();
          }
        } catch (err) {
          showToast(err.message);
        }
      } else if (data.startsWith("folder:")) {
        const src = data.slice(7);
        if (!src || src === target || target.startsWith(src + "/")) return;
        try {
          await api("/api/folders/move", {
            method: "PATCH",
            body: JSON.stringify({ source_path: src, target_path: target }),
          });
          await loadFolders();
          await loadPapers();
        } catch (err) {
          showToast(err.message);
        }
      }
    });
    el.addEventListener("contextmenu", (e) => {
      e.preventDefault();
      const menu = document.getElementById("folder-context-menu");
      menu.classList.remove("hidden");
      menu.dataset.folderPath = el.dataset.path;
      menu.style.left = e.pageX + "px";
      menu.style.top = e.pageY + "px";
    });
  });
}

function setActiveFolder(path) {
  currentView = path === "all" ? "all" : "folder";
  currentFolder = path;
  currentTag = null;
  document.querySelectorAll(".nav-item, .folder-node, .tag-node").forEach((el) => el.classList.remove("active"));
  if (path === "all") {
    document.querySelector('.nav-item[data-folder="all"]').classList.add("active");
  } else {
    const node = document.querySelector(`.folder-node[data-path="${path}"]`);
    if (node) node.classList.add("active");
  }
  loadPapers();
  loadTags();
}

function setActiveTag(tag) {
  currentView = "tag";
  currentTag = tag;
  currentFolder = "all";
  document.querySelectorAll(".nav-item, .folder-node, .tag-node").forEach((el) => el.classList.remove("active"));
  const node = document.querySelector(`.tag-node[data-tag="${tag}"]`);
  if (node) node.classList.add("active");
  loadPapers();
}

document.querySelector('.nav-item[data-folder="all"]').addEventListener("click", () => setActiveFolder("all"));

document.getElementById("folder-tree").addEventListener("click", (e) => {
  const node = e.target.closest(".folder-node");
  if (node) setActiveFolder(node.dataset.path);
});

document.getElementById("tag-list").addEventListener("click", (e) => {
  const node = e.target.closest(".tag-node");
  if (node) setActiveTag(node.dataset.tag);
});

document.getElementById("import-btn").addEventListener("click", () => {
  document.getElementById("import-modal").classList.remove("hidden");
  document.getElementById("import-form").classList.remove("hidden");
  document.getElementById("import-log-area").classList.add("hidden");
  document.getElementById("arxiv-id-input").value = "";
  document.getElementById("import-error").classList.add("hidden");
});

document.getElementById("import-cancel").addEventListener("click", () => {
  document.getElementById("import-modal").classList.add("hidden");
});

document.getElementById("import-submit").addEventListener("click", () => {
  const arxivId = document.getElementById("arxiv-id-input").value.trim();
  const errEl = document.getElementById("import-error");
  if (!arxivId) {
    errEl.textContent = "Enter arXiv ID";
    errEl.classList.remove("hidden");
    return;
  }
  errEl.classList.add("hidden");
  document.getElementById("import-form").classList.add("hidden");
  document.getElementById("import-log-area").classList.remove("hidden");
  document.getElementById("import-modal").querySelector(".modal-content").classList.add("import-log-view");
  document.getElementById("import-log-arxiv").textContent = arxivId;
  document.getElementById("import-log").textContent = "";
  document.getElementById("import-close-btn").classList.add("hidden");

  const logEl = document.getElementById("import-log");

  const appendLog = (text) => {
    logEl.textContent += text + "\n";
    logEl.scrollTop = logEl.scrollHeight;
  };

  (async () => {
    const res = await fetch("/api/import/stream?arxiv_id=" + encodeURIComponent(arxivId));
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const events = buffer.split("\n\n");
      buffer = events.pop();
      for (const evt of events) {
        const data = evt.replace(/^data: /gm, "").trim();
        if (!data) continue;
        if (data === "[DONE]") {
          appendLog(">>> Import completed.");
          showToast("Import completed");
          loadPapers();
          setTimeout(closeImportModal, 300);
        } else if (data.startsWith("[ERROR]")) {
          appendLog(data);
          document.getElementById("import-close-btn").classList.remove("hidden");
          showToast(data.slice(7).trim() || "Import failed");
        } else {
          appendLog(data);
        }
      }
    }
    if (buffer.trim()) {
      const data = buffer.replace(/^data: /gm, "").trim();
      if (data === "[DONE]") {
        appendLog(">>> Import completed.");
        showToast("Import completed");
        loadPapers();
        setTimeout(closeImportModal, 300);
      } else if (data.startsWith("[ERROR]")) {
        appendLog(data);
        document.getElementById("import-close-btn").classList.remove("hidden");
        showToast(data.slice(7).trim() || "Import failed");
      } else if (data) {
        appendLog(data);
      }
    }
  })();
});

function closeImportModal() {
  document.getElementById("import-log-area").classList.add("hidden");
  document.getElementById("import-form").classList.remove("hidden");
  document.getElementById("import-modal").classList.add("hidden");
  document.getElementById("import-modal").querySelector(".modal-content").classList.remove("import-log-view");
}

document.getElementById("import-close-btn").addEventListener("click", closeImportModal);

document.getElementById("new-folder-btn").addEventListener("click", () => {
  const modal = document.getElementById("new-folder-modal");
  modal.classList.remove("hidden");
  modal.dataset.parent = "";
  document.getElementById("folder-name-input").value = "";
  document.getElementById("folder-error").classList.add("hidden");
});

document.getElementById("folder-cancel").addEventListener("click", () => {
  document.getElementById("new-folder-modal").classList.add("hidden");
});

document.getElementById("folder-submit").addEventListener("click", async () => {
  const name = document.getElementById("folder-name-input").value.trim();
  const parent = document.getElementById("new-folder-modal").dataset.parent || "";
  const errEl = document.getElementById("folder-error");
  if (!name) {
    errEl.textContent = "Enter folder name";
    errEl.classList.remove("hidden");
    return;
  }
  errEl.classList.add("hidden");
  try {
    await api("/api/folders", {
      method: "POST",
      body: JSON.stringify({ path: name, parent }),
    });
    document.getElementById("new-folder-modal").classList.add("hidden");
    showToast("Folder created");
    loadFolders();
  } catch (err) {
    errEl.textContent = err.message;
    errEl.classList.remove("hidden");
  }
});

document.getElementById("folder-context-menu").addEventListener("click", (e) => {
  const item = e.target.closest("[data-action]");
  if (!item) return;
  const menu = document.getElementById("folder-context-menu");
  const folderPath = menu.dataset.folderPath || "";
  menu.classList.add("hidden");
  if (item.dataset.action === "new-subfolder") {
    const modal = document.getElementById("new-folder-modal");
    modal.classList.remove("hidden");
    modal.dataset.parent = folderPath;
    document.getElementById("folder-name-input").value = "";
    document.getElementById("folder-error").classList.add("hidden");
  } else if (item.dataset.action === "rename") {
    const modal = document.getElementById("rename-folder-modal");
    modal.classList.remove("hidden");
    modal.dataset.folderPath = folderPath;
    document.getElementById("rename-folder-input").value = folderPath.split("/").pop();
    document.getElementById("rename-folder-input").focus();
    document.getElementById("rename-folder-error").classList.add("hidden");
  } else if (item.dataset.action === "delete") {
    if (!confirm("Remove this folder? Only the folder is removed; papers stay in the library.")) return;
    (async () => {
      try {
        await api("/api/folders/delete", {
          method: "POST",
          body: JSON.stringify({ path: folderPath }),
        });
        showToast("Folder removed");
        if (currentFolder === folderPath || currentFolder.startsWith(folderPath + "/")) {
          setActiveFolder("all");
        }
        loadFolders();
        loadPapers();
      } catch (err) {
        showToast(err.message);
      }
    })();
  }
});

document.getElementById("rename-folder-cancel").addEventListener("click", () => {
  document.getElementById("rename-folder-modal").classList.add("hidden");
});

document.getElementById("rename-folder-submit").addEventListener("click", async () => {
  const path = document.getElementById("rename-folder-modal").dataset.folderPath || "";
  const newName = document.getElementById("rename-folder-input").value.trim();
  const errEl = document.getElementById("rename-folder-error");
  if (!newName) {
    errEl.textContent = "Enter folder name";
    errEl.classList.remove("hidden");
    return;
  }
  errEl.classList.add("hidden");
  try {
    await api("/api/folders/rename", {
      method: "PATCH",
      body: JSON.stringify({ path, new_name: newName }),
    });
    document.getElementById("rename-folder-modal").classList.add("hidden");
    showToast("Folder renamed");
    const parent = path.includes("/") ? path.split("/").slice(0, -1).join("/") : "";
    const newPath = parent ? `${parent}/${newName}` : newName;
    if (currentFolder === path || currentFolder.startsWith(path + "/")) {
      const newCurrent = currentFolder === path ? newPath : newPath + currentFolder.slice(path.length);
      await loadFolders();
      setActiveFolder(newCurrent);
    } else {
      loadFolders();
    }
  } catch (err) {
    errEl.textContent = err.message;
    errEl.classList.remove("hidden");
  }
});


document.getElementById("paper-list").addEventListener("click", (e) => {
  const link = e.target.closest("a[data-pdf]");
  if (link) {
    e.preventDefault();
    const arxivId = link.dataset.pdf;
    window.open(`/papers/${arxivId}/paper.pdf`, "_blank");
  }
});

document.getElementById("add-to-btn").addEventListener("click", (e) => {
  e.stopPropagation();
  const dropdown = document.getElementById("add-to-dropdown");
  if (dropdown.classList.contains("hidden")) {
    dropdown.classList.remove("hidden");
    const paths = flattenFolderTree(folderTree);
    dropdown.innerHTML = paths.length
      ? paths.map((path) =>
          `<button class="dropdown-item" data-path="${escapeHtml(path)}">${escapeHtml(path)}</button>`
        ).join("")
      : '<div class="dropdown-item" style="cursor:default;color:#999">No folders</div>';
    dropdown.style.left = "0";
    dropdown.style.top = "100%";
    dropdown.querySelectorAll(".dropdown-item[data-path]").forEach((item) => {
      item.addEventListener("click", async (ev) => {
        ev.stopPropagation();
        const target = item.dataset.path;
        try {
          let added = 0;
          for (const arxivId of selectedPapers) {
            const paper = await api(`/api/papers/${arxivId}`);
            const folders = paper.folders || [];
            if (!folders.includes(target)) {
              await api(`/api/papers/${arxivId}`, {
                method: "POST",
                body: JSON.stringify({ folders: [...folders, target], tags: paper.tags || [] }),
              });
              added++;
            }
          }
          showToast(`${added} paper(s) added to folder`);
          selectedPapers.clear();
          dropdown.classList.add("hidden");
          loadPapers();
          updateSelectionBar();
        } catch (err) {
          showToast(err.message);
        }
      });
    });
  } else {
    dropdown.classList.add("hidden");
  }
});

function flattenFolderTree(nodes) {
  const paths = [];
  function walk(n) {
    for (const node of n) {
      paths.push(node.path);
      if (node.children?.length) walk(node.children);
    }
  }
  walk(nodes);
  return paths;
}

document.getElementById("remove-from-folder-btn").addEventListener("click", async () => {
  if (currentFolder === "all") return;
  try {
    let count = 0;
    for (const arxivId of selectedPapers) {
      const paper = await api(`/api/papers/${arxivId}`);
      const folders = (paper.folders || []).filter((f) => f !== currentFolder);
      if (folders.length !== (paper.folders || []).length) {
        await api(`/api/papers/${arxivId}`, {
          method: "POST",
          body: JSON.stringify({ folders, tags: paper.tags || [] }),
        });
        count++;
      }
    }
    showToast(`${count} paper(s) removed from folder`);
    selectedPapers.clear();
    loadPapers();
    updateSelectionBar();
  } catch (err) {
    showToast(err.message);
  }
});

document.getElementById("delete-btn").addEventListener("click", async () => {
  if (!confirm(`Delete ${selectedPapers.size} paper(s) from library? (Files will remain on disk)`)) return;
  try {
    for (const arxivId of selectedPapers) {
      await api(`/api/papers/${arxivId}`, { method: "DELETE" });
    }
    showToast(`${selectedPapers.size} paper(s) deleted`);
    selectedPapers.clear();
    loadPapers();
    updateSelectionBar();
  } catch (err) {
    showToast(err.message);
  }
});

document.getElementById("paper-list").addEventListener("contextmenu", (e) => {
  const row = e.target.closest(".paper-row");
  if (!row || currentView !== "folder" || currentFolder === "all") return;
  e.preventDefault();
  const menu = document.getElementById("paper-context-menu");
  menu.classList.remove("hidden");
  menu.dataset.arxiv = row.dataset.arxiv;
  menu.style.left = e.pageX + "px";
  menu.style.top = e.pageY + "px";
});

document.getElementById("paper-context-menu").addEventListener("click", (e) => {
  const item = e.target.closest("[data-action]");
  if (!item || item.dataset.action !== "remove-from-folder") return;
  const arxivId = document.getElementById("paper-context-menu").dataset.arxiv;
  document.getElementById("paper-context-menu").classList.add("hidden");
  (async () => {
    try {
      const paper = await api(`/api/papers/${arxivId}`);
      const folders = (paper.folders || []).filter((f) => f !== currentFolder);
      await api(`/api/papers/${arxivId}`, {
        method: "POST",
        body: JSON.stringify({ folders, tags: paper.tags || [] }),
      });
      showToast("Removed from folder");
      loadPapers();
    } catch (err) {
      showToast(err.message);
    }
  })();
});

document.addEventListener("click", () => {
  document.getElementById("add-to-dropdown").classList.add("hidden");
  document.getElementById("folder-context-menu").classList.add("hidden");
  document.getElementById("paper-context-menu").classList.add("hidden");
});

document.getElementById("select-all").addEventListener("change", (e) => {
  const checked = e.target.checked;
  document.querySelectorAll(".paper-checkbox").forEach((cb) => {
    cb.checked = checked;
    if (checked) selectedPapers.add(cb.dataset.arxiv);
    else selectedPapers.delete(cb.dataset.arxiv);
  });
  updateSelectionBar();
});

document.querySelector('.nav-item[data-folder="all"]').classList.add("active");
loadFolders();
loadPapers();
