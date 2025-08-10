#!/usr/bin/env python3
"""
FORCEPS: Forensic Optimized Retrieval & Clustering of Evidence via Perceptual Search
Two-phase Streamlit app with local embeddings + optional Ollama captions.
All processing is local. Ollama/PhotoDNA optional.
"""
import os
import time
import threading
import re
from datetime import datetime as dt, time as dtime, date as ddate
import urllib.parse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("KMP_INIT_AT_FORK", "FALSE")

import streamlit as st
import base64
import mimetypes
import html
import platform
import sys
import subprocess
import shutil
from PIL import Image
import numpy as np
import faiss
try:
    faiss.omp_set_num_threads(1)
except Exception:
    pass

from app.embeddings import (
    load_models,
    compute_batch_embeddings,
    preprocess_image_for_vit_clip,
    compute_clip_text_embedding,
)
from app.utils import (
    is_image_file,
    fingerprint,
    load_cache,
    save_cache,
    compute_perceptual_hashes,
    read_exif,
    load_bookmarks,
    save_bookmarks,
    generate_bookmarks_csv,
    generate_bookmarks_pdf,
    _gather_metadata_for_path,
)
from app.llm_ollama import ollama_installed, generate_caption_ollama, model_available

# ---------------- Config ----------------
CACHE_DIR = Path("./cache")
CACHE_DIR.mkdir(exist_ok=True)
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 16))
MAX_WORKERS = int(os.environ.get("MAX_WORKERS", 8))
DEVICE = "cuda" if __import__("torch").cuda.is_available() else "cpu"

# Scaling/Index config
USE_PCA = bool(int(os.environ.get("USE_PCA", "0")))
PCA_DIM = int(os.environ.get("PCA_DIM", "384"))
IVF_NLIST = int(os.environ.get("IVF_NLIST", "4096"))
PQ_M = int(os.environ.get("PQ_M", "64"))
TRAIN_SAMPLES = int(os.environ.get("TRAIN_SAMPLES", "5000"))
ADD_BATCH = int(os.environ.get("ADD_BATCH", "8192"))
WATCH_INTERVAL = float(os.environ.get("WATCH_INTERVAL", "5"))

# ---------------- UI ----------------
st.set_page_config(layout="wide", page_title="FORCEPS")

# Minimal, professional header
st.markdown(
    """
    <style>
      .app-header {padding: 8px 0 4px 0; border-bottom: 1px solid #e5e7eb; margin-bottom: 6px;}
      .app-title {font-size: 20px; font-weight: 600; color: #111827; margin: 0;}
      .app-subtitle {font-size: 13px; color: #6b7280; margin: 2px 0 0 0;}
      html, body, * { -webkit-font-smoothing: antialiased; -moz-osx-font-smoothing: grayscale; }
      .tag-chip {display:inline-block; padding:2px 8px; background:#f3f4f6; border:1px solid #e5e7eb; border-radius:999px; font-size:11px; color:#374151; margin:2px 4px 0 0;}
      .bm-row { display:flex; align-items:center; gap:8px; margin-top:4px; }
      .bm-overlay { position: relative; margin-top: -28px; height: 0; z-index: 5; }
      .bm-icon { position: absolute; right: 6px; top: -22px; width: 24px; height: 24px; border-radius: 999px; background: rgba(255,255,255,0.9); border: 1px solid #e5e7eb; display:flex; align-items:center; justify-content:center; color: #111827; text-decoration:none; font-weight:600; box-shadow: 0 1px 2px rgba(0,0,0,0.06); }
      .bm-icon:hover { background: rgba(255,255,255,1); }
      .bm-icon.active { color:#b45309; border-color:#f59e0b; background:#fff8eb; }
      .img-card { position: relative; }
      .img-card img { width: 100%; height: auto; display: block; border-radius: 4px; }
      /* removed info overlay styles */
    </style>
    <div class="app-header">
      <div class="app-title">FORCEPS</div>
      <div class="app-subtitle">Forensic Optimized Retrieval & Clustering of Evidence via Perceptual Search</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Load models (ViT + CLIP)
vit_model, clip_model, preprocess_vit, preprocess_clip, vit_dim, clip_dim = load_models()
COMBINED_DIM = vit_dim + clip_dim

# Session state init
if "phase1_ready" not in st.session_state:
    st.session_state.phase1_ready = False
if "phase2_ready" not in st.session_state:
    st.session_state.phase2_ready = False
if "index_paths" not in st.session_state:
    st.session_state.index_paths = []
if "faiss_combined" not in st.session_state:
    st.session_state.faiss_combined = None
if "faiss_clip" not in st.session_state:
    st.session_state.faiss_clip = None
if "pca_obj" not in st.session_state:
    st.session_state.pca_obj = None
if "index_lock" not in st.session_state:
    st.session_state.index_lock = threading.Lock()
if "watcher_running" not in st.session_state:
    st.session_state.watcher_running = False
if "faiss_partial" not in st.session_state:
    st.session_state.faiss_partial = None  # IndexIDMap2 over IndexFlatL2 for combined embeddings
if "faiss_clip_partial" not in st.session_state:
    st.session_state.faiss_clip_partial = None  # IndexIDMap2 for CLIP embeddings during indexing
if "bookmarks" not in st.session_state:
    # image_path -> {"tags": List[str], "added_ts": float}
    st.session_state.bookmarks = load_bookmarks()
if "last_results" not in st.session_state:
    # Persist last shown results across UI interactions (e.g., bookmarking)
    st.session_state.last_results = []
if "folder_path" not in st.session_state:
    st.session_state.folder_path = str(Path(__file__).parent / "demo_images")
if "browse_dir" not in st.session_state:
    st.session_state.browse_dir = st.session_state.folder_path
if "phase1_duration" not in st.session_state:
    st.session_state.phase1_duration = None
if "phase2_duration" not in st.session_state:
    st.session_state.phase2_duration = None
if "total_index_duration" not in st.session_state:
    st.session_state.total_index_duration = None
if "index_start_time" not in st.session_state:
    st.session_state.index_start_time = None
if "indexing_running" not in st.session_state:
    st.session_state.indexing_running = False
if "p1_done" not in st.session_state:
    st.session_state.p1_done = 0
if "p1_total" not in st.session_state:
    st.session_state.p1_total = 0
if "p2_done" not in st.session_state:
    st.session_state.p2_done = 0
if "p2_total" not in st.session_state:
    st.session_state.p2_total = 0
if "phase1_status" not in st.session_state:
    st.session_state.phase1_status = "idle"
if "phase2_status" not in st.session_state:
    st.session_state.phase2_status = "idle"
if "index_error" not in st.session_state:
    st.session_state.index_error = None

def _get_query_params_compat():
    try:
        # Newer Streamlit
        qp = st.query_params
        # Normalize to plain dict with str or list values, flatten single-item lists
        d = dict(qp)
        flat = {}
        for k, v in d.items():
            if isinstance(v, list):
                flat[k] = v[0] if len(v) == 1 else v
            else:
                flat[k] = v
        return flat
    except Exception:
        try:
            qp_old = st.experimental_get_query_params()
            # Flatten single-value lists
            flat = {}
            for k, v in (qp_old or {}).items():
                if isinstance(v, list) and len(v) == 1:
                    flat[k] = v[0]
                else:
                    flat[k] = v
            return flat
        except Exception:
            return {}

def _clear_query_param_compat(key: str):
    try:
        del st.query_params[key]
        return
    except Exception:
        pass
    try:
        qp = st.experimental_get_query_params()
        qp.pop(key, None)
        st.experimental_set_query_params(**qp)
    except Exception:
        pass

def _choose_folder_os():
    """Best-effort OS folder picker with diagnostics.
    Returns (selected_path: str, logs: list[str])
    """
    logs = []
    # macOS via AppleScript
    try:
        logs.append(f"platform={platform.system()}")
        osa = shutil.which("osascript")
        logs.append(f"osascript_path={osa}")
        if platform.system() == "Darwin" and osa:
            script = 'tell application "System Events" to POSIX path of (choose folder with prompt "Select folder to index")'
            proc = subprocess.run([osa, "-e", script], capture_output=True, timeout=30)
            logs.append(f"osascript_rc={proc.returncode}")
            if proc.stdout:
                logs.append(f"osascript_stdout={proc.stdout.decode('utf-8', 'ignore').strip()[:200]}")
            if proc.stderr:
                logs.append(f"osascript_stderr={proc.stderr.decode('utf-8', 'ignore').strip()[:200]}")
            if proc.returncode == 0:
                sel = proc.stdout.decode("utf-8", errors="ignore").strip()
                return sel, logs
    except Exception as exc:
        logs.append(f"osascript_exc={exc}")
    # tkinter fallback executed in a separate subprocess to avoid crashing Streamlit
    try:
        code = r"""
import sys
try:
    import tkinter as tk
    from tkinter import filedialog
    root = tk.Tk()
    root.withdraw()
    try:
        root.attributes('-topmost', True)
    except Exception:
        pass
    path = filedialog.askdirectory()
    try:
        root.destroy()
    except Exception:
        pass
    sys.stdout.write(path or '')
except Exception:
    sys.exit(1)
"""
        proc = subprocess.run([sys.executable, "-c", code], capture_output=True, timeout=30)
        logs.append(f"tk_rc={proc.returncode}")
        if proc.stdout:
            logs.append(f"tk_stdout={proc.stdout.decode('utf-8', 'ignore').strip()[:200]}")
        if proc.stderr:
            logs.append(f"tk_stderr={proc.stderr.decode('utf-8', 'ignore').strip()[:200]}")
        if proc.returncode == 0:
            sel = proc.stdout.decode("utf-8", errors="ignore").strip()
            return sel, logs
    except Exception as exc:
        logs.append(f"tk_exc={exc}")
    return "", logs

# Handle overlay-driven actions via query params:
# - ?info=encoded_path     -> open info overlay for path
# - ?close_info=encoded_path -> close info overlay for path
# - ?add_bm=encoded_path   -> add bookmark
# - ?remove_bm=encoded_path-> remove bookmark
qp_now = _get_query_params_compat()
try:
    if "info" in qp_now:
        val = qp_now["info"]
        path_clicked = urllib.parse.unquote(val if isinstance(val, str) else str(val[0]))
        info_state_key = f"info_state_{path_clicked}"
        st.session_state[info_state_key] = True
        _clear_query_param_compat("info")
    if "close_info" in qp_now:
        val = qp_now["close_info"]
        path_clicked = urllib.parse.unquote(val if isinstance(val, str) else str(val[0]))
        info_state_key = f"info_state_{path_clicked}"
        st.session_state[info_state_key] = False
        _clear_query_param_compat("close_info")
    if "add_bm" in qp_now:
        val = qp_now["add_bm"]
        path_clicked = urllib.parse.unquote(val if isinstance(val, str) else str(val[0]))
        st.session_state.bookmarks[path_clicked] = st.session_state.bookmarks.get(path_clicked, {"tags": [], "added_ts": time.time()})
        save_bookmarks(st.session_state.bookmarks)
        _clear_query_param_compat("add_bm")
    if "remove_bm" in qp_now:
        val = qp_now["remove_bm"]
        path_clicked = urllib.parse.unquote(val if isinstance(val, str) else str(val[0]))
        if path_clicked in st.session_state.bookmarks:
            st.session_state.bookmarks.pop(path_clicked, None)
            save_bookmarks(st.session_state.bookmarks)
        _clear_query_param_compat("remove_bm")
except Exception:
    pass

# Sidebar controls
st.sidebar.header("FORCEPS Indexing Controls")

_inp = st.sidebar.text_input("Folder to index", value=st.session_state.folder_path, key="folder_input")
if _inp and _inp != st.session_state.folder_path:
    st.session_state.folder_path = _inp

with st.sidebar.expander("Folder picker", expanded=False):
    # OS-native picker (best-effort)
    log_area = st.empty()
    if st.button("Use OS file picker…"):
        sel, logs = _choose_folder_os()
        if sel:
            st.session_state.folder_path = sel
            st.session_state.browse_dir = sel
            st.rerun()
        else:
            st.warning("OS file picker unavailable or canceled. Use the navigator below.")
            if logs:
                with log_area:
                    st.code("\n".join(logs), language="text")

    def list_subdirs(p: Path):
        try:
            return [d for d in sorted(p.iterdir()) if d.is_dir()]
        except Exception:
            return []
    cur = Path(st.session_state.browse_dir)
    st.caption(f"Current: {cur}")
    c1, c2 = st.columns([1,1])
    with c1:
        if cur.parent != cur and st.button("Up one level"):
            st.session_state.browse_dir = str(cur.parent)
            st.rerun()
    with c2:
        if st.button("Use this folder"):
            st.session_state.folder_path = str(cur)
            st.rerun()
    subs = list_subdirs(cur)
    names = [d.name for d in subs]
    if names:
        sel = st.selectbox("Open subfolder", names, index=0, key="browse_sel")
        if st.button("Open"):
            st.session_state.browse_dir = str(cur / sel)
            st.rerun()
    else:
        st.caption("No subfolders here.")
folder = st.session_state.folder_path
start_btn = st.sidebar.button("Start Two-Phase Indexing")
stop_btn = st.sidebar.button("Cancel Background Tasks")
save_idx_btn = st.sidebar.button("Save indices to disk")
concurrent_phase2 = st.sidebar.checkbox("Run Phase 2 concurrently", value=False, help="Start captions in parallel with embeddings (may compete for CPU)")

if ollama_installed():
    st.sidebar.success("Ollama: available")
else:
    st.sidebar.error("Ollama: unavailable")

phase1_placeholder = st.sidebar.empty()
phase2_placeholder = st.sidebar.empty()
log_placeholder = st.sidebar.empty()
progress1 = st.sidebar.progress(0)
progress2 = st.sidebar.progress(0)

# Bookmarks / Export section
st.sidebar.markdown("---")
st.sidebar.subheader("Bookmarks / Reports")
bm_count = len(st.session_state.bookmarks or {})
st.sidebar.caption(f"Bookmarked: {bm_count}")
col_csv, col_pdf = st.sidebar.columns(2)
with col_csv:
    try:
        data_csv = generate_bookmarks_csv(st.session_state.bookmarks)
        st.download_button("CSV", data=data_csv, file_name="bookmarks.csv", mime="text/csv")
    except Exception as _:
        st.caption(":warning: CSV export failed")
with col_pdf:
    try:
        data_pdf = generate_bookmarks_pdf(st.session_state.bookmarks)
        st.download_button("PDF", data=data_pdf, file_name="bookmarks.pdf", mime="application/pdf")
    except Exception as _:
        st.caption(":warning: PDF export failed")

# Clear bookmarks/tags actions
st.sidebar.markdown("")
with st.sidebar.expander("Danger zone", expanded=False):
    confirm_clear_tags = st.checkbox("Confirm clear all tags")
    if st.button("Clear all tags"):
        if confirm_clear_tags:
            for _p in list(st.session_state.bookmarks.keys()):
                st.session_state.bookmarks[_p]["tags"] = []
            save_bookmarks(st.session_state.bookmarks)
            st.sidebar.success("All tags cleared")
        else:
            st.sidebar.warning("Please confirm clear all tags")
    confirm_clear_bm = st.checkbox("Confirm clear all bookmarks")
    if st.button("Clear all bookmarks"):
        if confirm_clear_bm:
            st.session_state.bookmarks = {}
            save_bookmarks(st.session_state.bookmarks)
            st.sidebar.success("All bookmarks cleared")
        else:
            st.sidebar.warning("Please confirm clear all bookmarks")

# Filters
st.sidebar.markdown("---")
st.sidebar.subheader("Filters")
exif_availability = st.sidebar.selectbox(
    "EXIF availability",
    ["All", "With EXIF", "Without EXIF"],
    index=0,
    help="Filter results based on whether EXIF metadata is present",
)
def _collect_exif_options(paths):
    makes = set()
    models = set()
    make_to_models = {}
    for pth in paths or []:
        ex = None
        try:
            fp = fingerprint(Path(pth))
            cached = load_cache(fp) or {}
            md = cached.get("metadata", {}) if isinstance(cached, dict) else {}
            ex = md.get("exif") if isinstance(md, dict) else None
            if not ex:
                ex = read_exif(Path(pth))
        except Exception:
            ex = {}
        mk = (ex.get("Make") or "")
        md = (ex.get("Model") or "")
        makes.add(mk)
        models.add(md)
        make_to_models.setdefault(mk, set()).add(md)
    return makes, models, make_to_models

_makes, _models, _make_models = _collect_exif_options(st.session_state.get("last_results", []))

def _labelize(value: str) -> str:
    return value if value else "Unknown"

make_options = ["Any"] + sorted({_labelize(m) for m in _makes})
selected_make_label = st.sidebar.selectbox("Camera make (exact)", make_options, index=0)

if selected_make_label == "Any":
    selected_make_value = None
elif selected_make_label == "Unknown":
    selected_make_value = ""
else:
    selected_make_value = selected_make_label

if selected_make_value is None:
    model_pool = _models
else:
    model_pool = _make_models.get(selected_make_value, set())

model_options = ["Any"] + sorted({_labelize(m) for m in model_pool})
selected_model_label = st.sidebar.selectbox("Camera model (exact)", model_options, index=0)

if selected_model_label == "Any":
    selected_model_value = None
elif selected_model_label == "Unknown":
    selected_model_value = ""
else:
    selected_model_value = selected_model_label

# Filename regex filters
st.sidebar.markdown("")
include_regex_str = st.sidebar.text_input("Filename include regex", value="", help="Show only filenames that match this regex")
exclude_regex_str = st.sidebar.text_input("Filename exclude regex", value="", help="Hide filenames that match this regex")
include_re = None
exclude_re = None
try:
    if include_regex_str:
        include_re = re.compile(include_regex_str)
except Exception:
    st.sidebar.caption(":warning: Invalid include regex")
try:
    if exclude_regex_str:
        exclude_re = re.compile(exclude_regex_str)
except Exception:
    st.sidebar.caption(":warning: Invalid exclude regex")

# File modified time filter
use_mtime_filter = st.sidebar.checkbox("Filter by modified time", value=False)
from_ts = None
to_ts = None
if use_mtime_filter:
    col_from, col_to = st.sidebar.columns(2)
    with col_from:
        fd = st.date_input("From date", value=ddate.today())
        ft = st.time_input("From time", value=dtime(hour=0, minute=0, second=0))
    with col_to:
        td = st.date_input("To date", value=ddate.today())
        tt = st.time_input("To time", value=dtime(hour=23, minute=59, second=59))
    try:
        from_ts = dt.combine(fd, ft).timestamp() if fd and ft else None
        to_ts = dt.combine(td, tt).timestamp() if td and tt else None
    except Exception:
        from_ts = None
        to_ts = None

def scan_images(root, trust_extensions_only: bool = True):
    p = Path(root)
    if not p.exists():
        return []
    # Faster extension-based scan to avoid walking every non-image file
    exts = [".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".webp"]
    candidates = []
    try:
        for ext in exts:
            candidates.extend([str(x) for x in p.rglob(f"*{ext}")])
            up = ext.upper()
            if up != ext:
                candidates.extend([str(x) for x in p.rglob(f"*{up}")])
    except Exception:
        # Fallback to generic scan
        candidates = [str(x) for x in p.rglob("*") if x.is_file()]
    if trust_extensions_only:
        files = candidates
    else:
        # Final filter (slower)
        files = [c for c in candidates if is_image_file(Path(c))]
    # Deduplicate while preserving order
    seen = set()
    unique = []
    for f in files:
        if f not in seen:
            unique.append(f)
            seen.add(f)
    return unique

def phase1_index(image_paths, progress_cb=None):
    n = len(image_paths)
    # Out-of-core: memmap to limit RAM usage
    comb_path = CACHE_DIR / "embeddings_combined.mmap"
    combined_embs = np.memmap(comb_path, dtype="float32", mode="w+", shape=(n, COMBINED_DIM))
    clip_embs = None
    if clip_dim > 0:
        clip_path = CACHE_DIR / "embeddings_clip.mmap"
        clip_embs = np.memmap(clip_path, dtype="float32", mode="w+", shape=(n, clip_dim))
    metadata = [None] * n
    idx_map = {p: i for i, p in enumerate(image_paths)}

    vit_batch = []
    clip_batch = []
    batch_idxs = []
    batch_paths = []

    # Submit tasks in chunks to avoid overwhelming the scheduler on huge datasets
    CHUNK = max(256, BATCH_SIZE * 8)
    processed = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        for off in range(0, n, CHUNK):
            batch_paths_all = image_paths[off: off + CHUNK]
            futures = {ex.submit(preprocess_image_for_vit_clip, p, preprocess_vit, preprocess_clip): p for p in batch_paths_all}
            for fut in as_completed(futures):
                processed += 1
                res = fut.result()
            if res is None:
                if progress_cb: progress_cb(processed, n)
                continue
            path_str, pil_im, vit_tensor, clip_tensor = res
            i = idx_map[path_str]
            fp = fingerprint(Path(path_str))
            cached = load_cache(fp)
            if cached and "embedding_combined" in cached:
                emb_c_np = np.array(cached["embedding_combined"], dtype=np.float32)
                combined_embs[i] = emb_c_np
                emb_clip_np = None
                if clip_embs is not None and cached.get("embedding_clip") is not None:
                    emb_clip_np = np.array(cached["embedding_clip"], dtype=np.float32)
                    clip_embs[i] = emb_clip_np
                metadata[i] = cached.get("metadata", {})
                # Immediately make cached items searchable in partial indices
                try:
                    if st.session_state.faiss_partial is not None:
                        st.session_state.faiss_partial.add_with_ids(emb_c_np.reshape(1, -1), np.array([i], dtype=np.int64))
                    if st.session_state.faiss_clip_partial is not None and emb_clip_np is not None:
                        st.session_state.faiss_clip_partial.add_with_ids(emb_clip_np.reshape(1, -1), np.array([i], dtype=np.int64))
                except Exception:
                    pass
            else:
                vit_batch.append(vit_tensor)
                clip_batch.append(clip_tensor if clip_tensor is not None else vit_tensor)
                batch_idxs.append(i)
                batch_paths.append(path_str)
                metadata[i] = {"exif": read_exif(Path(path_str)), "hashes": compute_perceptual_hashes(Path(path_str))}
            if len(batch_idxs) >= max(1, BATCH_SIZE // 2):
                combined_batch, clip_batch_emb = compute_batch_embeddings(vit_batch, clip_batch, vit_model, clip_model)
                for idx_local, emb_c, emb_clip, pth in zip(batch_idxs, combined_batch, clip_batch_emb if clip_batch_emb is not None else [None]*len(batch_idxs), batch_paths):
                    combined_embs[idx_local] = emb_c
                    if clip_embs is not None and emb_clip is not None:
                        clip_embs[idx_local] = emb_clip
                    save_cache(fingerprint(Path(pth)), {"embedding_combined": emb_c.tolist(), "embedding_clip": (emb_clip.tolist() if emb_clip is not None else None), "metadata": metadata[idx_local]})
                # Add to partial indices with IDs matching idx_local
                try:
                    if st.session_state.faiss_partial is not None:
                        ids = np.array(batch_idxs, dtype=np.int64)
                        st.session_state.faiss_partial.add_with_ids(np.array(combined_batch, dtype=np.float32), ids)
                    if st.session_state.faiss_clip_partial is not None and clip_batch_emb is not None:
                        ids = np.array(batch_idxs, dtype=np.int64)
                        st.session_state.faiss_clip_partial.add_with_ids(np.array(clip_batch_emb, dtype=np.float32), ids)
                except Exception:
                    pass
                vit_batch.clear(); clip_batch.clear(); batch_idxs.clear(); batch_paths.clear()
                if progress_cb: progress_cb(processed, n)
                # yield to event loop to keep Streamlit responsive
                time.sleep(0.0005)

        if batch_idxs:
            combined_batch, clip_batch_emb = compute_batch_embeddings(vit_batch, clip_batch, vit_model, clip_model)
            for idx_local, emb_c, emb_clip, pth in zip(batch_idxs, combined_batch, clip_batch_emb if clip_batch_emb is not None else [None]*len(batch_idxs), batch_paths):
                combined_embs[idx_local] = emb_c
                if clip_embs is not None and emb_clip is not None:
                    clip_embs[idx_local] = emb_clip
                save_cache(fingerprint(Path(pth)), {"embedding_combined": emb_c.tolist(), "embedding_clip": (emb_clip.tolist() if emb_clip is not None else None), "metadata": metadata[idx_local]})
            # flush to partial indices as well
            try:
                if st.session_state.faiss_partial is not None:
                    ids = np.array(batch_idxs, dtype=np.int64)
                    st.session_state.faiss_partial.add_with_ids(np.array(combined_batch, dtype=np.float32), ids)
                if st.session_state.faiss_clip_partial is not None and clip_batch_emb is not None:
                    ids = np.array(batch_idxs, dtype=np.int64)
                    st.session_state.faiss_clip_partial.add_with_ids(np.array(clip_batch_emb, dtype=np.float32), ids)
            except Exception:
                pass
            if progress_cb: progress_cb(processed, n)

    # Build FAISS IVF-PQ index (optionally with PCA) for scale
    faiss.omp_set_num_threads(1)
    xb = np.asarray(combined_embs)
    d = xb.shape[1]
    index_comb = None
    pca_ret = None
    # Effective training/sample sizes and cluster counts
    num_train = min(n, TRAIN_SAMPLES)
    # Heuristic: nlist ~ 4 * sqrt(N), capped by available training points
    nlist_eff = max(1, int(np.sqrt(max(1, n))) * 4)
    nlist_eff = min(IVF_NLIST, nlist_eff, max(1, num_train))
    # Choose PQ_M that divides dimension to avoid FAISS assertions
    def choose_m(dim: int, m_pref: int) -> int:
        candidates = [m for m in [m_pref, 64, 48, 32, 24, 16, 12, 8, 6, 4, 3, 2, 1] if m <= dim and dim % m == 0]
        return candidates[0] if candidates else 1
    if USE_PCA:
        eff_pca_dim = max(1, min(PCA_DIM, d, num_train))
        pca_mat = faiss.PCAMatrix(d, eff_pca_dim)
        pca_mat.train(xb[: num_train])
        pca_ret = pca_mat
        d_red = eff_pca_dim
        quantizer = faiss.IndexFlatL2(d_red)
        pq_m_eff = choose_m(d_red, PQ_M)
        index_comb = faiss.IndexIVFPQ(quantizer, d_red, nlist_eff, pq_m_eff, 8)
        # Train
        xb_train = pca_mat.apply_py(xb[: num_train])
        index_comb.train(xb_train)
        # Add in batches
        for off in range(0, n, ADD_BATCH):
            x = pca_mat.apply_py(xb[off : off + ADD_BATCH])
            index_comb.add(x)
    else:
        quantizer = faiss.IndexFlatL2(d)
        pq_m_eff = choose_m(d, PQ_M)
        index_comb = faiss.IndexIVFPQ(quantizer, d, nlist_eff, pq_m_eff, 8)
        index_comb.train(xb[: num_train])
        for off in range(0, n, ADD_BATCH):
            x = xb[off : off + ADD_BATCH]
            index_comb.add(x)
    index_comb.nprobe = min(64, max(1, nlist_eff // 16))

    idx_clip = None
    if clip_embs is not None:
        xc = np.asarray(clip_embs)
        dc = xc.shape[1]
        quantizer_c = faiss.IndexFlatL2(dc)
        num_train_c = min(n, TRAIN_SAMPLES)
        nlist_c = min(nlist_eff, max(1, num_train_c))
        pq_m_c = choose_m(dc, max(8, PQ_M // 2))
        idx_clip = faiss.IndexIVFPQ(quantizer_c, dc, nlist_c, pq_m_c, 8)
        idx_clip.train(xc[: num_train_c])
        for off in range(0, n, ADD_BATCH):
            x = xc[off : off + ADD_BATCH]
            idx_clip.add(x)
        idx_clip.nprobe = min(64, max(1, nlist_c // 16))

    # Flush memmaps
    try:
        combined_embs.flush()
        if clip_embs is not None:
            clip_embs.flush()
    except Exception:
        pass
    faiss.omp_set_num_threads(max(1, MAX_WORKERS))
    # After full index built, discard partial indices to save RAM
    try:
        st.session_state.faiss_partial = None
        st.session_state.faiss_clip_partial = None
    except Exception:
        pass
    return index_comb, idx_clip, combined_embs, clip_embs, metadata, pca_ret

def phase2_caption(image_paths, progress_cb=None):
    if not ollama_installed() or not model_available("llava"):
        for i in range(len(image_paths)):
            if progress_cb: progress_cb(i+1, len(image_paths))
        return
    for i, p in enumerate(image_paths):
        fp = fingerprint(Path(p))
        cached = load_cache(fp) or {}
        meta = cached.get("metadata", {})
        if meta.get("caption"):
            if progress_cb: progress_cb(i+1, len(image_paths)); continue
        cap = generate_caption_ollama(p)
        if cap:
            meta["caption"] = cap
            cached["metadata"] = meta
            save_cache(fp, cached)
        if progress_cb: progress_cb(i+1, len(image_paths))

# ---- Incremental watcher ----
def _incremental_add(new_paths):
    if not new_paths:
        return 0
    added = 0
    vit_batch = []
    clip_batch = []
    batch_paths = []
    for p in new_paths:
        res = preprocess_image_for_vit_clip(p, preprocess_vit, preprocess_clip)
        if res is None:
            continue
        path_str, _pil, vit_t, clip_t = res
        vit_batch.append(vit_t)
        clip_batch.append(clip_t if clip_t is not None else vit_t)
        batch_paths.append(path_str)
        if len(batch_paths) >= BATCH_SIZE:
            _flush_incremental_batch(vit_batch, clip_batch, batch_paths)
            added += len(batch_paths)
            vit_batch.clear(); clip_batch.clear(); batch_paths.clear()
    if batch_paths:
        _flush_incremental_batch(vit_batch, clip_batch, batch_paths)
        added += len(batch_paths)
        vit_batch.clear(); clip_batch.clear(); batch_paths.clear()
    return added

def _flush_incremental_batch(vit_batch, clip_batch, batch_paths):
    comb, clip_emb = compute_batch_embeddings(vit_batch, clip_batch, vit_model, clip_model)
    # Save caches and add to indices
    to_add_comb = comb.astype(np.float32)
    if USE_PCA and st.session_state.pca_obj is not None:
        try:
            to_add_comb = st.session_state.pca_obj.apply_py(to_add_comb)
        except Exception:
            pass
    with st.session_state.index_lock:
        if st.session_state.faiss_combined is not None:
            st.session_state.faiss_combined.add(to_add_comb)
        if st.session_state.faiss_clip is not None and clip_emb is not None:
            st.session_state.faiss_clip.add(clip_emb.astype(np.float32))
        # Extend known paths in the same order as additions
        st.session_state.index_paths.extend(batch_paths)
    for emb_c, emb_clip, pth in zip(comb, clip_emb if clip_emb is not None else [None]*len(batch_paths), batch_paths):
        save_cache(fingerprint(Path(pth)), {
            "embedding_combined": emb_c.tolist(),
            "embedding_clip": (emb_clip.tolist() if emb_clip is not None else None),
            "metadata": {"exif": read_exif(Path(pth)), "hashes": compute_perceptual_hashes(Path(pth))}
        })

def _folder_watcher(root_folder: str):
    prev = set(st.session_state.index_paths)
    while True:
        try:
            current = set(scan_images(root_folder))
            new_paths = sorted(list(current - set(st.session_state.index_paths)))
            if new_paths and st.session_state.faiss_combined is not None:
                _incremental_add(new_paths)
        except Exception:
            pass
        time.sleep(WATCH_INTERVAL)

def start_watcher(folder: str):
    if st.session_state.watcher_running:
        return
    st.session_state.watcher_running = True
    th = threading.Thread(target=_folder_watcher, args=(folder,), daemon=True)
    th.start()

# ---- Stats panel ----
def _index_stats(idx):
    if idx is None:
        return {}
    stats = {"ntotal": getattr(idx, "ntotal", 0)}
    try:
        # IVFPQ specifics
        stats["nlist"] = getattr(idx, "nlist", None)
        stats["pq_m"] = getattr(getattr(idx, "pq", None), "M", None) or getattr(idx, "pq_m", None)
    except Exception:
        pass
    return stats

# Start two-phase indexing (reverted to stable behavior)
if start_btn:
    image_paths = scan_images(folder)
    if not image_paths:
        st.error("No images found in folder.")
    else:
        st.session_state.index_paths = image_paths
        progress1.progress(0)
        progress2.progress(0)
        phase1_placeholder.text("Phase 1: starting…")
        phase2_placeholder.text("")

        def p1_cb(done, total):
            pct = int(done/total*100) if total else 0
            progress1.progress(pct)
            phase1_placeholder.text(f"Phase 1: embeddings {pct}% ({done}/{total})")
            if done % max(1, total//20 or 1) == 0:
                log_placeholder.write(f"Phase 1 progress: {done}/{total}")

        def p2_cb(done, total):
            pct = int(done/total*100) if total else 0
            progress2.progress(pct)
            phase2_placeholder.text(f"Phase 2: captions {pct}% ({done}/{total})")
            if done % max(1, total//20 or 1) == 0:
                log_placeholder.write(f"Phase 2 progress: {done}/{total}")

        if concurrent_phase2 and ollama_installed() and model_available("llava"):
            def run_p1():
                with st.spinner("Indexing images (Phase 1)…"):
                    try:
                        idx_comb, idx_clip, comb_embs, clip_embs, meta, pca_obj = phase1_index(image_paths, progress_cb=p1_cb)
                        st.session_state.faiss_combined = idx_comb
                        st.session_state.faiss_clip = idx_clip
                        st.session_state.pca_obj = pca_obj
                        st.session_state.phase1_ready = True
                        phase1_placeholder.text("Phase 1: complete")
                        start_watcher(folder)
                    except Exception as exc:
                        st.sidebar.error(f"Phase 1 error: {exc}")
            def run_p2():
                with st.spinner("Generating captions (Phase 2)…"):
                    try:
                        phase2_caption(image_paths, progress_cb=p2_cb)
                        st.session_state.phase2_ready = True
                        phase2_placeholder.text("Phase 2: complete")
                    except Exception as exc:
                        st.sidebar.error(f"Phase 2 error: {exc}")
            threading.Thread(target=run_p1, daemon=True).start()
            threading.Thread(target=run_p2, daemon=True).start()
        else:
            # Run sequentially for predictable resource usage
            with st.spinner("Indexing images (Phase 1)…"):
                try:
                    idx_comb, idx_clip, comb_embs, clip_embs, meta, pca_obj = phase1_index(image_paths, progress_cb=p1_cb)
                    st.session_state.faiss_combined = idx_comb
                    st.session_state.faiss_clip = idx_clip
                    st.session_state.pca_obj = pca_obj
                    st.session_state.phase1_ready = True
                    phase1_placeholder.text("Phase 1: complete")
                    start_watcher(folder)
                except Exception as exc:
                    st.sidebar.error(f"Phase 1 error: {exc}")
            with st.spinner("Generating captions (Phase 2)…"):
                try:
                    phase2_caption(image_paths, progress_cb=p2_cb)
                    st.session_state.phase2_ready = True
                    phase2_placeholder.text("Phase 2: complete")
                except Exception as exc:
                    st.sidebar.error(f"Phase 2 error: {exc}")

if save_idx_btn:
    if st.session_state.faiss_combined is not None:
        faiss.write_index(st.session_state.faiss_combined, "faiss_combined.index")
        if st.session_state.faiss_clip is not None:
            faiss.write_index(st.session_state.faiss_clip, "faiss_clip.index")
        st.success("Saved FAISS indices.")
    else:
        st.error("No index to save yet.")

# Search UI
st.header("Search")
col1, col2 = st.columns([3,1])
with col1:
    nl_query = st.text_input("Natural language query (optional)")
    tag_query = st.text_input("Filter by tag (optional)", help="Show only images whose tags include this exact term")
    uploaded = st.file_uploader("Or upload an image for search", type=["jpg","png","jpeg","bmp","gif"])
    run_search = st.button("Run Search")
with col2:
    st.markdown("**Index readiness**")
    st.write(f"- Phase 1 (embeddings): {'✅' if st.session_state.phase1_ready else '❌'}")
    st.write(f"- Phase 2 (captions): {'✅' if st.session_state.phase2_ready else '❌'}")
    st.write(f"- Ollama: {'available' if ollama_installed() else 'not available'}")
    if st.session_state.indexing_running:
        p1_pct = int((st.session_state.p1_done / st.session_state.p1_total) * 100) if st.session_state.p1_total else 0
        p2_pct = int((st.session_state.p2_done / st.session_state.p2_total) * 100) if st.session_state.p2_total else 0
        progress1.progress(p1_pct)
        progress2.progress(p2_pct)
        phase1_placeholder.text(f"Phase 1: embeddings {p1_pct}% ({st.session_state.p1_done}/{st.session_state.p1_total})")
        phase2_placeholder.text(f"Phase 2: captions {p2_pct}% ({st.session_state.p2_done}/{st.session_state.p2_total})")
    if st.session_state.index_error:
        st.error(st.session_state.index_error)
    # Stats after indexing
    if st.session_state.faiss_combined is not None:
        s = _index_stats(st.session_state.faiss_combined)
        st.caption(f"Combined index: ntotal={s.get('ntotal',0)}, nlist={s.get('nlist','-')}, pq_m={s.get('pq_m','-')}")
    if st.session_state.faiss_clip is not None:
        s = _index_stats(st.session_state.faiss_clip)
        st.caption(f"CLIP index: ntotal={s.get('ntotal',0)}, nlist={s.get('nlist','-')}, pq_m={s.get('pq_m','-')}")
    # Timing
    if st.session_state.phase1_duration is not None:
        st.caption(f"Phase 1 duration: {st.session_state.phase1_duration:.1f}s")
    if st.session_state.phase2_duration is not None:
        st.caption(f"Phase 2 duration: {st.session_state.phase2_duration:.1f}s")
    if st.session_state.total_index_duration is not None:
        st.caption(f"Total duration: {st.session_state.total_index_duration:.1f}s")

results = st.session_state.get("last_results", [])
if run_search:
    # start fresh for each explicit search action
    results = []
    if nl_query and (st.session_state.faiss_clip is not None or st.session_state.faiss_clip_partial is not None):
        q_emb = compute_clip_text_embedding(nl_query, clip_model)
        idx_to_use = st.session_state.faiss_clip or st.session_state.faiss_clip_partial
        D, I = idx_to_use.search(np.array([q_emb], dtype=np.float32), 20)
        for idx in I[0]:
            if idx == -1: continue
            results.append(st.session_state.index_paths[idx])
    elif nl_query and st.session_state.faiss_clip is None:
        st.warning("Natural language search requires CLIP index. Please run indexing with CLIP available.")
    elif uploaded:
        tmp = Image.open(uploaded).convert("RGB")
        vit_t = preprocess_vit(tmp).unsqueeze(0)
        clip_t = (preprocess_clip(tmp).unsqueeze(0) if preprocess_clip is not None else None)
        if clip_model is not None and clip_t is not None:
            q_comb, q_clip = compute_batch_embeddings([vit_t.squeeze(0)], [clip_t.squeeze(0)], vit_model, clip_model)
        else:
            q_comb, _ = compute_batch_embeddings([vit_t.squeeze(0)], [], vit_model, None)
        q_emb = q_comb[0].astype(np.float32)
        idx_to_use = st.session_state.faiss_combined or st.session_state.faiss_partial
        if idx_to_use is not None:
            # If PCA was used, project query too
            if USE_PCA and st.session_state.get("pca_obj") is not None:
                try:
                    q_proj = st.session_state.pca_obj.apply_py(np.array([q_emb], dtype=np.float32))
                    D, I = idx_to_use.search(q_proj.astype(np.float32), 20)
                except Exception:
                    D, I = idx_to_use.search(np.array([q_emb], dtype=np.float32), 20)
            else:
                D, I = idx_to_use.search(np.array([q_emb], dtype=np.float32), 20)
            for idx in I[0]:
                if idx == -1: continue
                results.append(st.session_state.index_paths[idx])
    # Persist computed results
    st.session_state.last_results = results

# Apply EXIF filters to current results for display
def _exif_for_path(path_str: str):
    try:
        p = Path(path_str)
        fp = fingerprint(p)
        cached = load_cache(fp) or {}
        md = cached.get("metadata", {}) if isinstance(cached, dict) else {}
        exif = md.get("exif") if isinstance(md, dict) else None
        if not exif:
            exif = read_exif(p)
        return exif or {}
    except Exception:
        return {}

display_results = []
for r in results:
    exif = _exif_for_path(r)
    has_exif = bool(exif)
    if exif_availability == "With EXIF" and not has_exif:
        continue
    if exif_availability == "Without EXIF" and has_exif:
        continue
    mk_raw = (exif.get("Make") or "")
    md_raw = (exif.get("Model") or "")
    if selected_make_value is not None and mk_raw != selected_make_value:
        continue
    if selected_model_value is not None and md_raw != selected_model_value:
        continue
    # Filename regex include/exclude
    fname = Path(r).name
    if include_re and not include_re.search(fname):
        continue
    if exclude_re and exclude_re.search(fname):
        continue
    # Modified time filter
    if use_mtime_filter:
        try:
            mtime = Path(r).stat().st_mtime
            if from_ts is not None and mtime < from_ts:
                continue
            if to_ts is not None and mtime > to_ts:
                continue
        except Exception:
            continue
    # Tag filter (exact term)
    if tag_query:
        tags = (st.session_state.bookmarks.get(r, {}).get("tags") if r in st.session_state.bookmarks else []) or []
        if tag_query not in tags:
            continue
    display_results.append(r)

if not results:
    st.info("Type a natural-language query (requires CLIP) or upload an image.")
elif not display_results:
    st.warning("No results match current filters.")

# ----- Tags page: overview of tags and images -----
st.markdown("---")
st.subheader("Tags Overview")
all_tags = {}
for path_str, meta in (st.session_state.bookmarks or {}).items():
    for t in (meta.get("tags") or []):
        all_tags.setdefault(t, []).append(path_str)

if not all_tags:
    st.caption("No tags yet.")
else:
    # tag selection and display
    tag_list = sorted(all_tags.keys())
    sel_tag = st.selectbox("Select a tag to view images", ["(All)"] + tag_list, index=0)
    cols = st.columns(5)
    show_paths = []
    if sel_tag == "(All)":
        # Flatten unique paths preserving order by tag buckets
        seen = set()
        for t in tag_list:
            for p in all_tags[t]:
                if p not in seen:
                    show_paths.append(p)
                    seen.add(p)
    else:
        show_paths = all_tags.get(sel_tag, [])

    for i, r in enumerate(show_paths[:50]):
        try:
            # reuse base64 render to avoid path issues
            try:
                mime, _ = mimetypes.guess_type(r)
                mime = mime or "image/jpeg"
                with open(r, "rb") as _f:
                    b64 = base64.b64encode(_f.read()).decode("utf-8")
                data_url = f"data:{mime};base64,{b64}"
            except Exception:
                data_url = ""
            slot = cols[i % 5]
            slot.markdown(f"<div class='img-card'><img src='{data_url}' /></div>", unsafe_allow_html=True)
            slot.caption(Path(r).name)
        except Exception:
            cols[i % 5].write(Path(r).name)

if display_results:
    st.markdown("### Top results")
    cols = st.columns(5)
    for i, r in enumerate(display_results[:25]):
        try:
            tile = cols[i%5]
            # Image container with overlay layers (use data URL to avoid broken local paths)
            encoded = urllib.parse.quote(r)
            try:
                mime, _ = mimetypes.guess_type(r)
                mime = mime or "image/jpeg"
                with open(r, "rb") as _f:
                    b64 = base64.b64encode(_f.read()).decode("utf-8")
                data_url = f"data:{mime};base64,{b64}"
            except Exception:
                data_url = ""
            tile.markdown(f"""
            <div class='img-card'>
              <img src='{data_url}' />
            </div>
            """, unsafe_allow_html=True)
            tile.caption(Path(r).name)
            meta = load_cache(fingerprint(Path(r)))
            if meta and meta.get("metadata", {}).get("caption"):
                tile.write(meta["metadata"]["caption"][:140])

            # --- Bookmark/Tags controls ---
            def _is_bm(pth: str) -> bool:
                return pth in (st.session_state.bookmarks or {})
            def _add_bm(pth: str):
                if pth not in st.session_state.bookmarks:
                    st.session_state.bookmarks[pth] = {"tags": [], "added_ts": time.time()}
                    save_bookmarks(st.session_state.bookmarks)
            def _remove_bm(pth: str):
                if pth in st.session_state.bookmarks:
                    st.session_state.bookmarks.pop(pth, None)
                    save_bookmarks(st.session_state.bookmarks)
            def _update_tags(pth: str, tags_str: str):
                tags = [t.strip() for t in (tags_str or "").replace(";", ",").split(",") if t.strip()]
                if pth not in st.session_state.bookmarks:
                    st.session_state.bookmarks[pth] = {"tags": tags, "added_ts": time.time()}
                else:
                    st.session_state.bookmarks[pth]["tags"] = tags
                save_bookmarks(st.session_state.bookmarks)

            tg_key = f"bm_tags_{r}"
            is_bm_now = _is_bm(r)
            # Overlay bookmark icon (visual)
            toggle_param = "add_bm" if not is_bm_now else "remove_bm"
            class_suffix = " active" if is_bm_now else ""
            star_text = "★" if is_bm_now else "☆"
            tile.markdown(
                f"<div class='bm-overlay'><span class='bm-icon{class_suffix}'>{star_text}</span></div>",
                unsafe_allow_html=True,
            )
            # Reliable Streamlit toggle button
            bm_btn_key = f"bm_btn_{r}"
            if tile.button(("Unbookmark ★" if is_bm_now else "Bookmark ☆"), key=bm_btn_key):
                if is_bm_now:
                    _remove_bm(r)
                    is_bm_now = False
                else:
                    _add_bm(r)
                    is_bm_now = True
            if is_bm_now:
                existing_tags = ", ".join(st.session_state.bookmarks.get(r, {}).get("tags", []))
                new_tags = tile.text_input("Add/edit tags", value=existing_tags, key=tg_key, help="Comma or semicolon separated tags")
                if new_tags != existing_tags:
                    _update_tags(r, new_tags)
                chips = st.session_state.bookmarks.get(r, {}).get("tags", [])
                if chips:
                    chip_html = " ".join([f"<span class='tag-chip'>{c}</span>" for c in chips])
                    tile.markdown(chip_html, unsafe_allow_html=True)

            # --- Info overlay rendering remains, but no explicit button ---
            info_state_key = f"info_state_{r}"
            if st.session_state.get(info_state_key, False):
                details = _gather_metadata_for_path(Path(r))
                # Shield against HTML injection in strings
                def esc(x):
                    try:
                        return html.escape(str(x))
                    except Exception:
                        return str(x)
                path_s = esc(details.get('path',''))
                make_s = esc(details.get('make',''))
                model_s = esc(details.get('model',''))
                dt_s = esc(details.get('datetime',''))
                ph = esc(details.get('phash',''))
                ah = esc(details.get('ahash',''))
                dh = esc(details.get('dhash',''))
                cap = details.get('caption','')

                # Structured table rendering
                html_parts = [
                    "<div class='info-panel'>",
                    "  <div class='panel-bg'></div>",
                    "  <div class='info-box'>",
                    f"    <a class='info-close' href='?close_info={encoded}'>×</a>",
                    "    <h4>EXIF & Metadata</h4>",
                    "    <table class='meta-table'>",
                    f"      <tr><th>Path</th><td>{path_s}</td></tr>",
                    f"      <tr><th>Make / Model</th><td>{make_s} <span class='code-badge'>/</span> {model_s}</td></tr>",
                    f"      <tr><th>Date / Time</th><td>{dt_s}</td></tr>",
                    f"      <tr><th>Hashes</th><td><span class='code-badge'>p:{ph}</span><span class='code-badge'>a:{ah}</span><span class='code-badge'>d:{dh}</span></td></tr>",
                    "    </table>",
                ]
                if cap:
                    html_parts.append("    <div class='caption-block'>" + esc(cap) + "</div>")
                html_parts.append("  </div>")
                html_parts.append("</div>")
                tile.markdown("\n".join(html_parts), unsafe_allow_html=True)
        except Exception:
            cols[i%5].write(Path(r).name)

st.sidebar.markdown("---")
if st.sidebar.button("Exit App"):
    st.stop()
