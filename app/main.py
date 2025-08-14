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
import redis
import json
import pickle
from whoosh.index import open_dir as open_whoosh_dir
from whoosh.qparser import QueryParser
from sklearn.cluster import MiniBatchKMeans
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
    st.session_state.bookmarks = {}
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

# Indexing logic is now handled by the standalone engine scripts.
# The UI will now act as a controller for the distributed backend.

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
    st.info("Starting background process to enqueue indexing jobs...")
    st.info("Ensure your workers (`engine.py`) and index builder (`build_index.py`) are running.")

    enqueue_script_path = Path(__file__).parent / "enqueue_jobs.py"
    command = [
        sys.executable,
        str(enqueue_script_path),
        "--input_dir",
        folder
    ]

    try:
        # Launch the enqueuer script as a background process
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        st.session_state['enqueuer_process'] = process.pid
        st.success(f"Successfully launched job enqueuer (PID: {process.pid}).")
        st.code(" ".join(command))
        # We don't wait for it to finish. The UI can now monitor Redis.
    except Exception as e:
        st.error(f"Failed to launch enqueuer script: {e}")

if save_idx_btn:
    if st.session_state.faiss_combined is not None:
        faiss.write_index(st.session_state.faiss_combined, "faiss_combined.index")
        if st.session_state.faiss_clip is not None:
            faiss.write_index(st.session_state.faiss_clip, "faiss_clip.index")
        st.success("Saved FAISS indices.")
    else:
        st.error("No index to save yet.")

# Main app layout
search_tab, reporting_tab = st.tabs(["Search & Results", "Reporting"])

with search_tab:
    st.header("Backend Monitoring")
    if st.button("Refresh Queue Stats"):
        try:
            r = redis.Redis(host=st.session_state.redis_host, port=st.session_state.redis_port, db=0)
            jobs_in_queue = r.llen("forceps:job_queue")
            results_in_queue = r.llen("forceps:results_queue")

            c1, c2 = st.columns(2)
            with c1:
                st.metric("Jobs in Queue", f"{jobs_in_queue:,}")
            with c2:
                st.metric("Results in Queue", f"{results_in_queue:,}")

        except Exception as e:
            st.error(f"Could not connect to Redis: {e}")

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
        results = []
        vector_results = set()
        keyword_results = set()

        # 1. Vector Search (FAISS)
        if nl_query and st.session_state.get("faiss_clip"):
            q_emb = compute_clip_text_embedding(nl_query, clip_model)
            _, I = st.session_state.faiss_clip.search(np.array([q_emb], dtype=np.float32), 100)
            vector_results.update([st.session_state.index_paths[i] for i in I[0] if i != -1])
        elif uploaded:
            tmp = Image.open(uploaded).convert("RGB")
            vit_t = preprocess_vit(tmp).unsqueeze(0)
            clip_t = (preprocess_clip(tmp).unsqueeze(0) if preprocess_clip is not None else None)
            q_comb, _ = compute_batch_embeddings([vit_t.squeeze(0)], [clip_t.squeeze(0)] if clip_t is not None else [], vit_model, clip_model)
            q_emb = q_comb[0].astype(np.float32)

            if st.session_state.get("faiss_combined"):
                idx_to_use = st.session_state.faiss_combined
                if st.session_state.get("pca_obj"):
                    q_emb = st.session_state.pca_obj.apply_py(np.array([q_emb]))[0]

                _, I = idx_to_use.search(np.array([q_emb], dtype=np.float32), 100)
                vector_results.update([st.session_state.index_paths[i] for i in I[0] if i != -1])

        # 2. Keyword Search (Whoosh)
        if nl_query and st.session_state.get("whoosh_searcher"):
            searcher = st.session_state.whoosh_searcher
            parser = QueryParser("content", schema=searcher.schema)
            query = parser.parse(nl_query)
            keyword_hits = searcher.search(query, limit=200)
            keyword_results.update([hit['path'] for hit in keyword_hits])

        # 3. Combine and Re-rank Results
        # Simple re-ranking: boost items that appear in both result sets.
        final_results = []
        intersection = vector_results.intersection(keyword_results)

        # Add boosted intersection results first
        final_results.extend(list(intersection))

        # Add remaining vector results
        final_results.extend([p for p in vector_results if p not in intersection])

        # Add remaining keyword results
        final_results.extend([p for p in keyword_results if p not in intersection])

        st.session_state.last_results = final_results

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

    if st.session_state.get("last_results") and st.session_state.get("embeddings") is not None:
        st.markdown("---")
        st.subheader("Cluster Analysis")

        num_clusters = st.slider("Number of Clusters", min_value=2, max_value=50, value=10)

        if st.button("Cluster Displayed Results"):
            with st.spinner("Clustering results..."):
                paths_to_cluster = st.session_state.last_results
                path_to_idx = {path: i for i, path in enumerate(st.session_state.index_paths)}
                result_indices = [path_to_idx[p] for p in paths_to_cluster if p in path_to_idx]

                if result_indices:
                    result_embeddings = st.session_state.embeddings[result_indices]

                    kmeans = MiniBatchKMeans(n_clusters=num_clusters, random_state=0, batch_size=256, n_init='auto')
                    kmeans.fit(result_embeddings)

                    st.session_state.cluster_labels = {path: label for path, label in zip(paths_to_cluster, kmeans.labels_)}
                    st.success(f"Successfully clustered results into {num_clusters} groups.")
                    st.rerun()
                else:
                    st.warning("No results to cluster.")

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

    def display_image_tile(tile_container, image_path):
        """Renders a single image tile with all its controls."""
        r = image_path
        try:
            # --- Image Display ---
            encoded = urllib.parse.quote(r)
            try:
                mime, _ = mimetypes.guess_type(r)
                mime = mime or "image/jpeg"
                with open(r, "rb") as _f:
                    b64 = base64.b64encode(_f.read()).decode("utf-8")
                data_url = f"data:{mime};base64,{b64}"
            except Exception:
                data_url = ""
            tile_container.markdown(f"<div class='img-card'><img src='{data_url}' /></div>", unsafe_allow_html=True)
            tile_container.caption(Path(r).name)

            # --- Details Expander ---
            if 'manifest' in st.session_state and r in st.session_state.manifest:
                with tile_container.expander("Show Details"):
                    manifest_item = st.session_state.manifest[r]
                    cached_meta = load_cache(fingerprint(Path(r))) or {}

                    st.code(manifest_item['path'], language='text')

                    st.markdown("**Hashes**")
                    hashes = manifest_item.get('hashes', {})
                    if hashes:
                        for hash_name, hash_value in hashes.items():
                            st.text(f"{hash_name.upper()}: {hash_value}")

                    st.markdown("**EXIF Data**")
                    exif_data = cached_meta.get("metadata", {}).get("exif", {})
                    if exif_data:
                        for key, value in exif_data.items():
                            st.text(f"{key}: {value}")

                    st.markdown("**AI Caption**")
                    caption = cached_meta.get("metadata", {}).get("caption")
                    st.text(caption or "Not available.")

            # --- Bookmark/Tags Expander ---
            def _is_bm(pth: str) -> bool:
                return pth in (st.session_state.get("bookmarks", {}) or {})

            def _save_bms():
                case_dir = Path(st.session_state.cases_dir) / st.session_state.selected_case
                save_bookmarks(st.session_state.bookmarks, case_dir)

            with tile_container.expander("Manage Bookmark"):
                is_bookmarked = _is_bm(r)

                if st.checkbox("Bookmarked", value=is_bookmarked, key=f"bm_check_{r}"):
                    if not is_bookmarked:
                        st.session_state.bookmarks[r] = st.session_state.bookmarks.get(r, {"tags": [], "notes": "", "added_ts": time.time()})
                        _save_bms()
                        st.rerun()

                    bookmark_data = st.session_state.bookmarks.get(r, {})

                    existing_tags = "\n".join(bookmark_data.get("tags", []))
                    new_tags_str = st.text_area("Tags (one per line)", value=existing_tags, key=f"tags_{r}")

                    existing_notes = bookmark_data.get("notes", "")
                    new_notes = st.text_area("Notes", value=existing_notes, key=f"notes_{r}")

                    if st.button("Save Bookmark Details", key=f"save_bm_{r}"):
                        st.session_state.bookmarks[r]['tags'] = [t.strip() for t in new_tags_str.split("\n") if t.strip()]
                        st.session_state.bookmarks[r]['notes'] = new_notes
                        _save_bms()
                        st.success("Bookmark updated!")

                elif is_bookmarked:
                    st.session_state.bookmarks.pop(r, None)
                    _save_bms()
                    st.rerun()
        except Exception as e:
            tile_container.error(f"Error: {e}")
            tile_container.write(Path(r).name)


    if display_results:
        st.markdown("### Top results")

        # If results have been clustered, group them
        if 'cluster_labels' in st.session_state:
            clusters = {}
            for path in display_results:
                label = st.session_state.cluster_labels.get(path)
                if label is not None:
                    if label not in clusters:
                        clusters[label] = []
                    clusters[label].append(path)

            if st.button("Clear Clustering"):
                del st.session_state['cluster_labels']
                st.rerun()

            for label, paths in sorted(clusters.items()):
                st.markdown(f"--- \n#### Cluster {label+1} ({len(paths)} images)")
                cols = st.columns(5)
                for i, r in enumerate(paths):
                    display_image_tile(cols[i%5], r)
        else:
            # Original, unclustered display
            cols = st.columns(5)
            for i, r in enumerate(display_results[:25]):
                display_image_tile(cols[i%5], r)

with reporting_tab:
    st.header("Case Report Generator")
    st.info("This section allows you to review bookmarked items and generate a final PDF report.")

    if not st.session_state.get("bookmarks"):
        st.warning("No bookmarks in the current case to report.")
    else:
        st.subheader("Bookmarked Items")

        # Display a table of bookmarked items
        for path, data in st.session_state.bookmarks.items():
            with st.expander(f"{Path(path).name}"):
                st.image(path, width=150)
                st.text(f"Path: {path}")
                st.text(f"Tags: {', '.join(data.get('tags', []))}")
                st.text_area("Notes", value=data.get('notes', ''), height=100, disabled=True, key=f"notes_report_{path}")

        # Generate report button
        st.subheader("Generate Report")
        if st.button("Generate PDF Report"):
            with st.spinner("Generating PDF..."):
                # The PDF generation function is already updated to handle the new data structure
                pdf_data = generate_bookmarks_pdf(st.session_state.bookmarks)
                st.download_button(
                    label="Download PDF Report",
                    data=pdf_data,
                    file_name=f"case_report_{st.session_state.get('selected_case', 'current')}.pdf",
                    mime="application/pdf"
                )
cases_dir_path = st.sidebar.text_input("Cases Directory", value="output_index")
st.session_state.cases_dir = cases_dir_path

case_options = []
try:
    p = Path(cases_dir_path)
    if p.is_dir():
        case_options = sorted([d.name for d in p.iterdir() if d.is_dir()], reverse=True)
except Exception as e:
    st.sidebar.error(f"Error scanning cases: {e}")

if not case_options:
    st.sidebar.caption("No cases found in directory.")
else:
    selected_case = st.sidebar.selectbox("Select Case to Load", case_options)
    st.session_state.selected_case = selected_case

    if st.sidebar.button("Load Selected Case"):
        index_dir = Path(st.session_state.cases_dir) / st.session_state.selected_case
        st.sidebar.info(f"Loading from: {index_dir}")
        try:
            if not index_dir.is_dir():
                st.error(f"Case directory not found: {index_dir}")
            else:
                # Load main index
                combined_idx_path = index_dir / "combined.index"
                if combined_idx_path.exists():
                    st.session_state.faiss_combined = faiss.read_index(str(combined_idx_path))
                    st.success("Loaded combined index.")
                else:
                     st.error("combined.index not found in directory.")

                # Load manifest
                manifest_path = index_dir / "manifest.json"
                if manifest_path.exists():
                    with open(manifest_path, "r") as f:
                        manifest_data = json.load(f)
                    # Create a map for easy lookup
                    st.session_state.manifest = {item['path']: item for item in manifest_data}
                    # Keep the ordered list of paths for indexing
                    st.session_state.index_paths = [item['path'] for item in manifest_data]
                    st.success(f"Loaded manifest for {len(st.session_state.index_paths)} images.")
                else:
                    st.error("manifest.json not found in directory.")

                # Load optional clip index
                clip_idx_path = index_dir / "clip.index"
                if clip_idx_path.exists():
                    st.session_state.faiss_clip = faiss.read_index(str(clip_idx_path))
                    st.success("Loaded CLIP index.")

                # Load optional PCA matrix
                pca_path = index_dir / "pca.matrix.pkl"
                if pca_path.exists():
                    with open(pca_path, "rb") as f:
                        st.session_state.pca_obj = pickle.load(f)
                    st.success("Loaded PCA matrix.")

                # Load bookmarks for the case
                st.session_state.bookmarks = load_bookmarks(index_dir)
                st.success(f"Loaded {len(st.session_state.bookmarks)} bookmarks.")

            # Load Whoosh index
            whoosh_path = index_dir / "whoosh_index"
            if whoosh_path.exists():
                whoosh_ix = open_whoosh_dir(str(whoosh_path))
                st.session_state.whoosh_searcher = whoosh_ix.searcher()
                st.success("Loaded text index searcher.")

            # Load embeddings for clustering
            embeddings_path = index_dir / "embeddings_combined.mmap"
            if embeddings_path.exists() and 'manifest' in st.session_state:
                num_embeddings = len(st.session_state.index_paths)
                # This is a bit of a hack, we need to get the dimension from the faiss index
                embedding_dim = st.session_state.faiss_combined.d
                st.session_state.embeddings = np.memmap(embeddings_path, dtype='float32', mode='r', shape=(num_embeddings, embedding_dim))
                st.success(f"Memory-mapped {num_embeddings} embeddings for clustering.")

        except Exception as e:
            st.error(f"Failed to load index files: {e}")


st.header("Backend Monitoring")
if st.button("Refresh Queue Stats"):
    try:
        r = redis.Redis(host=st.session_state.redis_host, port=st.session_state.redis_port, db=0)
        jobs_in_queue = r.llen("forceps:job_queue")
        results_in_queue = r.llen("forceps:results_queue")

        c1, c2 = st.columns(2)
        with c1:
            st.metric("Jobs in Queue", f"{jobs_in_queue:,}")
        with c2:
            st.metric("Results in Queue", f"{results_in_queue:,}")

    except Exception as e:
        st.error(f"Could not connect to Redis: {e}")

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
    results = []
    vector_results = set()
    keyword_results = set()

    # 1. Vector Search (FAISS)
    if nl_query and st.session_state.get("faiss_clip"):
        q_emb = compute_clip_text_embedding(nl_query, clip_model)
        _, I = st.session_state.faiss_clip.search(np.array([q_emb], dtype=np.float32), 100)
        vector_results.update([st.session_state.index_paths[i] for i in I[0] if i != -1])
    elif uploaded:
        tmp = Image.open(uploaded).convert("RGB")
        vit_t = preprocess_vit(tmp).unsqueeze(0)
        clip_t = (preprocess_clip(tmp).unsqueeze(0) if preprocess_clip is not None else None)
        q_comb, _ = compute_batch_embeddings([vit_t.squeeze(0)], [clip_t.squeeze(0)] if clip_t is not None else [], vit_model, clip_model)
        q_emb = q_comb[0].astype(np.float32)

        if st.session_state.get("faiss_combined"):
            idx_to_use = st.session_state.faiss_combined
            if st.session_state.get("pca_obj"):
                q_emb = st.session_state.pca_obj.apply_py(np.array([q_emb]))[0]

            _, I = idx_to_use.search(np.array([q_emb], dtype=np.float32), 100)
            vector_results.update([st.session_state.index_paths[i] for i in I[0] if i != -1])

    # 2. Keyword Search (Whoosh)
    if nl_query and st.session_state.get("whoosh_searcher"):
        searcher = st.session_state.whoosh_searcher
        parser = QueryParser("content", schema=searcher.schema)
        query = parser.parse(nl_query)
        keyword_hits = searcher.search(query, limit=200)
        keyword_results.update([hit['path'] for hit in keyword_hits])

    # 3. Combine and Re-rank Results
    # Simple re-ranking: boost items that appear in both result sets.
    final_results = []
    intersection = vector_results.intersection(keyword_results)

    # Add boosted intersection results first
    final_results.extend(list(intersection))

    # Add remaining vector results
    final_results.extend([p for p in vector_results if p not in intersection])

    # Add remaining keyword results
    final_results.extend([p for p in keyword_results if p not in intersection])

    st.session_state.last_results = final_results

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

if st.session_state.get("last_results") and st.session_state.get("embeddings") is not None:
    st.markdown("---")
    st.subheader("Cluster Analysis")

    num_clusters = st.slider("Number of Clusters", min_value=2, max_value=50, value=10)

    if st.button("Cluster Displayed Results"):
        with st.spinner("Clustering results..."):
            paths_to_cluster = st.session_state.last_results
            path_to_idx = {path: i for i, path in enumerate(st.session_state.index_paths)}
            result_indices = [path_to_idx[p] for p in paths_to_cluster if p in path_to_idx]

            if result_indices:
                result_embeddings = st.session_state.embeddings[result_indices]

                kmeans = MiniBatchKMeans(n_clusters=num_clusters, random_state=0, batch_size=256, n_init='auto')
                kmeans.fit(result_embeddings)

                st.session_state.cluster_labels = {path: label for path, label in zip(paths_to_cluster, kmeans.labels_)}
                st.success(f"Successfully clustered results into {num_clusters} groups.")
                st.rerun()
            else:
                st.warning("No results to cluster.")

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

def display_image_tile(tile_container, image_path):
    """Renders a single image tile with all its controls."""
    r = image_path
    try:
        # --- Image Display ---
        encoded = urllib.parse.quote(r)
        try:
            mime, _ = mimetypes.guess_type(r)
            mime = mime or "image/jpeg"
            with open(r, "rb") as _f:
                b64 = base64.b64encode(_f.read()).decode("utf-8")
            data_url = f"data:{mime};base64,{b64}"
        except Exception:
            data_url = ""
        tile_container.markdown(f"<div class='img-card'><img src='{data_url}' /></div>", unsafe_allow_html=True)
        tile_container.caption(Path(r).name)

        # --- Details Expander ---
        if 'manifest' in st.session_state and r in st.session_state.manifest:
            with tile_container.expander("Show Details"):
                manifest_item = st.session_state.manifest[r]
                cached_meta = load_cache(fingerprint(Path(r))) or {}

                st.code(manifest_item['path'], language='text')

                st.markdown("**Hashes**")
                hashes = manifest_item.get('hashes', {})
                if hashes:
                    for hash_name, hash_value in hashes.items():
                        st.text(f"{hash_name.upper()}: {hash_value}")

                st.markdown("**EXIF Data**")
                exif_data = cached_meta.get("metadata", {}).get("exif", {})
                if exif_data:
                    for key, value in exif_data.items():
                        st.text(f"{key}: {value}")

                st.markdown("**AI Caption**")
                caption = cached_meta.get("metadata", {}).get("caption")
                st.text(caption or "Not available.")

        # --- Bookmark/Tags Expander ---
        def _is_bm(pth: str) -> bool:
            return pth in (st.session_state.get("bookmarks", {}) or {})

        def _save_bms():
            case_dir = Path(st.session_state.cases_dir) / st.session_state.selected_case
            save_bookmarks(st.session_state.bookmarks, case_dir)

        with tile_container.expander("Manage Bookmark"):
            is_bookmarked = _is_bm(r)

            if st.checkbox("Bookmarked", value=is_bookmarked, key=f"bm_check_{r}"):
                if not is_bookmarked:
                    st.session_state.bookmarks[r] = st.session_state.bookmarks.get(r, {"tags": [], "notes": "", "added_ts": time.time()})
                    _save_bms()
                    st.rerun()

                bookmark_data = st.session_state.bookmarks.get(r, {})

                existing_tags = "\n".join(bookmark_data.get("tags", []))
                new_tags_str = st.text_area("Tags (one per line)", value=existing_tags, key=f"tags_{r}")

                existing_notes = bookmark_data.get("notes", "")
                new_notes = st.text_area("Notes", value=existing_notes, key=f"notes_{r}")

                if st.button("Save Bookmark Details", key=f"save_bm_{r}"):
                    st.session_state.bookmarks[r]['tags'] = [t.strip() for t in new_tags_str.split("\n") if t.strip()]
                    st.session_state.bookmarks[r]['notes'] = new_notes
                    _save_bms()
                    st.success("Bookmark updated!")

            elif is_bookmarked:
                st.session_state.bookmarks.pop(r, None)
                _save_bms()
                st.rerun()
    except Exception as e:
        tile_container.error(f"Error: {e}")
        tile_container.write(Path(r).name)


if display_results:
    st.markdown("### Top results")

    # If results have been clustered, group them
    if 'cluster_labels' in st.session_state:
        clusters = {}
        for path in display_results:
            label = st.session_state.cluster_labels.get(path)
            if label is not None:
                if label not in clusters:
                    clusters[label] = []
                clusters[label].append(path)

        if st.button("Clear Clustering"):
            del st.session_state['cluster_labels']
            st.rerun()

        for label, paths in sorted(clusters.items()):
            st.markdown(f"--- \n#### Cluster {label+1} ({len(paths)} images)")
            cols = st.columns(5)
            for i, r in enumerate(paths):
                display_image_tile(cols[i%5], r)
    else:
        # Original, unclustered display
        cols = st.columns(5)
        for i, r in enumerate(display_results[:25]):
            display_image_tile(cols[i%5], r)

st.sidebar.markdown("---")
if st.sidebar.button("Exit App"):
    st.stop()
