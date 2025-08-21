#!/usr/bin/env python3
"""
FORCEPS: Forensic Optimized Retrieval & Clustering of Evidence via Perceptual Search
CLIP-based Streamlit app for image search and similarity.
All processing is local. CLIP embeddings only.
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
# FAISS removed - CLIP-only search
import redis
import json
import pickle
import yaml
import torch
from whoosh.index import open_dir as open_whoosh_dir
from whoosh.qparser import QueryParser
from sklearn.cluster import MiniBatchKMeans
# FAISS removed - CLIP-only search

# --- Safe Module Loading ---
def safe_import_modules():
    """Safely import optional modules and set availability flags"""
    global HAS_EMBEDDINGS, HAS_UTILS
    global load_models, compute_batch_embeddings, compute_clip_text_embedding
    global is_image_file, fingerprint, load_cache, save_cache, compute_perceptual_hashes
    global read_exif, _gather_metadata_for_path, hamming_distance
    global load_bookmarks, save_bookmarks, generate_bookmarks_pdf
    global compute_clip_text_embedding
    
    # Initialize flags
    HAS_EMBEDDINGS = False
    HAS_UTILS = False
    # Ollama removed - no captioning needed
    
    # Initialize function references
    load_models = None
    compute_batch_embeddings = None
    compute_clip_text_embedding = None
    is_image_file = None
    fingerprint = None
    load_cache = None
    save_cache = None
    compute_perceptual_hashes = None
    read_exif = None
    _gather_metadata_for_path = None
    hamming_distance = None
    load_bookmarks = None
    save_bookmarks = None
    generate_bookmarks_pdf = None
    # Ollama functions removed
    
    # Try to import embeddings
    try:
        from embeddings import load_models, compute_batch_embeddings, compute_clip_text_embedding
        HAS_EMBEDDINGS = True
        st.sidebar.success("✅ Embeddings module loaded")
    except ImportError as e:
        st.sidebar.warning(f"❌ Embeddings module: {e}")
    
    # Try to import utils
    try:
        from utils import (
            is_image_file, fingerprint, load_cache, save_cache, compute_perceptual_hashes,
            read_exif, _gather_metadata_for_path, hamming_distance,
            load_bookmarks, save_bookmarks, generate_bookmarks_pdf
        )
        HAS_UTILS = True
        st.sidebar.success("✅ Utils module loaded")
    except ImportError as e:
        st.sidebar.warning(f"❌ Utils module: {e}")
    
    # Ollama removed - no captioning needed

# Import modules safely after function definition
safe_import_modules()

# ---------------- Config ----------------
CACHE_DIR = Path("./cache")
CACHE_DIR.mkdir(exist_ok=True)
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 16))
MAX_WORKERS = int(os.environ.get("MAX_WORKERS", 8))
DEVICE = "cuda" if __import__("torch").cuda.is_available() else "cpu"

# CLIP-only config
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

# Safe mode info
st.info("This app is running in safe mode with graceful error handling for missing modules.")

# Load models (ViT + CLIP) - safely
if HAS_EMBEDDINGS:
    try:
        vit_model, clip_model, preprocess_vit, preprocess_clip, vit_dim, clip_dim = load_models()
        COMBINED_DIM = vit_dim + clip_dim
        
        # Store CLIP models in session state for sidebar display
        if clip_model is not None:
            st.session_state.clip_model = clip_model
            st.session_state.clip_preprocess = preprocess_clip
    except Exception as e:
        st.error(f"Error loading models: {e}")
        vit_model, clip_model, preprocess_vit, preprocess_clip, vit_dim, clip_dim = None, None, None, None, 0, 0
        COMBINED_DIM = 0
else:
    st.warning("Embeddings module not available. ML features will be disabled.")
    vit_model, clip_model, preprocess_vit, preprocess_clip, vit_dim, clip_dim = None, None, None, None, 0, 0
    COMBINED_DIM = 0

# --- Minimal Mode (fast, in-memory, no Redis/queues) ---
st.sidebar.markdown("### Minimal Mode")
min_mode = st.sidebar.checkbox("Fast in-memory indexing (simplified)", key="min_mode")
if min_mode:
    folder_min = st.sidebar.text_input("Folder", value=st.session_state.get("folder_path", str(Path(__file__).parent / "demo_images")), key="min_folder")
    # Speed controls
    model_choice = st.sidebar.selectbox(
        "Model",
        [
            "google/vit-tiny-patch16-224",
            "google/vit-small-patch16-224",
            "google/vit-base-patch16-224-in21k",
        ],
        index=0,
        help="Smaller models are much faster (tiny > small > base).",
        key="min_model",
    )
    batch_size_ctl = st.sidebar.number_input("Batch size", min_value=16, max_value=256, value=128, step=16, key="min_batch")
    io_workers_ctl = st.sidebar.number_input("Loader threads", min_value=4, max_value=32, value=12, step=1, key="min_io_workers")
    use_ivfpq = st.sidebar.checkbox("Train IVF-PQ after embeddings (faster search)", value=False, key="min_ivfpq")
    nlist_ctl = st.sidebar.number_input("IVF nlist", min_value=64, max_value=65536, value=4096, step=64, key="min_nlist") if use_ivfpq else 0
    pqm_ctl = st.sidebar.number_input("PQ M", min_value=4, max_value=128, value=64, step=4, key="min_pqm") if use_ivfpq else 0

    # Pre-resize utility (to cache)
    def _preresize_to_cache(src_dir: str, target: int = 224, workers: int = 12) -> str:
        src = Path(src_dir)
        if not src.is_dir():
            return src_dir
        cache_root = Path("./cache/resized")
        cache_root.mkdir(parents=True, exist_ok=True)
        key = str(src.resolve())
        import hashlib
        digest = hashlib.sha1(key.encode()).hexdigest()[:12]
        dst = cache_root / f"{digest}_{target}"
        if dst.exists():
            return str(dst)
        # build file list
        exts = {".jpg",".jpeg",".png",".bmp",".tiff",".webp"}
        files = []
        for r,_,fs in os.walk(src):
            for fn in fs:
                if Path(fn).suffix.lower() in exts:
                    files.append(Path(r)/fn)
        dst.mkdir(parents=True, exist_ok=True)
        p = st.sidebar.progress(0)
        t = st.sidebar.empty()
        done = 0
        from concurrent.futures import ThreadPoolExecutor
        def _resize_one(pth: Path):
            rel = pth.relative_to(src)
            out_path = dst / rel
            out_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                im = Image.open(pth).convert("RGB")
                if im.size != (target, target):
                    im = im.resize((target, target), Image.LANCZOS)
                im.save(out_path, quality=90)
            except Exception:
                try:
                    Image.new("RGB", (target, target)).save(out_path, quality=90)
                except Exception:
                    pass
            return 1
        with ThreadPoolExecutor(max_workers=workers) as ex:
            for _ in ex.map(_resize_one, files):
                done += 1
                if done % 50 == 0 or done == len(files):
                    pct = int(done*100/len(files)) if files else 100
                    p.progress(pct)
                    t.text(f"Pre-resize {pct}% ({done}/{len(files)})")
        return str(dst)

    if st.sidebar.button("Pre-resize to 224 and use cache", key="min_preresize") and folder_min:
        if not HAS_UTILS:
            st.sidebar.error("Utils module not available for image processing")
        else:
            folder_min = _preresize_to_cache(folder_min, 224, io_workers_ctl)
            st.sidebar.success(f"Using cached resized folder: {folder_min}")

    start_min = st.sidebar.button("Index Now", key="min_start")
    pbar = st.sidebar.progress(0)
    status = st.sidebar.empty()

    if start_min and folder_min:
        if not HAS_EMBEDDINGS:
            st.sidebar.error("Embeddings module not available for indexing")
        else:
            # FAISS removed - CLIP-only search
            imgs = []
            exts = {".jpg",".jpeg",".png",".bmp",".tiff",".webp"}
            for r,_,files in os.walk(folder_min):
                for fn in files:
                    if Path(fn).suffix.lower() in exts:
                        imgs.append(os.path.join(r, fn))

            if not imgs:
                st.warning("No images found.")
            else:
                batch = int(batch_size_ctl)
                all_embs = []
                total = len(imgs)
                done = 0
                from concurrent.futures import ThreadPoolExecutor
                for i in range(0, total, batch):
                    batch_paths = imgs[i:i+batch]
                    # threaded load
                    tensors = [None]*len(batch_paths)
                    def _load(ix_p):
                        ix, p = ix_p
                        try:
                            im = Image.open(p).convert("RGB")
                        except Exception:
                            im = Image.new("RGB", (224,224))
                        tensors[ix] = preprocess_vit(im)
                    with ThreadPoolExecutor(max_workers=int(io_workers_ctl)) as ex:
                        list(ex.map(_load, enumerate(batch_paths)))
                    # Load models per selected backbone
                    vm, cm, pv, pc, vd, cd = load_models(model_choice)
                    embs, _ = compute_batch_embeddings(tensors, [], vm, None)
                    # normalize for IP
                    embs = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-10)
                    all_embs.append(embs.astype(np.float32))
                    done = min(total, i+batch)
                    pct = int(done*100/total)
                    pbar.progress(pct)
                    status.text(f"Embeddings {pct}% ({done}/{total})")

                embs_mat = np.vstack(all_embs)
                if use_ivfpq:
                    # CLIP-only search - no FAISS needed
                    st.success(f"CLIP embeddings computed for {len(imgs)} images. Use CLIP search above.")

    st.markdown("---")
    st.subheader("Search (Minimal Mode)")
    upl = st.file_uploader("Upload an image to search", type=["jpg","jpeg","png","bmp","gif"], key="min_upl")
    if upl and st.session_state.get("min_index") is not None:
        if not HAS_EMBEDDINGS:
            st.error("Embeddings module not available for search")
        else:
            st.info("Use the CLIP search functionality above for image similarity search")

    # Stop rendering the rest of the complex UI in minimal mode
    st.stop()

# Session state init
if "phase1_ready" not in st.session_state:
    st.session_state.phase1_ready = False
if "phase2_ready" not in st.session_state:
    st.session_state.phase2_ready = False
if "index_paths" not in st.session_state:
    st.session_state.index_paths = []
if "clip_model" not in st.session_state:
    st.session_state.clip_model = None
if "clip_preprocess" not in st.session_state:
    st.session_state.clip_preprocess = None
if "index_lock" not in st.session_state:
    st.session_state.index_lock = threading.Lock()
if "watcher_running" not in st.session_state:
    st.session_state.watcher_running = False

if "last_results" not in st.session_state:
    # Persist last shown results across UI interactions (e.g., bookmarking)
    st.session_state.last_results = []
elif st.session_state.last_results is None:
    # Safety check: ensure last_results is never None
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
if "redis_host" not in st.session_state:
    st.session_state.redis_host = os.environ.get("REDIS_HOST", "127.0.0.1")
if "redis_port" not in st.session_state:
    try:
        st.session_state.redis_port = int(os.environ.get("REDIS_PORT", "6379"))
    except Exception:
        st.session_state.redis_port = 6379
if "case_type" not in st.session_state:
    st.session_state.case_type = None # Can be 'full' or 'triage'
if "search_distances" not in st.session_state:
    st.session_state.search_distances = {}
if "selected_items" not in st.session_state:
    st.session_state.selected_items = set()
if "bookmarks" not in st.session_state:
    st.session_state.bookmarks = {}
if "manifest" not in st.session_state:
    st.session_state.manifest = {}
if "cases_dir" not in st.session_state:
    st.session_state.cases_dir = "output_index"
if "selected_case" not in st.session_state:
    st.session_state.selected_case = None
if "min_index" not in st.session_state:
    st.session_state.min_index = None
if "min_paths" not in st.session_state:
    st.session_state.min_paths = []
if "phase1_ready" not in st.session_state:
    st.session_state.phase1_ready = False
if "phase2_ready" not in st.session_state:
    st.session_state.phase2_ready = False
if "index_paths" not in st.session_state:
    st.session_state.index_paths = []
    # FAISS removed - CLIP-only search
if "cluster_labels" not in st.session_state:
    st.session_state.cluster_labels = {}
elif st.session_state.cluster_labels is None:
    # Safety check: ensure cluster_labels is never None
    st.session_state.cluster_labels = {}
if "whoosh_searcher" not in st.session_state:
    st.session_state.whoosh_searcher = None

# Sidebar: Redis connection settings
st.sidebar.markdown("### Redis Settings")
st.session_state.redis_host = st.sidebar.text_input("Redis host", value=st.session_state.redis_host, key="redis_host_input")
st.session_state.redis_port = st.sidebar.number_input("Redis port", min_value=1, max_value=65535, value=int(st.session_state.redis_port), step=1, key="redis_port_input")
if st.sidebar.button("Test Redis connection", key="test_redis_conn"):
    try:
        r = redis.Redis(host=st.session_state.redis_host, port=int(st.session_state.redis_port), db=0)
        r.ping()
        st.sidebar.success(f"Connected to Redis at {st.session_state.redis_host}:{st.session_state.redis_port}")
    except Exception as e:
        st.sidebar.error(f"Redis connection failed: {e}")

# Module Status
st.sidebar.markdown("### Module Status")
st.sidebar.write(f"**Embeddings**: {'✅' if HAS_EMBEDDINGS else '❌'}")
st.sidebar.write(f"**Utils**: {'✅' if HAS_UTILS else '❌'}")
    # Ollama removed
st.sidebar.write(f"**CLIP**: {'✅' if st.session_state.get('clip_model') else '❌'}")
st.sidebar.write(f"**PyTorch**: {'✅' if torch else '❌'}")
st.sidebar.write(f"**Whoosh**: {'✅' if 'whoosh' in sys.modules else '❌'}")

# Show warnings for missing critical modules
if not HAS_EMBEDDINGS:
    st.sidebar.warning("⚠️ Embeddings module missing - ML features disabled")
if not HAS_UTILS:
    st.sidebar.warning("⚠️ Utils module missing - Basic functions disabled")

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
    # Bookmarks removed
except Exception:
    pass

# Sidebar controls
st.sidebar.header("FORCEPS Indexing Controls")

col_path, col_btn = st.sidebar.columns([0.75, 0.25])
with col_path:
    _inp = st.text_input("Folder to index", value=st.session_state.folder_path, key="folder_input")
    if _inp and _inp != st.session_state.folder_path:
        st.session_state.folder_path = _inp
with col_btn:
    if st.button("Browse…", key="os_picker_blocking"):
        sel, logs = _choose_folder_os()
        if sel:
            st.session_state.folder_path = sel
            st.session_state.browse_dir = sel
            st.rerun()
        else:
            st.sidebar.warning("OS file picker unavailable or canceled.")

## Removed inline folder browser; OS-native picker only
folder = st.session_state.folder_path
start_btn = st.sidebar.button("Start CLIP Indexing")
stop_btn = st.sidebar.button("Cancel Background Tasks")
save_idx_btn = st.sidebar.button("Save CLIP Embeddings")

concurrent_phase2 = False
# Phase 2 captions removed - CLIP-only search

phase1_placeholder = st.sidebar.empty()
log_placeholder = st.sidebar.empty()
progress1 = st.sidebar.progress(0)
# progress2 removed - no Phase 2 needed

# Live progress from Redis (updates every render)
try:
    # Optional lightweight auto-refresh for the sidebar
    try:
        st.autorefresh(interval=3000, key="progress_autorefresh")
    except Exception:
        pass
    import redis as _r
    rc = _r.Redis(host=st.session_state.redis_host, port=st.session_state.redis_port, db=0)
    total = int(rc.get("forceps:stats:total_images") or 0)
    done1 = int(rc.get("forceps:stats:embeddings_done") or 0)
    done2 = int(rc.get("forceps:stats:captions_done") or 0)
    p1_pct = min(100, int(done1 * 100 / total)) if total > 0 else 0
    p2_pct = min(100, int(done2 * 100 / total)) if total > 0 else 0
    progress1.progress(p1_pct)
    progress2.progress(p2_pct)
    phase1_placeholder.text(f"Phase 1: embeddings {p1_pct}% ({done1}/{total})")
    phase2_placeholder.text(f"Phase 2: captions {p2_pct}% ({done2}/{total})")
    if total > 0 and done1 >= total:
        st.session_state.phase1_ready = True
    if total > 0 and done2 >= total:
        st.session_state.phase2_ready = True
except Exception:
    # Ignore if Redis not available
    pass

# Bookmarks / Export section
st.sidebar.markdown("---")

st.sidebar.markdown("---")
st.sidebar.subheader("Interactive Indexing")
selected_count = len(st.session_state.get("selected_items", set()))
st.sidebar.info(f"{selected_count} items selected for indexing.")

# This feature is only active in triage mode when items are selected.
if st.session_state.get("case_type") == "triage" and selected_count > 0:
    st.sidebar.markdown("---") # Visual separator

    default_case_name = f"{st.session_state.get('selected_case', 'case')}-selection-{selected_count}"
    new_case_name = st.sidebar.text_input("New Case Name for Selection", value=default_case_name)

    if st.sidebar.button("🚀 Create Full Index from Selection"):
        if new_case_name and not new_case_name.isspace():
            try:
                # 1. Load config to get Redis details
                with open("config.yaml", 'r') as f:
                    config = yaml.safe_load(f)
                cfg_redis = config['redis']

                # 2. Connect to Redis
                r = redis.Redis(host=cfg_redis['host'], port=cfg_redis['port'], db=0)
                r.ping()

                # 3. Prepare the job data
                paths_to_index = list(st.session_state.selected_items)
                job_data = []
                for path in paths_to_index:
                    if path in st.session_state.manifest:
                        # We need the hashes from the original manifest
                        job_data.append({
                            "path": path,
                            "hashes": st.session_state.manifest[path].get("hashes", {})
                        })

                if not job_data:
                    st.sidebar.error("No valid items to enqueue.")
                else:
                    # 4. Enqueue the job (as a single batch)
                    r.rpush(cfg_redis['job_queue'], json.dumps(job_data))

                    # 5. Launch the index builder as a background process
                    st.sidebar.info("Launching background index builder...")
                    command = [
                        sys.executable,
                        str(Path(__file__).parent.parent / "app" / "build_index.py"),
                        "--case_name",
                        new_case_name,
                        "--mode",
                        "full"
                    ]
                    process = subprocess.Popen(
                        command,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT, # Combine stdout and stderr
                        text=True,
                        encoding='utf-8',
                        bufsize=1 # Line-buffered
                    )
                    st.session_state.indexer_process = process

                    # 6. Provide feedback and next steps
                    st.sidebar.success(f"Enqueued {len(job_data)} items for indexing.")
                    st.sidebar.success(f"Indexer for '{new_case_name}' started (PID: {process.pid}).")
                    st.sidebar.warning("Monitor progress in the 'Process Logs' tab.")

                    # Clear the selection and rerun to update UI
                    st.session_state.selected_items = set()
                    st.rerun()

            except FileNotFoundError:
                st.sidebar.error("Could not find app/config.yaml")
            except redis.exceptions.ConnectionError:
                st.sidebar.error(f"Could not connect to Redis at {cfg_redis.get('host', 'localhost')}:{cfg_redis.get('port', 6379)}.")
            except Exception as e:
                st.sidebar.error(f"An error occurred: {e}")
        else:
            st.sidebar.error("Please provide a valid name for the new case.")

# Clear bookmarks/tags actions
st.sidebar.markdown("")

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
        # Store embeddings in cache for CLIP search
        if clip_emb is not None:
            # Save CLIP embeddings for later search
            pass
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
            # FAISS removed - CLIP-only search
            if new_paths:
                st.info("New images detected - rebuild case for CLIP search")
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

def _local_index_fast(image_dir: str, case_name: str, batch_size: int = max(32, BATCH_SIZE)):
    # FAISS removed - CLIP-only search
    from embeddings import load_models, compute_batch_embeddings
    from concurrent.futures import ThreadPoolExecutor

    start_ts = time.time()
    img_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    paths = []
    for r, _, files in os.walk(image_dir):
        for fn in files:
            if Path(fn).suffix.lower() in img_exts:
                paths.append(os.path.join(r, fn))

    st.session_state.p1_total = len(paths)
    st.session_state.p1_done = 0
    phase1_placeholder.text("Phase 1: CLIP embeddings 0% (0/{})".format(len(paths)))
    progress1.progress(0)

    if not paths:
        st.warning("No images found.")
        return

    vit_model, clip_model, preprocess_vit, preprocess_clip, vit_dim, clip_dim = load_models()

    all_embs = []
    # batching
    for i in range(0, len(paths), batch_size):
        batch_paths = paths[i:i+batch_size]
        # Threaded image loading to overlap I/O for both VIT and CLIP
        vit_list = [None] * len(batch_paths)
        clip_list = [None] * len(batch_paths)
        def _load_into(idx_p):
            idx, pth = idx_p
            try:
                im = Image.open(pth).convert('RGB')
            except Exception:
                im = Image.new('RGB', (224, 224))
            vit_list[idx] = preprocess_vit(im)
            clip_list[idx] = preprocess_clip(im)
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=8) as ex:
            list(ex.map(_load_into, enumerate(batch_paths)))

        # Compute both VIT and CLIP embeddings
        comb_np, clip_np = compute_batch_embeddings(vit_list, clip_list, vit_model, clip_model)
        # L2-normalize for cosine IP
        comb_np = comb_np / (np.linalg.norm(comb_np, axis=1, keepdims=True) + 1e-10)
        if clip_np is not None:
            clip_np = clip_np / (np.linalg.norm(clip_np, axis=1, keepdims=True) + 1e-10)
        all_embs.append((comb_np.astype(np.float32), clip_np.astype(np.float32) if clip_np is not None else None, batch_paths))

        st.session_state.p1_done = min(len(paths), (i+batch_size))
        pct = int(st.session_state.p1_done * 100 / len(paths))
        progress1.progress(pct)
        phase1_placeholder.text(f"Phase 1: CLIP embeddings {pct}% ({st.session_state.p1_done}/{len(paths)})")

    # Process all embeddings and create manifest with CLIP embeddings
    out_dir = Path(st.session_state.cases_dir) / case_name
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    
    for comb_embs, clip_embs, batch_paths in all_embs:
        for i, path in enumerate(batch_paths):
            item = {
                "path": path,
                "hashes": {},
                "combined_embedding": comb_embs[i].tolist(),
            }
            if clip_embs is not None:
                item["clip_embedding"] = clip_embs[i].tolist()
            manifest.append(item)
    
    with open(out_dir / "manifest.json", 'w') as f:
        json.dump(manifest, f, indent=2)

    # Make the freshly built index immediately searchable in-session
            # FAISS removed - CLIP-only search
    st.session_state.index_paths = paths
    st.session_state.manifest = {item["path"]: item for item in manifest}
    st.session_state.pca_obj = None

    st.session_state.phase1_ready = True
    st.session_state.selected_case = case_name
    st.success(f"Embeddings complete. Saved to {out_dir}")

    # Captions removed - CLIP-only search
    st.session_state.phase2_ready = True

    st.session_state.total_index_duration = time.time() - start_ts

if start_btn:
    case_name = f"local-{int(time.time())}"
    _local_index_fast(folder, case_name, batch_size=BATCH_SIZE)

if save_idx_btn:
    # FAISS removed - CLIP-only search
    st.success("CLIP embeddings saved.")
else:
    st.error("No index to save yet.")

# Main app layout
search_tab, reporting_tab, logs_tab = st.tabs(["Search & Results", "Reporting", "Process Logs"])

with logs_tab:
    st.header("Background Process Logs")

    # This function will run in a separate thread to read logs without blocking the UI
    def log_reader_thread(process, buffer):
        try:
            for line in iter(process.stdout.readline, ''):
                buffer.append(line)
            process.stdout.close()
        except Exception as e:
            buffer.append(f"\n--- Log reader error: {e} ---\n")

    # Check if a process is running
    if "indexer_process" in st.session_state and st.session_state.indexer_process:
        proc = st.session_state.indexer_process

        # Start the log reader thread if it's not already running for this process
        if "log_reader_thread" not in st.session_state or not st.session_state.log_reader_thread.is_alive():
            st.session_state.log_buffer = []
            thread = threading.Thread(
                target=log_reader_thread,
                args=(proc, st.session_state.log_buffer),
                daemon=True
            )
            thread.start()
            st.session_state.log_reader_thread = thread

        # Display the logs from the buffer
        if "log_buffer" in st.session_state:
            st.code("".join(st.session_state.log_buffer), language="log")

        # Check if the process has finished
        if proc.poll() is not None:
            return_code = proc.poll()
            if return_code == 0:
                st.success(f"Indexer process (PID: {proc.pid}) finished successfully.")
            else:
                st.error(f"Indexer process (PID: {proc.pid}) failed with exit code: {return_code}.")
            # Clean up session state
            st.session_state.indexer_process = None
            st.session_state.log_reader_thread = None
        else:
            st.info(f"Monitoring indexer process (PID: {proc.pid})...")
            time.sleep(1) # Auto-refresh the log view
            st.rerun()
    else:
        st.info("No background processes are currently running.")


with search_tab:
    st.header("Backend Monitoring")
    if st.button("Refresh Queue Stats", key="refresh_q_top"):
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
        # Unified simple search UI (top of page)
        nl_query = st.text_input("Natural language query (optional)", key="nl_query_top")
        tag_query = st.text_input("Filter by tag (optional)", help="Show only images whose tags include this exact term", key="tag_query_top")
        uploaded = st.file_uploader("Or upload an image for search", type=["jpg","png","jpeg","bmp","gif"], key="uploader_top")
        run_search = st.button("Run Search", key="run_search_top")
        

                
    with col2:
        st.markdown("**Index readiness**")
        # Update phase flags from Redis stats if available
        try:
            import redis as _r
            rc = _r.Redis(host=st.session_state.redis_host, port=st.session_state.redis_port, db=0)
            total = int(rc.get("forceps:stats:total_images") or 0)
            done = int(rc.get("forceps:stats:embeddings_done") or 0)
            if total > 0 and done >= total:
                st.session_state.phase1_ready = True
        except Exception:
            pass
        st.write(f"- Phase 1 (embeddings): {'✅' if st.session_state.phase1_ready else '❌'}")
        # Phase 2 and Ollama removed - CLIP-only search
        if st.session_state.indexing_running:
            p1_pct = int((st.session_state.p1_done / st.session_state.p1_total) * 100) if st.session_state.p1_total else 0
            # Try to show real-time progress from Redis if available
            try:
                import redis as _r
                rc = _r.Redis(host=st.session_state.redis_host, port=st.session_state.redis_port, db=0)
                total = int(rc.get("forceps:stats:total_images") or 0)
                done = int(rc.get("forceps:stats:embeddings_done") or 0)
                if total > 0:
                    p1_pct = min(100, int(done * 100 / total))
            except Exception:
                pass
            progress1.progress(p1_pct)
            phase1_placeholder.text(f"Phase 1: CLIP embeddings {p1_pct}% ({st.session_state.p1_done}/{st.session_state.p1_total})")
        if st.session_state.index_error:
            st.error(st.session_state.index_error)
        # Stats after indexing
        if st.session_state.phase1_duration is not None:
            st.caption(f"CLIP embeddings duration: {st.session_state.phase1_duration:.1f}s")
        if st.session_state.total_index_duration is not None:
            st.caption(f"Total duration: {st.session_state.total_index_duration:.1f}s")

    results = st.session_state.get("last_results", [])
    # Ensure results is always a list
    if results is None:
        results = []
    
    if run_search:
        st.session_state.search_distances = {} # Clear old distances

        # CLIP-only search (no FAISS)
        if st.session_state.get("manifest") and st.session_state.get("clip_model"):
            with st.spinner("Searching with CLIP..."):
                try:
                    # Get CLIP model
                    clip_model = st.session_state.clip_model
                    
                    if nl_query:
                        # Text-to-image search
                        st.info(f"🔍 Searching for: '{nl_query}'")
                        query_emb = compute_clip_text_embedding(nl_query, clip_model)
                        
                        # Compare with all images in manifest
                        similarities = []
                        for path, item in st.session_state.manifest.items():
                            if 'clip_embedding' in item:
                                img_emb = np.array(item['clip_embedding'])
                                # Compute cosine similarity
                                similarity = np.dot(query_emb, img_emb) / (np.linalg.norm(query_emb) * np.linalg.norm(img_emb))
                                similarities.append((path, similarity))
                        
                        # Sort by similarity and filter by threshold
                        similarities.sort(key=lambda x: x[1], reverse=True)
                        threshold = 0.1  # Lower threshold for synthetic images
                        results = [path for path, sim in similarities if sim >= threshold][:100]
                        
                        # Store similarities for display
                        st.session_state.search_distances = {path: sim for path, sim in similarities if path in results}
                        
                        st.success(f"✅ Found {len(results)} images with similarity ≥ {threshold}")
                        
                        # Debug: Show top similarities
                        if similarities:
                            st.write("**Top similarities:**")
                            for i, (path, sim) in enumerate(similarities[:5]):
                                filename = Path(path).name
                                st.write(f"- {filename}: {sim:.4f}")
                        
                    elif uploaded:
                        # Image-to-image search
                        st.info("🔍 Searching for similar images...")
                        
                        # Process uploaded image
                        tmp = Image.open(uploaded).convert("RGB")
                        # Use CLIP preprocessing
                        clip_preprocess = st.session_state.get("clip_preprocess")
                        if clip_preprocess:
                            img_tensor = clip_preprocess(tmp).unsqueeze(0)
                            query_emb = clip_model.encode_image(img_tensor).detach().cpu().numpy()[0]
                            
                            # Compare with all images in manifest
                            similarities = []
                            for path, item in st.session_state.manifest.items():
                                if 'clip_embedding' in item:
                                    img_emb = np.array(item['clip_embedding'])
                                    # Compute cosine similarity
                                    similarity = np.dot(query_emb, img_emb) / (np.linalg.norm(query_emb) * np.linalg.norm(img_emb))
                                    similarities.append((path, similarity))
                            
                            # Sort by similarity and filter by threshold
                            similarities.sort(key=lambda x: x[1], reverse=True)
                            threshold = 0.1  # Lower threshold for synthetic images
                            results = [path for path, sim in similarities if sim >= threshold][:100]
                            
                            # Store similarities for display
                            st.session_state.search_distances = {path: sim for path, sim in similarities if path in results}
                            
                            st.success(f"✅ Found {len(results)} similar images with similarity ≥ {threshold}")
                            
                            # Debug: Show top similarities
                            if similarities:
                                st.write("**Top similarities:**")
                                for i, (path, sim) in enumerate(similarities[:5]):
                                    filename = Path(path).name
                                    st.write(f"- {filename}: {sim:.4f}")
                        else:
                            st.error("❌ CLIP preprocessing not available")
                    
                    else:
                        st.warning("⚠️ Please provide either a text query or upload an image")
                        
                except Exception as e:
                    st.error(f"❌ CLIP search failed: {e}")
                    import traceback
                    st.code(traceback.format_exc())
        else:
            st.error("❌ CLIP model or manifest not available. Please load a case with CLIP embeddings.")

        st.session_state.last_results = results
        
        # Debug search results
        st.write(f"**Search completed:** {len(results)} results found")
        if results:
            st.write(f"**First result:** {results[0]}")
            st.write(f"**Results stored in session:** {len(st.session_state.last_results)}")
        st.write(f"**Search type:** {'triage' if st.session_state.get('case_type') == 'triage' else 'full'}")
        
        # Debug search state
        st.write(f"**Debug info:**")
        st.write(f"- clip_model: {'✅' if st.session_state.get('clip_model') else '❌'}")
        manifest = st.session_state.get('manifest') or {}
        st.write(f"- manifest entries: {len(manifest)}")
        st.write(f"- search results: {len(results)}")
        
        # If no results, show helpful message
        if not results:
            st.info("💡 Search tips:")
            st.write("- Use text queries like 'red car' or 'person'")
            st.write("- Upload an image to find similar images")
            st.write("- Make sure CLIP model is loaded")

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
    # Safety check: ensure results is iterable
    if results is None:
        results = []
    
    st.write(f"**Filtering {len(results)} results...**")
    st.write(f"**Filter settings:**")
    st.write(f"- EXIF filter: {exif_availability}")
    st.write(f"- Make filter: {selected_make_value}")
    st.write(f"- Model filter: {selected_model_value}")
    st.write(f"- Include regex: {include_regex_str if 'include_regex_str' in locals() else 'None'}")
    st.write(f"- Exclude regex: {exclude_regex_str if 'exclude_regex_str' in locals() else 'None'}")
    st.write(f"- Time filter: {use_mtime_filter}")
    st.write(f"- Tag query: {tag_query if 'tag_query' in locals() else 'None'}")
    
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
    
    st.write(f"**After filtering: {len(display_results)} results remain**")

    # Debug information
    st.markdown("---")
    st.subheader("Debug Info")
    st.write(f"**Case loaded:** {st.session_state.get('selected_case', 'None')}")
    st.write(f"**CLIP model:** {'✅' if st.session_state.get('clip_model') else '❌'}")
    manifest = st.session_state.get('manifest') or {}
    st.write(f"**Manifest entries:** {len(manifest)}")
    st.write(f"**Search results:** {len(results)}")
    st.write(f"**Display results:** {len(display_results)}")
    
    if not results:
        st.info("Type a natural-language query (requires CLIP) or upload an image.")
    elif not display_results:
        st.warning("No results match current filters.")

    # CLIP search results display
    if st.session_state.get("last_results"):
        st.markdown("---")
        st.subheader("Search Results")
        st.info("🔍 Results from CLIP search (no clustering)")
        
        # Display the actual search results
        search_results = st.session_state.last_results
        if search_results:
            st.write(f"**Showing {len(search_results)} search results:**")
            
            # Create columns for image display
            cols = st.columns(5)
            for i, result_path in enumerate(search_results[:50]):  # Show first 50 results
                try:
                    # Get similarity score if available
                    similarity = st.session_state.get("search_distances", {}).get(result_path, 0)
                    
                    # Load and display image
                    try:
                        mime, _ = mimetypes.guess_type(result_path)
                        mime = mime or "image/jpeg"
                        with open(result_path, "rb") as f:
                            b64 = base64.b64encode(f.read()).decode("utf-8")
                        data_url = f"data:{mime};base64,{b64}"
                    except Exception:
                        data_url = ""
                    
                    # Display in appropriate column
                    col_idx = i % 5
                    with cols[col_idx]:
                        st.markdown(f"<div class='img-card'><img src='{data_url}' /></div>", unsafe_allow_html=True)
                        caption = Path(result_path).name
                        if similarity > 0:
                            caption += f" (sim: {similarity:.3f})"
                        st.caption(caption)
                        
                except Exception as e:
                    # If image display fails, show filename
                    col_idx = i % 5
                    cols[col_idx].write(f"Error: {Path(result_path).name}")
        else:
            st.warning("No search results to display")

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

            # Display filename and Hamming distance if available
            caption_text = Path(r).name
            dist = st.session_state.get("search_distances", {}).get(r)
            if dist is not None and dist > 0:
                caption_text += f" (dist: {dist})"
            tile_container.caption(caption_text)

            # --- Selection Checkbox ---
            is_selected = r in st.session_state.get("selected_items", set())
            # The checkbox state determines if the item should be in the set.
            if tile_container.checkbox("Select for full index", value=is_selected, key=f"select_top_{r}"):
                if not is_selected:
                    st.session_state.selected_items.add(r)
                    st.rerun()
            elif is_selected:
                st.session_state.selected_items.remove(r)
                st.rerun()

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

                if st.checkbox("Bookmarked", value=is_bookmarked, key=f"bm_check_top_{r}"):
                    if not is_bookmarked:
                        st.session_state.bookmarks[r] = st.session_state.bookmarks.get(r, {"tags": [], "notes": "", "added_ts": time.time()})
                        _save_bms()
                        st.rerun()

                    bookmark_data = st.session_state.bookmarks.get(r, {})

                    existing_tags = "\n".join(bookmark_data.get("tags", []))
                    new_tags_str = st.text_area("Tags (one per line)", value=existing_tags, key=f"tags_top_{r}")

                    existing_notes = bookmark_data.get("notes", "")
                    new_notes = st.text_area("Notes", value=existing_notes, key=f"notes_top_{r}")

                    if st.button("Save Bookmark Details", key=f"save_bm_top_{r}"):
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
        if 'cluster_labels' in st.session_state and st.session_state.cluster_labels is not None:
            clusters = {}
            for path in display_results:
                label = st.session_state.cluster_labels.get(path)
                if label is not None:
                    if label not in clusters:
                        clusters[label] = []
                    clusters[label].append(path)

            if st.button("Clear Clustering", key="clear_cluster_top"):
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
def reset_case_state():
    """Clears all session state related to a loaded case."""
    # Reset complex objects to None
    complex_keys = ["manifest", "index_paths",
                   "whoosh_searcher", "embeddings", "case_type", "last_results"]
    for key in complex_keys:
        if key in st.session_state:
            st.session_state[key] = None
    
    # Reset collections to empty containers
    if "bookmarks" in st.session_state:
        st.session_state.bookmarks = {}
    if "cluster_labels" in st.session_state:
        st.session_state.cluster_labels = {}

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
        reset_case_state() # Clear previous case data
        index_dir = Path(st.session_state.cases_dir) / st.session_state.selected_case
        st.sidebar.info(f"Loading case: {st.session_state.selected_case}")

        try:
            # A case MUST have a manifest.json file. This is the source of truth.
            manifest_path = index_dir / "manifest.json"
            if not manifest_path.exists():
                st.error(f"Case is invalid: manifest.json not found in {index_dir}")
            else:
                with open(manifest_path, "r") as f:
                    manifest_data = json.load(f)
                st.session_state.manifest = {item['path']: item for item in manifest_data}
                st.session_state.index_paths = [item['path'] for item in manifest_data]
                st.success(f"Loaded manifest for {len(st.session_state.index_paths)} images.")

                # Now, conditionally load all other assets if they exist.
                # This determines if the case is 'full' or 'triage'.

                # CLIP-only case (no FAISS needed)
                st.session_state.case_type = "clip" # This is a CLIP-only case
                st.success("Loaded CLIP-only case.")
                st.session_state.phase1_ready = True

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
                    # Heuristic: mark captions ready if any cached caption exists
                    try:
                        has_caption = False
                        for pth in st.session_state.index_paths[:100]:  # sample first 100
                            cached = load_cache(fingerprint(Path(pth))) or {}
                            cap = (cached.get("metadata", {}) or {}).get("caption")
                            if cap:
                                has_caption = True
                                break
                        if has_caption:
                            st.session_state.phase2_ready = True
                    except Exception:
                        pass
                else:
                    st.error("manifest.json not found in directory.")

                # Load CLIP model for direct search (no FAISS index needed)
                try:
                    from embeddings import load_models
                    vit_model, clip_model, preprocess_vit, preprocess_clip, vit_dim, clip_dim = load_models()
                    if clip_model:
                        st.session_state.clip_model = clip_model
                        st.session_state.clip_preprocess = preprocess_clip
                        st.success("✅ CLIP model loaded for direct search")
                    else:
                        st.warning("⚠️ CLIP model not available")
                except Exception as e:
                    st.warning(f"⚠️ Could not load CLIP model: {e}")

                # PCA Matrix
                pca_path = index_dir / "pca.matrix.pkl"
                if pca_path.exists():
                    with open(pca_path, "rb") as f:
                        st.session_state.pca_obj = pickle.load(f)

                # Text Index (Whoosh)
                whoosh_path = index_dir / "whoosh_index"
                if whoosh_path.exists():
                    whoosh_ix = open_whoosh_dir(str(whoosh_path))
                    st.session_state.whoosh_searcher = whoosh_ix.searcher()

                # CLIP embeddings loaded from manifest (no FAISS needed)

                # Bookmarks are always loaded
                st.session_state.bookmarks = load_bookmarks(index_dir)
                st.success(f"Loaded {len(st.session_state.bookmarks)} bookmarks.")
        except Exception as e:
            st.error(f"An error occurred while loading the case: {e}")
            reset_case_state() # Clean up on failure


# --- Case management (delete, refresh) ---
with st.sidebar.expander("Manage Cases", expanded=False):
    try:
        p = Path(st.session_state.cases_dir)
        all_cases = sorted([d.name for d in p.iterdir() if d.is_dir()], reverse=True) if p.is_dir() else []
    except Exception:
        all_cases = []

    if not all_cases:
        st.caption("No cases to manage.")
    else:
        to_delete = st.multiselect("Select case(s) to delete", all_cases, key="cases_to_delete")
        confirm_del = st.checkbox("I understand this permanently deletes selected case folders.")
        if st.button("Delete Selected Cases"):
            if not to_delete:
                st.warning("No cases selected.")
            elif not confirm_del:
                st.warning("Please confirm deletion.")
            else:
                deleted = []
                errors = []
                for name in to_delete:
                    case_dir = Path(st.session_state.cases_dir) / name
                    try:
                        if case_dir.is_dir():
                            shutil.rmtree(case_dir)
                            deleted.append(name)
                            # If we deleted currently loaded case, clear session indices
                            if st.session_state.get("selected_case") == name:
                                st.session_state.clip_model = None
                                st.session_state.clip_preprocess = None
                                st.session_state.index_paths = []
                                st.session_state.manifest = {}
                                st.session_state.phase1_ready = False
                                st.session_state.phase2_ready = False
                        else:
                            errors.append(f"Not a directory: {case_dir}")
                    except Exception as e:
                        errors.append(f"{name}: {e}")
                if deleted:
                    st.success(f"Deleted: {', '.join(deleted)}")
                for msg in errors:
                    st.error(msg)
                # Refresh page to reflect updated case list
                st.rerun()


## Duplicate Backend Monitoring and Search UI removed (content provided in the main Search & Results tab)

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

    # CLIP search results display
    if st.session_state.get("last_results"):
        st.markdown("---")
        st.subheader("Search Results")
        st.info("🔍 Results from CLIP search (no clustering)")

    # Display results
    if display_results:
        st.markdown("### Top results")
        
        # Simple display without clustering
        cols = st.columns(5)
        for i, r in enumerate(display_results[:25]):
            display_image_tile(cols[i%5], r)

st.sidebar.markdown("---")
if st.sidebar.button("Exit App"):
    st.stop()
