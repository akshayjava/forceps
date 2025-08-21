import streamlit as st
import os
import sys
import json
import yaml
import time
import numpy as np
import faiss
import pickle
from pathlib import Path
import logging
import redis
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
import torch
from PIL import Image

# Add app directory to sys.path to import modules
sys.path.insert(0, os.path.abspath('./app'))

# Import refactored functions or classes
from app.hashers import get_hashers
from app.distributed_engine import OptimizedRedisClient, WorkerStats
from app.optimized_embeddings import OptimizedEmbeddingComputer, optimize_gpu_settings
from app.embeddings import load_models, compute_batch_embeddings
from app.utils import fingerprint, load_cache, save_cache, read_exif
from app.llm_ollama import ollama_installed, generate_caption_ollama, model_available, general_ollama_query

# Configure logging for Streamlit
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Configuration Loading ---
@st.cache_data
def load_config(config_path="app/config.yaml"):
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        st.error(f"Configuration file not found at: {config_path}")
        return None
    except Exception as e:
        st.error(f"Error reading configuration file: {e}")
        return None

config = load_config()
if not config:
    st.stop()

# --- Refactored Functions from original scripts ---

def enqueue_jobs_programmatic(input_dir: str, config: dict):
    st.info(f"Scanning for images in {input_dir} and enqueuing jobs...")
    cfg_redis = config['redis']
    cfg_perf = config['performance']['enqueuer']

    hashers_to_run = config.get('hashing', [])
    hashers = get_hashers(hashers_to_run)
    if not hashers:
        st.warning("No hashers configured. No hashes will be computed.")
        return 0

    try:
        r = redis.Redis(host=cfg_redis['host'], port=cfg_redis['port'], db=0, decode_responses=True)
        r.ping()
        st.success(f"Successfully connected to Redis at {cfg_redis['host']}:{cfg_redis['port']}")
    except redis.exceptions.ConnectionError as e:
        st.error(f"Could not connect to Redis: {e}. Please ensure Redis is running.")
        return 0

    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".webp"}
    image_data = []
    
    def process_single_file(file_path, hashers_list):
        try:
            all_hashes = {}
            for hasher in hashers_list:
                all_hashes.update(hasher.compute(file_path))
            return {"path": str(file_path), "hashes": all_hashes}
        except Exception as e:
            logger.warning(f"Could not process file {file_path}: {e}")
            return None

    # Use a ThreadPoolExecutor to parallelize both walking and processing
    with ThreadPoolExecutor(max_workers=cfg_perf['scan_max_workers']) as executor:
        futures = []
        for root, _, files in os.walk(input_dir):
            for filename in files:
                if Path(filename).suffix.lower() in image_extensions:
                    file_path = Path(root) / filename
                    futures.append(executor.submit(process_single_file, file_path, hashers))

        for i, future in enumerate(as_completed(futures)):
            result = future.result()
            if result:
                image_data.append(result)
            if (i + 1) % 100 == 0: # Update progress more frequently for UI
                st.info(f"Scanned and processed {i + 1} files...")

    st.info(f"Found and processed {len(image_data)} total images.")

    # Initialize progress stats in Redis
    try:
        r.set("forceps:stats:total_images", len(image_data))
        r.set("forceps:stats:embeddings_done", 0)
        r.set("forceps:stats:captions_done", 0)
    except Exception as e:
        st.warning(f"Failed to set Redis counters: {e}")

    jobs_enqueued = 0
    for i in range(0, len(image_data), cfg_perf['job_batch_size']):
        batch = image_data[i:i + cfg_perf['job_batch_size']]
        r.rpush(cfg_redis['job_queue'], json.dumps(batch))
        jobs_enqueued += 1

    st.success(f"Enqueued {jobs_enqueued} jobs with a total of {len(image_data)} images to queue '{cfg_redis['job_queue']}'.")
    return len(image_data)

def run_optimized_worker_once(config: dict):
    st.info("Starting optimized worker to process jobs from Redis...")
    cfg_redis = config['redis']
    
    try:
        r = redis.Redis(host=cfg_redis['host'], port=cfg_redis['port'], db=0, decode_responses=True)
        r.ping()
    except redis.exceptions.ConnectionError as e:
        st.error(f"Could not connect to Redis: {e}. Please ensure Redis is running.")
        return

    # Initialize embedding computer
    worker_config = config.get('performance', {}).get('worker', {})
    embedding_computer = OptimizedEmbeddingComputer(
        max_batch_size=worker_config.get('batch_size', 128),
        use_mixed_precision=worker_config.get('use_mixed_precision', True),
        enable_cuda_streams=worker_config.get('enable_cuda_streams', True)
    )

    processed_count = 0
    while True:
        jobs = r.lrange(cfg_redis['job_queue'], 0, 0) # Peek one job
        if not jobs:
            break # No more jobs

        # Atomically pop the job
        _, job_data_raw = r.blpop(cfg_redis['job_queue'], timeout=1)
        if not job_data_raw:
            break # Should not happen if lrange showed a job, but for safety

        job_data = json.loads(job_data_raw)
        
        try:
            # Process the job
            image_paths = [item['path'] for item in job_data]
            path_to_data = {item['path']: item for item in job_data}
            
            results = []
            for embedding_result in embedding_computer.compute_embeddings_streaming(
                image_paths, 
                batch_size=None  # Use auto-sizing
            ):
                path = embedding_result['path']
                original_data = path_to_data.get(path, {})
                
                result = {
                    "path": path,
                    "combined_emb": embedding_result['combined_embedding'],
                    "hashes": original_data.get('hashes', {})
                }
                if 'clip_embedding' in embedding_result:
                    result["clip_emb"] = embedding_result['clip_embedding']
                
                results.append(result)
            
            if results:
                r.rpush(cfg_redis['results_queue'], json.dumps(results))
                r.incrby("forceps:stats:embeddings_done", len(results))
                processed_count += len(results)
                st.progress(processed_count / r.get("forceps:stats:total_images"), text=f"Processed {processed_count} embeddings...")

        except Exception as e:
            st.error(f"Error processing job: {e}")
            # Optionally re-queue the job or move to an error queue
            break # Stop on error for simplicity in web app

    st.success(f"Optimized worker finished. Processed {processed_count} embeddings.")


def build_index_programmatic(config: dict, case_name: str = None):
    st.info("Building FAISS and Whoosh indexes...")
    cfg_redis = config['redis']
    cfg_data = config['data']
    cfg_faiss = config['performance']['faiss']

    case_name = case_name or f'case_{int(time.time())}'
    case_output_dir = Path(cfg_data['output_dir']) / case_name
    case_output_dir.mkdir(parents=True, exist_ok=True)

    faiss_args = argparse.Namespace(**cfg_faiss)

    try:
        r = redis.Redis(host=cfg_redis['host'], port=cfg_redis['port'], db=0, decode_responses=True)
        r.ping()
    except redis.exceptions.ConnectionError as e:
        st.error(f"Could not connect to Redis: {e}. Please ensure Redis is running.")
        return None

    all_results = []
    while True:
        items = r.lrange(cfg_redis['results_queue'], 0, 99)
        if not items:
            break
        r.ltrim(cfg_redis['results_queue'], len(items), -1) # Remove consumed items
        for item in items:
            all_results.extend(json.loads(item))
        st.info(f"Consumed {len(items)} results. Total embeddings so far: {len(all_results)}")
        time.sleep(0.1) # Yield to Streamlit

    if not all_results:
        st.warning("No embeddings were found in the results queue. Index not built.")
        return None

    # Prepare manifest data
    manifest_data = [{"path": res["path"], "hashes": res.get("hashes")} for res in all_results]

    # Prepare embedding data
    combined_embs = np.array([res["combined_emb"] for res in all_results], dtype=np.float32)
    has_clip = "clip_emb" in all_results[0] if all_results else False
    clip_embs = None
    if has_clip:
        clip_embs = np.array([res["clip_emb"] for res in all_results if "clip_emb" in res], dtype=np.float32)
    n, d_comb = combined_embs.shape

    # Build Vector Index (FAISS) - Simplified version of build_index_for_embeddings
    st.info("Building vector (FAISS) index...")
    use_gpu = torch.cuda.is_available()
    gpu_res = faiss.GpuResources() if use_gpu else None

    pca_ret = None
    d_final = d_comb
    if faiss_args.use_pca:
        eff_pca_dim = max(1, min(faiss_args.pca_dim, d_comb, n))
        pca_mat = faiss.PCAMatrix(d_comb, eff_pca_dim)
        pca_mat.train(combined_embs[:faiss_args.train_samples])
        pca_ret = pca_mat
        d_final = eff_pca_dim

    nlist = min(faiss_args.ivf_nlist, n // 100) if n > 100 else 1
    nlist = max(nlist, 1)
    pq_m = faiss_args.pq_m # Assuming pq_m is directly usable
    quantizer = faiss.IndexFlatL2(d_final)
    cpu_index = faiss.IndexIVFPQ(quantizer, d_final, nlist, pq_m, 8)

    index_to_train = faiss.index_cpu_to_gpu(gpu_res, 0, cpu_index) if use_gpu else cpu_index
    index_to_train.train(pca_ret.apply_py(combined_embs[:faiss_args.train_samples]) if faiss_args.use_pca else combined_embs[:faiss_args.train_samples])

    add_batch_size = faiss_args.add_batch
    for off in range(0, n, add_batch_size):
        end = off + add_batch_size
        batch_data = combined_embs[off:end]
        if faiss_args.use_pca: batch_data = pca_ret.apply_py(batch_data)
        index_to_train.add(batch_data)

    final_index_comb = faiss.index_gpu_to_cpu(index_to_train) if use_gpu else index_to_train
    final_index_comb.nprobe = min(64, max(1, nlist // 16))

    index_clip = None
    if has_clip and clip_embs is not None:
        # Simplified clip index building, assuming no PCA for clip for now
        _, d_clip = clip_embs.shape
        clip_quantizer = faiss.IndexFlatL2(d_clip)
        clip_cpu_index = faiss.IndexIVFPQ(clip_quantizer, d_clip, nlist, pq_m, 8)
        clip_index_to_train = faiss.index_cpu_to_gpu(gpu_res, 0, clip_cpu_index) if use_gpu else clip_cpu_index
        clip_index_to_train.train(clip_embs[:faiss_args.train_samples])
        for off in range(0, n, add_batch_size):
            end = off + add_batch_size
            clip_index_to_train.add(clip_embs[off:end])
        index_clip = faiss.index_gpu_to_cpu(clip_index_to_train) if use_gpu else clip_index_to_train
        index_clip.nprobe = min(64, max(1, nlist // 16))

    st.success("Vector index building complete.")

    # Save FAISS and PCA files
    st.info("Saving FAISS indexes and manifest...")
    faiss.write_index(final_index_comb, str(case_output_dir / "image_index.faiss"))
    if index_clip:
        faiss.write_index(index_clip, str(case_output_dir / "clip.index"))
    if pca_ret:
        with open(case_output_dir / "pca.matrix.pkl", "wb") as f:
            pickle.dump(pca_ret, f)
    
    # Save image_paths.pkl and metadata.pkl
    image_paths_only = [item['path'] for item in manifest_data]
    with open(case_output_dir / "image_paths.pkl", "wb") as f:
        pickle.dump(image_paths_only, f)
    
    # For metadata.pkl, we need to extract actual metadata if available,
    # for now, just save a dummy or simplified version
    # In a real scenario, this would come from the original processing
    metadata_data = {} # Placeholder, actual metadata would be more complex
    with open(case_output_dir / "metadata.pkl", "wb") as f:
        pickle.dump(metadata_data, f)

    # Save manifest.json for compatibility/inspection
    with open(case_output_dir / "manifest.json", "w") as f:
        json.dump(manifest_data, f, indent=2)

    st.success(f"Index building complete for case '{case_name}'. Saved to {case_output_dir}")
    return case_output_dir

def run_captions_programmatic(image_paths: list, config: dict):
    st.info("Starting caption generation (Phase 2)...")
    if not ollama_installed() or not model_available("llava"):
        st.warning("Ollama or llava model not available. Skipping captioning. Please ensure Ollama is running and 'llava' model is pulled.")
        return

    max_workers = config["performance"]["worker"]["max_workers"]
    
    # Dummy args object for phase2_caption
    class Args:
        def __init__(self, mw: int):
            self.max_workers = mw
    
    # Use the actual phase2_caption from app.engine
    from app.engine import phase2_caption
    phase2_caption(image_paths, Args(max_workers))
    st.success("Caption generation complete.")

# --- Streamlit UI ---
st.set_page_config(layout="wide", page_title="FORCEPS Lite")

st.title("FORCEPS Lite Web App")

st.sidebar.header("Configuration")
input_dir_default = config['data']['input_dir']
input_dir = st.sidebar.text_input("Image Directory for Indexing", value=input_dir_default)
output_dir_default = config['data']['output_dir']
output_dir = st.sidebar.text_input("Output Directory for Indexes", value=output_dir_default)

st.sidebar.markdown("---")
st.sidebar.header("Redis Status")
redis_host = config['redis']['host']
redis_port = config['redis']['port']
try:
    r_check = redis.Redis(host=redis_host, port=redis_port, db=0, decode_responses=True)
    r_check.ping()
    st.sidebar.success(f"Connected to Redis at {redis_host}:{redis_port}")
    total_images = r_check.get("forceps:stats:total_images") or 0
    embeddings_done = r_check.get("forceps:stats:embeddings_done") or 0
    captions_done = r_check.get("forceps:stats:captions_done") or 0
    st.sidebar.info(f"Total Images: {total_images}, Embeddings Done: {embeddings_done}, Captions Done: {captions_done}")
except Exception:
    st.sidebar.error(f"Could not connect to Redis at {redis_host}:{redis_port}. Please start Redis.")

st.header("1. Build Index and Captions")
st.write("This process will scan your image directory, compute embeddings, build a searchable index, and generate captions using Ollama (if configured).")

if st.button("Start Indexing and Captioning"):
    if not os.path.isdir(input_dir):
        st.error(f"Input directory '{input_dir}' does not exist.")
    else:
        st.session_state.indexing_case_dir = None
        with st.spinner("Starting indexing and captioning..."):
            # Step 1: Enqueue Jobs
            total_images_found = enqueue_jobs_programmatic(input_dir, config)
            if total_images_found > 0:
                # Step 2: Run Worker to process jobs
                run_optimized_worker_once(config)
                
                # Step 3: Build Index
                case_output_path = build_index_programmatic(config)
                st.session_state.indexing_case_dir = case_output_path
                
                # Step 4: Generate Captions
                if case_output_path:
                    # Need to load image paths from the newly built index
                    try:
                        with open(case_output_path / "image_paths.pkl", "rb") as f:
                            indexed_image_paths = pickle.load(f)
                        run_captions_programmatic(indexed_image_paths, config)
                    except Exception as e:
                        st.error(f"Error loading indexed image paths for captioning: {e}")
                st.success("Indexing and Captioning process completed!")
            else:
                st.warning("No images found or jobs enqueued. Indexing skipped.")

st.header("2. Search by Image")
st.write("Upload an image to find similar images in your indexed collection.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "bmp", "gif", "tiff", "webp"])

if uploaded_file is not None:
    if 'indexing_case_dir' not in st.session_state or not st.session_state.indexing_case_dir:
        st.warning("Please build an index first in 'Build Index and Captions' section.")
    else:
        st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)
        st.write("")
        st.write("Searching...")

        # Load models for query
        vit_model, clip_model, preprocess_vit, preprocess_clip, vit_dim, clip_dim = load_models()

        # Load index and image paths from the last built index
        case_dir = st.session_state.indexing_case_dir
        try:
            index_path = case_dir / "image_index.faiss"
            with open(case_dir / "image_paths.pkl", "rb") as f:
                image_paths = pickle.load(f)
            
            index = faiss.read_index(str(index_path))
            st.info(f"Loaded index with {index.ntotal} vectors.")

            # Preprocess and compute embedding for uploaded image
            img = Image.open(uploaded_file).convert("RGB")
            vit_tensor = preprocess_vit(img)
            query_emb, _ = compute_batch_embeddings([vit_tensor], [], vit_model, None)
            query_vector = query_emb[0].astype(np.float32)
            query_vector = query_vector / (np.linalg.norm(query_vector) + 1e-10) # Normalize

            # Perform search
            distances, indices = index.search(np.array([query_vector]), 10) # Top 10 results

            st.subheader("Top 10 Similar Images:")
            for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
                if idx >= 0 and idx < len(image_paths):
                    result_path = image_paths[idx]
                    st.write(f"**{i+1}.** {Path(result_path).name} (Similarity: {dist:.3f})")
                    st.image(result_path, width=100)
                else:
                    st.write(f"**{i+1}.** Invalid index: {idx}")

        except Exception as e:
            st.error(f"Error during image search: {e}")

st.header("3. Query Captions or Ollama")
st.write("Search through generated captions or query the Ollama LLM directly.")

query_type = st.radio("Select Query Type", ("Caption Search", "Ollama Query"))
text_query = st.text_input("Enter your query here:")

if text_query:
    if query_type == "Caption Search":
        if 'indexing_case_dir' not in st.session_state or not st.session_state.indexing_case_dir:
            st.warning("Please build an index first to search captions.")
        else:
            st.write(f"Searching captions for: '{text_query}'")
            case_dir = st.session_state.indexing_case_dir
            # For simplicity, we'll do a basic text search on cached captions
            # A proper Whoosh index search would be more robust
            try:
                # This part needs to load all cached metadata and search
                # For now, let's assume metadata.pkl contains paths and captions
                # This is a simplified approach, a real Whoosh search would be better
                st.info("Performing basic caption search. For large datasets, a dedicated text index is recommended.")
                
                # Re-load image_paths to iterate and check cache
                with open(case_dir / "image_paths.pkl", "rb") as f:
                    indexed_image_paths = pickle.load(f)
                
                found_results = []
                for img_path in indexed_image_paths:
                    fp = fingerprint(Path(img_path))
                    cached_data = load_cache(fp)
                    if cached_data and "metadata" in cached_data and "caption" in cached_data["metadata"]:
                        caption = cached_data["metadata"]["caption"]
                        if text_query.lower() in caption.lower():
                            found_results.append({"path": img_path, "caption": caption})
                
                if found_results:
                    st.subheader("Matching Captions:")
                    for res in found_results:
                        st.write(f"**Image:** {Path(res['path']).name}")
                        st.write(f"**Caption:** {res['caption']}")
                        st.image(res['path'], width=100)
                else:
                    st.info("No matching captions found.")

            except Exception as e:
                st.error(f"Error during caption search: {e}")

    elif query_type == "Ollama Query":
        st.write(f"Querying Ollama with: '{text_query}'")
        if not ollama_installed():
            st.warning("Ollama is not installed or not in PATH. Cannot query LLM.")
        elif not model_available("llama2"): # Assuming llama2 is the model for general queries
            st.warning("Llama2 model not available. Please pull 'llama2' model for Ollama queries.")
        else:
            with st.spinner("Getting response from Ollama..."):
                try:
                    response = general_ollama_query(text_query)
                    st.write("---")
                    st.write(response)
                except Exception as e:
                    st.error(f"Error querying Ollama: {e}")
