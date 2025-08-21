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
import subprocess
import uuid # For unique worker ID

# Configure logging for the script
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add app directory to sys.path to import modules
sys.path.insert(0, os.path.abspath('./app'))

# Import necessary modules from app
from app.hashers import get_hashers
from app.distributed_engine import OptimizedRedisClient, WorkerStats
from app.optimized_embeddings import OptimizedEmbeddingComputer, optimize_gpu_settings
from app.embeddings import load_models, compute_batch_embeddings
from app.utils import fingerprint, load_cache, save_cache, read_exif
from app.llm_ollama import ollama_installed, generate_caption_ollama, model_available
from app.engine import phase2_caption # Import phase2_caption directly

# --- Configuration Loading ---
def load_config(config_path="app/config.yaml"):
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        logger.error(f"Configuration file not found at: {config_path}")
        return None
    except Exception as e:
        logger.error(f"Error reading configuration file: {e}")
        return None

# --- Refactored Functions (without Streamlit calls) ---

def enqueue_jobs_programmatic_test(input_dir: str, config: dict):
    logger.info(f"Scanning for images in {input_dir} and enqueuing jobs...")
    cfg_redis = config['redis']
    cfg_perf = config['performance']['enqueuer']

    hashers_to_run = config.get('hashing', [])
    hashers = get_hashers(hashers_to_run)
    if not hashers:
        logger.warning("No hashers configured. No hashes will be computed.")
        return 0

    try:
        r = redis.Redis(host=cfg_redis['host'], port=cfg_redis['port'], db=0, decode_responses=True)
        r.ping()
        logger.info(f"Successfully connected to Redis at {cfg_redis['host']}:{cfg_redis['port']}")
    except redis.exceptions.ConnectionError as e:
        logger.error(f"Could not connect to Redis: {e}. Please ensure Redis is running.")
        return 0

    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp'}
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
            if (i + 1) % 100 == 0:
                logger.info(f"Scanned and processed {i + 1} files...")

    logger.info(f"Found and processed {len(image_data)} total images.")

    try:
        r.set("forceps:stats:total_images", len(image_data))
        r.set("forceps:stats:embeddings_done", 0)
        r.set("forceps:stats:captions_done", 0)
    except Exception as e:
        logger.warning(f"Failed to set Redis counters: {e}")

    jobs_enqueued = 0
    for i in range(0, len(image_data), cfg_perf['job_batch_size']):
        batch = image_data[i:i + cfg_perf['job_batch_size']]
        r.rpush(cfg_redis['job_queue'], json.dumps(batch))
        jobs_enqueued += 1

    logger.info(f"Enqueued {jobs_enqueued} jobs with a total of {len(image_data)} images to queue '{cfg_redis['job_queue']}'.")
    return len(image_data)

def build_index_programmatic_test(config: dict, case_name: str = None):
    logger.info("Building FAISS and Whoosh indexes...")
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
        logger.error(f"Could not connect to Redis: {e}. Please ensure Redis is running.")
        return None, []

    all_results = []
    while True:
        items = r.lrange(cfg_redis['results_queue'], 0, 99)
        if not items:
            break
        r.ltrim(cfg_redis['results_queue'], len(items), -1) # Remove consumed items
        for item in items:
            all_results.extend(json.loads(item))
        logger.info(f"Consumed {len(items)} results. Total embeddings so far: {len(all_results)}")
        time.sleep(0.01) # Small sleep to yield CPU

    if not all_results:
        logger.warning("No embeddings were found in the results queue. Index not built.")
        return None, []

    manifest_data = [{"path": res["path"], "hashes": res.get("hashes")} for res in all_results]

    combined_embs = np.array([res["combined_emb"] for res in all_results], dtype=np.float32)
    has_clip = "clip_emb" in all_results[0] if all_results else False
    clip_embs = None
    if has_clip:
        clip_embs = np.array([res["clip_emb"] for res in all_results if "clip_emb" in res], dtype=np.float32)
    n, d_comb = combined_embs.shape

    logger.info("Building vector (FAISS) index...")
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
    pq_m = faiss_args.pq_m
    quantizer = faiss.IndexFlatL2(d_final)
    cpu_index = faiss.IndexIVFPQ(quantizer, d_final, nlist, pq_m, 8)

    index_to_train = faiss.index_cpu_to_gpu(gpu_res, 0, cpu_index) if use_gpu else cpu_index
    train_data = pca_ret.apply_py(combined_embs[:faiss_args.train_samples]) if faiss_args.use_pca else combined_embs[:faiss_args.train_samples]
    if len(train_data) > 0:
        index_to_train.train(train_data)
    else:
        logger.warning("Not enough data to train FAISS index.")

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
        _, d_clip = clip_embs.shape
        clip_quantizer = faiss.IndexFlatL2(d_clip)
        clip_cpu_index = faiss.IndexIVFPQ(clip_quantizer, d_clip, nlist, pq_m, 8)
        clip_index_to_train = faiss.index_cpu_to_gpu(gpu_res, 0, clip_cpu_index) if use_gpu else clip_cpu_index
        if len(clip_embs[:faiss_args.train_samples]) > 0:
            clip_index_to_train.train(clip_embs[:faiss_args.train_samples])
        else:
            logger.warning("Not enough data to train CLIP FAISS index.")
        for off in range(0, n, add_batch_size):
            end = off + add_batch_size
            clip_index_to_train.add(clip_embs[off:end])
        index_clip = faiss.index_gpu_to_cpu(clip_index_to_train) if use_gpu else clip_index_to_train
        index_clip.nprobe = min(64, max(1, nlist // 16))

    logger.info("Vector index building complete.")

    logger.info("Saving FAISS indexes and manifest...")
    faiss.write_index(final_index_comb, str(case_output_dir / "image_index.faiss"))
    if index_clip:
        faiss.write_index(index_clip, str(case_output_dir / "clip.index"))
    if pca_ret:
        with open(case_output_dir / "pca.matrix.pkl", "wb") as f:
            pickle.dump(pca_ret, f)
    
    image_paths_only = [item['path'] for item in manifest_data]
    with open(case_output_dir / "image_paths.pkl", "wb") as f:
        pickle.dump(image_paths_only, f)
    
    metadata_data = {} # Placeholder, actual metadata would be more complex
    with open(case_output_dir / "metadata.pkl", "wb") as f:
        pickle.dump(metadata_data, f)

    with open(case_output_dir / "manifest.json", "w") as f:
        json.dump(manifest_data, f, indent=2)

    logger.info(f"Index building complete for case '{case_name}'. Saved to {case_output_dir}")
    return case_output_dir, image_paths_only

def run_captions_programmatic_test(image_paths: list, config: dict):
    logger.info("Starting caption generation (Phase 2)...")
    if not ollama_installed() or not model_available("llava"):
        logger.warning("Ollama or llava model not available. Skipping captioning. Please ensure Ollama is running and 'llava' model is pulled.")
        return

    max_workers = config["performance"]["worker"]["max_workers"]
    
    class Args:
        def __init__(self, mw: int):
            self.max_workers = mw
    
    phase2_caption(image_paths, Args(max_workers))
    logger.info("Caption generation complete.")

# --- Main Load Test Logic ---
def main():
    config = load_config()
    if not config:
        sys.exit(1)

    input_dir = "/Users/akshayjava/Downloads/Celebrity Faces Dataset"
    if not os.path.isdir(input_dir):
        logger.error(f"Input directory '{input_dir}' does not exist. Please provide a valid path.")
        sys.exit(1)

    cfg_redis = config['redis']
    try:
        r = redis.Redis(host=cfg_redis['host'], port=cfg_redis['port'], db=0, decode_responses=True)
        r.ping()
        logger.info("Successfully connected to Redis. Clearing queues...")
        r.delete(cfg_redis['job_queue'])
        r.delete(cfg_redis['results_queue'])
        r.delete("forceps:stats:total_images")
        r.delete("forceps:stats:embeddings_done")
        r.delete("forceps:stats:captions_done")
        logger.info("Redis queues cleared.")
    except redis.exceptions.ConnectionError as e:
        logger.error(f"Could not connect to Redis: {e}. Please ensure Redis is running.")
        sys.exit(1)

    total_images_processed = 0
    case_output_path = None
    indexed_image_paths = []

    logger.info("\n--- Starting Indexing Load Test ---")

    # Phase 1: Enqueueing Jobs
    start_time = time.time()
    total_images_found = enqueue_jobs_programmatic_test(input_dir, config)
    end_time = time.time()
    time_enqueue = end_time - start_time
    logger.info(f"Phase 1 (Enqueueing Jobs) completed in {time_enqueue:.2f} seconds.")
    logger.info(f"Total images found: {total_images_found}")

    if total_images_found == 0:
        logger.warning("No images found. Skipping further steps.")
        sys.exit(0)

    # Phase 2: Embedding Computation (Worker)
    worker_id = str(uuid.uuid4())
    worker_process = None
    start_time = time.time()
    try:
        logger.info(f"Starting worker process with ID: {worker_id}")
        # Construct the command to run the worker script
        worker_cmd = [
            sys.executable, # Use the same python interpreter
            os.path.join(os.path.abspath('./app'), 'optimized_worker.py'),
            "--worker_id", worker_id,
            "--config", "app/config.yaml" # Pass the config path
        ]
        # Start the worker as a subprocess
        worker_process = subprocess.Popen(worker_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        logger.info(f"Worker process started with PID: {worker_process.pid}")

        # Monitor Redis for completion
        initial_jobs_in_queue = r.llen(cfg_redis['job_queue'])
        total_embeddings_expected = int(r.get("forceps:stats:total_images") or 0)
        
        if total_embeddings_expected == 0:
            logger.warning("No embeddings expected. Worker might not have jobs to process.")

        while True:
            embeddings_done = int(r.get("forceps:stats:embeddings_done") or 0)
            jobs_remaining = r.llen(cfg_redis['job_queue'])
            logger.info(f"Embeddings done: {embeddings_done}/{total_embeddings_expected}, Jobs remaining: {jobs_remaining}")
            
            if embeddings_done >= total_embeddings_expected and jobs_remaining == 0:
                logger.info("All embeddings processed and jobs queue is empty. Terminating worker.")
                break
            
            # Check if worker process is still alive
            if worker_process.poll() is not None:
                logger.error(f"Worker process unexpectedly exited with code {worker_process.returncode}")
                stdout, stderr = worker_process.communicate()
                logger.error(f"Worker stdout:\n{stdout}")
                logger.error(f"Worker stderr:\n{stderr}")
                break # Exit loop if worker crashed

            time.sleep(5) # Poll every 5 seconds

    finally:
        if worker_process and worker_process.poll() is None:
            logger.info(f"Sending SIGTERM to worker process {worker_process.pid}")
            worker_process.terminate()
            try:
                worker_process.wait(timeout=10) # Give it some time to terminate
            except subprocess.TimeoutExpired:
                logger.warning(f"Worker process {worker_process.pid} did not terminate gracefully. Sending SIGKILL.")
                worker_process.kill()
        
        end_time = time.time()
        time_worker = end_time - start_time
        embeddings_processed = int(r.get("forceps:stats:embeddings_done") or 0)
        logger.info(f"Phase 2 (Embedding Computation) completed in {time_worker:.2f} seconds.")
        logger.info(f"Embeddings processed by worker: {embeddings_processed}")
        if time_worker > 0:
            logger.info(f"Worker throughput: {embeddings_processed / time_worker:.2f} images/second.")

    # Phase 3: Index Building
    start_time = time.time()
    case_output_path, indexed_image_paths = build_index_programmatic_test(config)
    end_time = time.time()
    time_build_index = end_time - start_time
    logger.info(f"Phase 3 (Index Building) completed in {time_build_index:.2f} seconds.")
    if case_output_path and len(indexed_image_paths) > 0 and time_build_index > 0:
        logger.info(f"Index building throughput: {len(indexed_image_paths) / time_build_index:.2f} images/second.")

    # Phase 4: Caption Generation
    if indexed_image_paths:
        start_time = time.time()
        run_captions_programmatic_test(indexed_image_paths, config)
        end_time = time.time()
        time_captions = end_time - start_time
        logger.info(f"Phase 4 (Caption Generation) completed in {time_captions:.2f} seconds.")
        if time_captions > 0:
            logger.info(f"Captioning throughput: {len(indexed_image_paths) / time_captions:.2f} images/second.")
    else:
        logger.warning("No indexed images found for captioning. Skipping Phase 4.")

    logger.info("\n--- Load Test Summary ---")
    total_time = time_enqueue + time_worker + time_build_index + (time_captions if indexed_image_paths else 0)
    logger.info(f"Total images processed: {total_images_found}")
    logger.info(f"Total time for all phases: {total_time:.2f} seconds.")
    if total_time > 0:
        logger.info(f"Overall throughput: {total_images_found / total_time:.2f} images/second.")
    logger.info("---------------------------")

if __name__ == "__main__":
    main()