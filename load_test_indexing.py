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
import urllib.request
import zipfile
import tempfile
import shutil

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
# Removed caption generation imports to reduce cost and complexity

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
    
    # Memory-efficient file processing with generator
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
        file_count = 0
        
        # First pass: collect all image files
        for root, _, files in os.walk(input_dir):
            for filename in files:
                if Path(filename).suffix.lower() in image_extensions:
                    file_path = Path(root) / filename
                    futures.append(executor.submit(process_single_file, file_path, hashers))
                    file_count += 1
                    
                    # Process in batches to avoid memory issues
                    if len(futures) >= 1000:  # Process in batches of 1000
                        for future in as_completed(futures):
                            result = future.result()
                            if result:
                                image_data.append(result)
                        futures = []  # Clear processed futures
                        logger.info(f"Processed batch of 1000 files. Total processed: {len(image_data)}")

        # Process remaining futures
        for future in as_completed(futures):
            result = future.result()
            if result:
                image_data.append(result)

    logger.info(f"Found and processed {len(image_data)} total images.")

    try:
        r.set("forceps:stats:total_images", len(image_data))
        r.set("forceps:stats:embeddings_done", 0)
        r.set("forceps:stats:captions_done", 0)
    except Exception as e:
        logger.warning(f"Failed to set Redis counters: {e}")

    jobs_enqueued = 0
    # Use pipeline for more efficient Redis operations
    pipeline = r.pipeline()
    
    for i in range(0, len(image_data), cfg_perf['job_batch_size']):
        batch = image_data[i:i + cfg_perf['job_batch_size']]
        pipeline.rpush(cfg_redis['job_queue'], json.dumps(batch))
        jobs_enqueued += 1
    
    # Execute all Redis operations in a single pipeline
    pipeline.execute()
    
    logger.info(f"Enqueued {jobs_enqueued} jobs with a total of {len(image_data)} images to queue '{cfg_redis['job_queue']}' using pipeline.")
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
    # Use pipeline for more efficient Redis operations
    pipeline = r.pipeline()
    
    while True:
        items = r.lrange(cfg_redis['results_queue'], 0, 199)  # Larger batch size
        if not items:
            break
        
        # Use pipeline for atomic trim operation
        pipeline.ltrim(cfg_redis['results_queue'], len(items), -1)
        pipeline.execute()
        
        for item in items:
            all_results.extend(json.loads(item))
        logger.info(f"Consumed {len(items)} results. Total embeddings so far: {len(all_results)}")
        time.sleep(0.005)  # Reduced sleep for better performance

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

    # For large datasets, use optimized IVF index with PCA
    if n < 100:
        logger.info(f"Small dataset ({n} images), using simple flat index")
        index_to_train = faiss.IndexFlatL2(d_final)
        if use_gpu:
            index_to_train = faiss.index_cpu_to_gpu(gpu_res, 0, index_to_train)
    else:
        # Use optimized parameters for larger datasets
        nlist = min(faiss_args.ivf_nlist, max(1, n // 100))
        pq_m = min(faiss_args.pq_m, d_final // 4)  # Ensure pq_m doesn't exceed dimension
        
        logger.info(f"Large dataset ({n} images), using IVF index with nlist={nlist}, pq_m={pq_m}")
        quantizer = faiss.IndexFlatL2(d_final)
        cpu_index = faiss.IndexIVFPQ(quantizer, d_final, nlist, pq_m, 8)
        index_to_train = faiss.index_cpu_to_gpu(gpu_res, 0, cpu_index) if use_gpu else cpu_index
        
        # Use more training samples for better index quality
        train_samples = min(faiss_args.train_samples, n)
        train_data = pca_ret.apply_py(combined_embs[:train_samples]) if faiss_args.use_pca else combined_embs[:train_samples]
        
        if len(train_data) > 0:
            logger.info(f"Training FAISS index with {len(train_data)} samples...")
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
        if n < 100:
            logger.info(f"Small dataset ({n} images), using simple flat index for CLIP")
            clip_index_to_train = faiss.IndexFlatL2(d_clip)
            if use_gpu:
                clip_index_to_train = faiss.index_cpu_to_gpu(gpu_res, 0, clip_index_to_train)
        else:
            # Use optimized parameters for larger CLIP datasets
            clip_nlist = min(faiss_args.ivf_nlist, max(1, n // 100))
            clip_pq_m = min(faiss_args.pq_m, d_clip // 4)
            
            logger.info(f"Large CLIP dataset ({n} images), using IVF index with nlist={clip_nlist}, pq_m={clip_pq_m}")
            clip_quantizer = faiss.IndexFlatL2(d_clip)
            clip_cpu_index = faiss.IndexIVFPQ(clip_quantizer, d_clip, clip_nlist, clip_pq_m, 8)
            clip_index_to_train = faiss.index_cpu_to_gpu(gpu_res, 0, clip_cpu_index) if use_gpu else clip_cpu_index
            
            # Use more training samples for better CLIP index quality
            clip_train_samples = min(faiss_args.train_samples, n)
            if len(clip_embs[:clip_train_samples]) > 0:
                logger.info(f"Training CLIP FAISS index with {clip_train_samples} samples...")
                clip_index_to_train.train(clip_embs[:clip_train_samples])
            else:
                logger.warning("Not enough data to train CLIP FAISS index.")
        
        for off in range(0, n, add_batch_size):
            end = off + add_batch_size
            clip_index_to_train.add(clip_embs[off:end])
        index_clip = faiss.index_gpu_to_cpu(clip_index_to_train) if use_gpu else clip_index_to_train
        if n >= 100:  # Only set nprobe for IVF indexes
            index_clip.nprobe = min(64, max(1, nlist // 16))

    logger.info("Vector index building complete.")

    logger.info("Saving FAISS indexes and manifest...")
    faiss.write_index(final_index_comb, str(case_output_dir / "image_index.faiss"))
    if index_clip:
        faiss.write_index(index_clip, str(case_output_dir / "clip.index"))
    if pca_ret:
        try:
            # Save PCA matrix using FAISS's built-in save method instead of pickle
            pca_path = case_output_dir / "pca.matrix.faiss"
            faiss.write_index(pca_ret, str(pca_path))
            logger.info(f"Saved PCA matrix to {pca_path}")
        except Exception as e:
            logger.warning(f"Failed to save PCA matrix: {e}")
            # Fallback: try to save just the transformation matrix
            try:
                pca_data = {
                    'd_in': pca_ret.d_in,
                    'd_out': pca_ret.d_out,
                    'is_trained': pca_ret.is_trained,
                    'eigenvalues': pca_ret.eigenvalues,
                    'PC': pca_ret.PC
                }
                with open(case_output_dir / "pca.matrix.pkl", "wb") as f:
                    pickle.dump(pca_data, f)
                logger.info("Saved PCA matrix data as fallback")
            except Exception as e2:
                logger.warning(f"Failed to save PCA matrix data: {e2}")
    
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

# Caption generation removed to reduce cost and complexity

# --- Dataset Download Functions ---
def download_benchmark_dataset(target_dir="./demo_images", min_images=50):
    """Download a benchmark dataset for image similarity testing."""
    logger.info(f"Checking if benchmark dataset is needed in {target_dir}")
    
    # Check if we already have enough diverse images
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp'}
    existing_images = []
    
    if os.path.exists(target_dir):
        for root, _, files in os.walk(target_dir):
            for filename in files:
                if Path(filename).suffix.lower() in image_extensions:
                    existing_images.append(os.path.join(root, filename))
    
    if len(existing_images) >= min_images:
        logger.info(f"Already have {len(existing_images)} images, no need to download dataset")
        return target_dir
    
    logger.info(f"Only {len(existing_images)} images found, downloading benchmark dataset...")
    
    # Create target directory if it doesn't exist
    os.makedirs(target_dir, exist_ok=True)
    
    # Try to download from a public dataset first
    if not download_public_dataset(target_dir, min_images):
        # Fallback to synthetic images if download fails
        logger.info("Public dataset download failed, creating synthetic images...")
        create_synthetic_images(target_dir, min_images)
    
    # Also try to copy existing images from the project if available
    copy_existing_project_images(target_dir)
    
    return target_dir

def download_public_dataset(target_dir, min_images):
    """Download a comprehensive image dataset for benchmarking."""
    try:
        logger.info("Attempting to download a comprehensive image dataset...")
        
        # Try to download a smaller, manageable dataset first
        # Using smaller datasets that are good for testing
        dataset_urls = [
            # CIFAR-10 dataset (smaller, good for testing)
            "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz",
            # Oxford 102 Flower Dataset (smaller, diverse)
            "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz",
            # Caltech 101 dataset (smaller, diverse)
            "http://www.vision.caltech.edu/Image_Datasets/Caltech101/101_ObjectCategories.tar.gz",
            # MNIST dataset (very small, good for testing)
            "https://storage.googleapis.com/learning-datasets/mnist/train-images-idx3-ubyte.gz"
        ]
        
        for dataset_url in dataset_urls:
            try:
                logger.info(f"Trying to download dataset from: {dataset_url}")
                
                # Extract dataset name from URL
                dataset_name = dataset_url.split('/')[-1].split('.')[0]
                archive_path = os.path.join(target_dir, f"{dataset_name}.archive")
                
                # Download the archive
                logger.info(f"Downloading {dataset_name}...")
                urllib.request.urlretrieve(dataset_url, archive_path)
                
                # Extract the archive
                logger.info(f"Extracting {dataset_name}...")
                if archive_path.endswith('.tar.gz') or archive_path.endswith('.tgz'):
                    import tarfile
                    with tarfile.open(archive_path, 'r:gz') as tar:
                        tar.extractall(target_dir)
                elif archive_path.endswith('.zip'):
                    with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                        zip_ref.extractall(target_dir)
                
                # Clean up the archive
                os.remove(archive_path)
                
                # Count extracted images
                image_count = count_images_in_directory(target_dir)
                logger.info(f"Successfully extracted {image_count} images from {dataset_name}")
                
                if image_count >= min_images:
                    return True
                    
            except Exception as e:
                logger.warning(f"Failed to download {dataset_url}: {e}")
                continue
        
        # If all downloads fail, create synthetic images
        logger.warning("All dataset downloads failed, will create synthetic images")
        return False
        
    except Exception as e:
        logger.warning(f"Failed to download any dataset: {e}")
        return False

def count_images_in_directory(directory):
    """Count the number of image files in a directory and subdirectories."""
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp'}
    count = 0
    
    for root, _, files in os.walk(directory):
        for filename in files:
            if Path(filename).suffix.lower() in image_extensions:
                count += 1
    
    return count

def create_synthetic_images(target_dir, count):
    """Create synthetic test images for additional variety."""
    if count <= 0:
        return
    
    logger.info(f"Creating {count} synthetic test images...")
    
    # Create simple synthetic images using PIL
    for i in range(count):
        try:
            # Create images with different patterns and complexity
            img_size = 256
            img = Image.new('RGB', (img_size, img_size), color=(0, 0, 0))
            
            from PIL import ImageDraw
            draw = ImageDraw.Draw(img)
            
            # Create different types of synthetic images
            pattern_type = i % 5
            
            if pattern_type == 0:
                # Geometric patterns
                for j in range(5):
                    x1 = (i * 20 + j * 40) % img_size
                    y1 = (i * 15 + j * 30) % img_size
                    x2 = (x1 + 60) % img_size
                    y2 = (y1 + 60) % img_size
                    color = ((i * 30 + j * 50) % 255, (i * 40 + j * 60) % 255, (i * 50 + j * 70) % 255)
                    draw.rectangle([x1, y1, x2, y2], fill=color)
                    
            elif pattern_type == 1:
                # Circular patterns
                for j in range(8):
                    x = (i * 25 + j * 30) % img_size
                    y = (i * 20 + j * 25) % img_size
                    radius = 15 + (i * 3 + j * 5) % 25
                    color = ((i * 25 + j * 30) % 255, (i * 35 + j * 40) % 255, (i * 45 + j * 50) % 255)
                    draw.ellipse([x-radius, y-radius, x+radius, y+radius], fill=color)
                    
            elif pattern_type == 2:
                # Line patterns
                for j in range(10):
                    x1 = (i * 15 + j * 25) % img_size
                    y1 = (i * 10 + j * 20) % img_size
                    x2 = (x1 + 80) % img_size
                    y2 = (y1 + 80) % img_size
                    color = ((i * 20 + j * 25) % 255, (i * 30 + j * 35) % 255, (i * 40 + j * 45) % 255)
                    draw.line([x1, y1, x2, y2], fill=color, width=3)
                    
            elif pattern_type == 3:
                # Checkerboard pattern
                square_size = 16
                for x in range(0, img_size, square_size):
                    for y in range(0, img_size, square_size):
                        if ((x // square_size) + (y // square_size) + i) % 2 == 0:
                            color = ((i * 20 + x) % 255, (i * 30 + y) % 255, (i * 40) % 255)
                            draw.rectangle([x, y, x + square_size, y + square_size], fill=color)
                            
            else:
                # Gradient pattern
                for x in range(img_size):
                    for y in range(img_size):
                        r = int((x / img_size) * 255)
                        g = int((y / img_size) * 255)
                        b = int(((x + y) / (2 * img_size)) * 255)
                        color = ((r + i * 10) % 255, (g + i * 15) % 255, (b + i * 20) % 255)
                        draw.point([x, y], fill=color)
            
            # Save the image
            filename = f"synthetic_{i:03d}.png"
            filepath = os.path.join(target_dir, filename)
            img.save(filepath)
            
        except Exception as e:
            logger.warning(f"Failed to create synthetic image {i}: {e}")
            continue

def copy_existing_project_images(target_dir):
    """Copy existing images from the project directory for additional variety."""
    try:
        # Look for images in common project directories
        project_image_dirs = [
            "venv_forceps/images",
            "venv_forceps/lib/python3.9/site-packages/streamlit/static",
            "venv_forceps/lib/python3.9/site-packages/streamlit/static/static/media",
            "demo_images"  # Don't copy from self
        ]
        
        copied_count = 0
        for source_dir in project_image_dirs:
            if os.path.exists(source_dir) and source_dir != target_dir:
                logger.info(f"Looking for images in {source_dir}")
                
                for root, _, files in os.walk(source_dir):
                    for filename in files:
                        if Path(filename).suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp'}:
                            source_path = os.path.join(root, filename)
                            target_filename = f"project_{copied_count:03d}_{filename}"
                            target_path = os.path.join(target_dir, target_filename)
                            
                            if not os.path.exists(target_path):
                                try:
                                    shutil.copy2(source_path, target_path)
                                    copied_count += 1
                                    logger.info(f"Copied {filename} to {target_filename}")
                                    
                                    if copied_count >= 20:  # Limit the number of copied images
                                        break
                                except Exception as e:
                                    logger.warning(f"Failed to copy {filename}: {e}")
                    
                    if copied_count >= 20:
                        break
                        
                if copied_count >= 20:
                    break
        
        if copied_count > 0:
            logger.info(f"Copied {copied_count} existing project images")
        
    except Exception as e:
        logger.warning(f"Failed to copy existing project images: {e}")

# --- Main Load Test Logic ---
def main():
    config = load_config()
    if not config:
        sys.exit(1)

    # Ensure we have a good benchmark dataset
    input_dir = download_benchmark_dataset("./demo_images", min_images=500)  # Increased to 500 images for large dataset testing
    
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

    # Phase 2: Embedding Computation (Multiple Workers)
    start_time = time.time()
    worker_processes = []
    max_workers = config['performance']['worker']['max_workers']
    
    try:
        logger.info(f"Starting {max_workers} worker processes for parallel processing...")
        
        # Start multiple worker processes
        for i in range(max_workers):
            worker_id = str(uuid.uuid4())
            worker_cmd = [
                sys.executable,
                os.path.join(os.path.abspath('./app'), 'optimized_worker.py'),
                "--worker_id", worker_id,
                "--config", "app/config.yaml"
            ]
            
            worker_env = os.environ.copy()
            if 'PYTHONPATH' in worker_env:
                worker_env['PYTHONPATH'] = os.path.abspath('.') + os.pathsep + worker_env['PYTHONPATH']
            else:
                worker_env['PYTHONPATH'] = os.path.abspath('.')
            
            # Start worker process
            worker_process = subprocess.Popen(
                worker_cmd, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE, 
                text=True, 
                cwd=os.getcwd(), 
                env=worker_env
            )
            worker_processes.append(worker_process)
            logger.info(f"Worker {i+1}/{max_workers} started with PID: {worker_process.pid}")

        # Monitor Redis for completion with all workers
        total_embeddings_expected = int(r.get("forceps:stats:total_images") or 0)
        
        if total_embeddings_expected == 0:
            logger.warning("No embeddings expected. Workers might not have jobs to process.")

        # Progress tracking for large datasets
        last_progress_time = time.time()
        progress_interval = 5  # More frequent updates for better monitoring

        while True:
            embeddings_done = int(r.get("forceps:stats:embeddings_done") or 0)
            jobs_remaining = r.llen(cfg_redis['job_queue'])
            
            current_time = time.time()
            if current_time - last_progress_time >= progress_interval:
                progress_percent = (embeddings_done / total_embeddings_expected * 100) if total_embeddings_expected > 0 else 0
                active_workers = sum(1 for p in worker_processes if p.poll() is None)
                logger.info(f"Progress: {embeddings_done}/{total_embeddings_expected} ({progress_percent:.1f}%) - Jobs remaining: {jobs_remaining} - Active workers: {active_workers}")
                last_progress_time = current_time
            
            # Check for completion: all embeddings processed AND job queue is empty
            if embeddings_done >= total_embeddings_expected and jobs_remaining == 0:
                logger.info("All embeddings processed and jobs queue is empty. Terminating workers.")
                break
            
            # Check if any worker process has crashed
            crashed_workers = [i for i, p in enumerate(worker_processes) if p.poll() is not None]
            if crashed_workers:
                for i in crashed_workers:
                    worker_process = worker_processes[i]
                    logger.error(f"Worker {i+1} unexpectedly exited with code {worker_process.returncode}")
                    stdout, stderr = worker_process.communicate()
                    logger.error(f"Worker {i+1} stdout:\n{stdout}")
                    logger.error(f"Worker {i+1} stderr:\n{stderr}")
                # Continue with remaining workers

            time.sleep(1)  # Poll more frequently for better responsiveness

    finally:
        # Terminate all worker processes
        for i, worker_process in enumerate(worker_processes):
            if worker_process.poll() is None:
                logger.info(f"Sending SIGTERM to worker {i+1} (PID: {worker_process.pid})")
                worker_process.terminate()
                try:
                    worker_process.wait(timeout=5)  # Shorter timeout for faster cleanup
                except subprocess.TimeoutExpired:
                    logger.warning(f"Worker {i+1} (PID: {worker_process.pid}) did not terminate gracefully. Sending SIGKILL.")
                    worker_process.kill()
        
        end_time = time.time()
        time_worker = end_time - start_time
        embeddings_processed = int(r.get("forceps:stats:embeddings_done") or 0)
        logger.info(f"Phase 2 (Embedding Computation) completed in {time_worker:.2f} seconds.")
        logger.info(f"Embeddings processed by {max_workers} workers: {embeddings_processed}")
        if time_worker > 0:
            logger.info(f"Worker throughput: {embeddings_processed / time_worker:.2f} images/second.")
            logger.info(f"Per-worker throughput: {embeddings_processed / time_worker / max_workers:.2f} images/second/worker")

    # Phase 3: Index Building
    start_time = time.time()
    case_output_path, indexed_image_paths = build_index_programmatic_test(config)
    end_time = time.time()
    time_build_index = end_time - start_time
    logger.info(f"Phase 3 (Index Building) completed in {time_build_index:.2f} seconds.")
    if case_output_path and len(indexed_image_paths) > 0 and time_build_index > 0:
        logger.info(f"Index building throughput: {len(indexed_image_paths) / time_build_index:.2f} images/second.")

    # Phase 4: Caption Generation - REMOVED to reduce cost and complexity
    logger.info("Caption generation phase skipped to reduce cost and complexity.")

    logger.info("\n--- Load Test Summary ---")
    total_time = time_enqueue + time_worker + time_build_index
    logger.info(f"Total images processed: {total_images_found}")
    logger.info(f"Total time for all phases: {total_time:.2f} seconds.")
    if total_time > 0:
        logger.info(f"Overall throughput: {total_images_found / total_time:.2f} images/second.")
    logger.info("---------------------------")

if __name__ == "__main__":
    main()
