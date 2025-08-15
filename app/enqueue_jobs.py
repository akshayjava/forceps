#!/usr/bin/env python3
"""
FORCEPS Job Enqueueing Script

Scans a directory for images and pushes batches of paths to a Redis queue.
"""
import argparse
import logging
import redis
import json
import yaml
from pathlib import Path
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from app.hashers import get_hashers

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def process_file(file_path, hashers):
    """Process a single file: find it, and compute all configured hashes."""
    try:
        all_hashes = {}
        for hasher in hashers:
            all_hashes.update(hasher.compute(file_path))
        return {"path": str(file_path), "hashes": all_hashes}
    except Exception as e:
        logger.warning(f"Could not process file {file_path}: {e}")
        return None

def discover_and_process_files(root_dir: str, hashers: list, max_workers: int) -> list[dict]:
    logger.info(f"Starting parallel file discovery and processing in '{root_dir}'...")
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".webp"}

    # Use a ThreadPoolExecutor to parallelize both walking and processing
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for root, _, files in os.walk(root_dir):
            for filename in files:
                if Path(filename).suffix.lower() in image_extensions:
                    file_path = Path(root) / filename
                    futures.append(executor.submit(process_file, file_path, hashers))

        results = []
        for i, future in enumerate(as_completed(futures)):
            if (i + 1) % 1000 == 0:
                logger.info(f"Processed {i + 1} files...")
            result = future.result()
            if result:
                results.append(result)

    return results

def main():
    parser = argparse.ArgumentParser(description="FORCEPS Job Enqueuer")
    parser.add_argument("--config", type=str, default="app/config.yaml", help="Path to the configuration file.")
    parser.add_argument("--mode", type=str, default="full", choices=["full", "triage"], help="Indexing mode.")
    parser.add_argument("--input_dir", type=str, default=None, help="Override input_dir from config")
    parser.add_argument("--job_batch_size", type=int, default=None, help="Override batch size for jobs enqueued")
    args = parser.parse_args()

    try:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        logger.error(f"Configuration file not found at: {args.config}")
        return
    except Exception as e:
        logger.error(f"Error reading configuration file: {e}")
        return

    cfg_redis = config['redis']
    cfg_data = config['data']
    # Optional override from CLI
    if args.input_dir:
        cfg_data['input_dir'] = args.input_dir
    cfg_perf = config['performance']['enqueuer']
    if args.job_batch_size:
        cfg_perf['job_batch_size'] = max(1, int(args.job_batch_size))

    logger.info(f"--- FORCEPS Job Enqueuer (Mode: {args.mode}) ---")

    # 1. Initialize hashers based on mode
    if args.mode == "triage":
        logger.info("Triage mode enabled: only perceptual hashes will be computed.")
        hashers_to_run = ["perceptual"]
    else:
        hashers_to_run = config.get('hashing', [])

    hashers = get_hashers(hashers_to_run)
    if not hashers:
        logger.warning("No hashers configured. No hashes will be computed.")

    # 2. Connect to Redis
    try:
        r = redis.Redis(host=cfg_redis['host'], port=cfg_redis['port'], db=0, decode_responses=True)
        r.ping()
        logger.info(f"Successfully connected to Redis at {cfg_redis['host']}:{cfg_redis['port']}")
    except redis.exceptions.ConnectionError as e:
        logger.error(f"Could not connect to Redis: {e}")
        return

    # 3. Scan for images and compute their hashes
    image_data = discover_and_process_files(cfg_data['input_dir'], hashers, cfg_perf['scan_max_workers'])
    logger.info(f"Found and processed {len(image_data)} total images.")

    # 4. Enqueue jobs in batches
    # Initialize progress stats in Redis
    try:
        r.set("forceps:stats:total_images", len(image_data))
        r.set("forceps:stats:embeddings_done", 0)
        r.set("forceps:stats:captions_done", 0)
    except Exception:
        pass

    jobs_enqueued = 0
    for i in range(0, len(image_data), cfg_perf['job_batch_size']):
        batch = image_data[i:i + cfg_perf['job_batch_size']]
        r.rpush(cfg_redis['job_queue'], json.dumps(batch))
        jobs_enqueued += 1

    logger.info(f"Enqueued {jobs_enqueued} jobs with a total of {len(image_data)} images to queue '{cfg_redis['job_queue']}'.")
    logger.info("--- Enqueueing complete ---")

if __name__ == "__main__":
    main()
