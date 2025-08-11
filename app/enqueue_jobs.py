#!/usr/bin/env python3
"""
FORCEPS Job Enqueueing Script

Scans a directory for images and pushes batches of paths to a Redis queue.
"""
import argparse
import logging
import redis
import json
from pathlib import Path
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# This function is copied from engine.py to make the script self-contained
def scan_images(root_dir: str, max_workers: int) -> list[str]:
    logger.info(f"Starting parallel scan in '{root_dir}' with {max_workers} workers.")
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".webp"}
    image_paths = []

    def scan_subdirectory(directory):
        sub_paths = []
        try:
            for entry in os.scandir(directory):
                if entry.is_file() and Path(entry.name).suffix.lower() in image_extensions:
                    sub_paths.append(entry.path)
        except (PermissionError, FileNotFoundError):
            logger.warning(f"Could not access directory: {directory}")
        return sub_paths

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for root, _, _ in os.walk(root_dir):
            futures.append(executor.submit(scan_subdirectory, root))

        for i, future in enumerate(as_completed(futures)):
            if (i + 1) % 1000 == 0:
                logger.info(f"Scanned {i + 1} directories...")
            image_paths.extend(future.result())

    seen = set()
    unique_paths = [p for p in image_paths if not (p in seen or seen.add(p))]
    return unique_paths

def main():
    parser = argparse.ArgumentParser(description="FORCEPS Job Enqueuer")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory of images to scan.")
    parser.add_argument("--redis_host", type=str, default="localhost", help="Redis server host.")
    parser.add_argument("--redis_port", type=int, default=6379, help="Redis server port.")
    parser.add_argument("--queue_name", type=str, default="forceps:job_queue", help="Name of the Redis list to use as a queue.")
    parser.add_argument("--batch_size", type=int, default=1024, help="Number of image paths per job.")
    parser.add_argument("--max_workers", type=int, default=8, help="Max workers for scanning.")

    args = parser.parse_args()

    logger.info("--- FORCEPS Job Enqueuer ---")

    try:
        r = redis.Redis(host=args.redis_host, port=args.redis_port, db=0, decode_responses=True)
        r.ping()
        logger.info(f"Successfully connected to Redis at {args.redis_host}:{args.redis_port}")
    except redis.exceptions.ConnectionError as e:
        logger.error(f"Could not connect to Redis: {e}")
        return

    # Scan for images
    image_paths = scan_images(args.input_dir, args.max_workers)
    logger.info(f"Found {len(image_paths)} total images.")

    # Enqueue jobs in batches
    jobs_enqueued = 0
    for i in range(0, len(image_paths), args.batch_size):
        batch = image_paths[i:i + args.batch_size]
        # Use JSON to store the list as a single queue item
        r.rpush(args.queue_name, json.dumps(batch))
        jobs_enqueued += 1

    logger.info(f"Enqueued {jobs_enqueued} jobs with a total of {len(image_paths)} images.")
    logger.info("--- Enqueueing complete ---")

if __name__ == "__main__":
    main()
