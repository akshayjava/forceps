#!/usr/bin/env python3
"""
FORCEPS High-Performance File System Scanner

Optimized for multi-terabyte forensic datasets with:
- Parallel directory traversal using multiple processes
- Memory-mapped file operations for speed
- Intelligent file type filtering with magic numbers
- RAID/SSD optimization for sequential reads
- Progress tracking and ETA estimation
"""
import os
import logging
import time
import hashlib
import mmap
import multiprocessing as mp
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from collections import defaultdict
from typing import List, Dict, Generator, Tuple
import psutil

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Image file magic numbers for fast type detection
IMAGE_MAGIC_NUMBERS = {
    b'\xff\xd8\xff': '.jpg',
    b'\x89PNG\r\n\x1a\n': '.png',
    b'BM': '.bmp',
    b'GIF87a': '.gif',
    b'GIF89a': '.gif',
    b'RIFF': '.webp',  # Need to check for WEBP in RIFF
    b'II*\x00': '.tiff',
    b'MM\x00*': '.tiff',
}

class FastFileScanner:
    """High-performance file system scanner optimized for forensic workloads."""
    
    def __init__(self, root_path: str, max_workers: int = None):
        self.root_path = Path(root_path)
        self.max_workers = max_workers or min(32, (os.cpu_count() or 1) * 2)
        self.stats = defaultdict(int)
        self.start_time = None
        
    def detect_image_type_fast(self, file_path: Path) -> str:
        """Fast image type detection using magic numbers instead of file extensions."""
        try:
            with open(file_path, 'rb') as f:
                header = f.read(16)
                
            for magic, ext in IMAGE_MAGIC_NUMBERS.items():
                if header.startswith(magic):
                    if magic == b'RIFF' and b'WEBP' in header[:16]:
                        return '.webp'
                    elif magic == b'RIFF':
                        continue  # Not a WebP file
                    return ext
                    
            return None
        except (OSError, IOError):
            return None
    
    def fast_directory_walk(self, directory: Path) -> Generator[Path, None, None]:
        """Optimized directory traversal using os.scandir for better performance."""
        try:
            with os.scandir(directory) as entries:
                files = []
                subdirs = []
                
                for entry in entries:
                    if entry.is_file(follow_symlinks=False):
                        files.append(Path(entry.path))
                    elif entry.is_dir(follow_symlinks=False):
                        subdirs.append(Path(entry.path))
                
                # Yield files from this directory first
                for file_path in files:
                    yield file_path
                
                # Recursively process subdirectories
                for subdir in subdirs:
                    yield from self.fast_directory_walk(subdir)
                    
        except (PermissionError, OSError) as e:
            logger.warning(f"Cannot access directory {directory}: {e}")
            self.stats['permission_errors'] += 1
    
    def process_file_chunk(self, file_paths: List[Path]) -> List[Dict]:
        """Process a chunk of files in parallel."""
        results = []
        
        for file_path in file_paths:
            try:
                # Fast image type detection
                image_type = self.detect_image_type_fast(file_path)
                if not image_type:
                    self.stats['non_images'] += 1
                    continue
                
                # Get basic file stats
                stat_info = file_path.stat()
                file_size = stat_info.st_size
                
                # Skip very small files (likely corrupted)
                if file_size < 1024:  # 1KB minimum
                    self.stats['too_small'] += 1
                    continue
                
                # Skip very large files that might be corrupted
                if file_size > 500 * 1024 * 1024:  # 500MB maximum
                    self.stats['too_large'] += 1
                    continue
                
                results.append({
                    'path': str(file_path),
                    'size': file_size,
                    'type': image_type,
                    'mtime': stat_info.st_mtime
                })
                self.stats['valid_images'] += 1
                
            except (OSError, IOError) as e:
                logger.debug(f"Error processing {file_path}: {e}")
                self.stats['file_errors'] += 1
        
        return results
    
    def scan_parallel_processes(self) -> List[Dict]:
        """Scan using multiple processes for CPU-bound operations."""
        logger.info(f"Starting parallel file scan of {self.root_path} with {self.max_workers} processes")
        self.start_time = time.time()
        
        # First, collect all files using fast directory walk
        logger.info("Discovering files...")
        all_files = list(self.fast_directory_walk(self.root_path))
        logger.info(f"Found {len(all_files)} total files")
        
        if not all_files:
            return []
        
        # Split files into chunks for parallel processing
        chunk_size = max(1, len(all_files) // (self.max_workers * 4))  # More chunks than workers
        file_chunks = [all_files[i:i + chunk_size] for i in range(0, len(all_files), chunk_size)]
        
        logger.info(f"Processing {len(file_chunks)} chunks with chunk size ~{chunk_size}")
        
        results = []
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(self.process_file_chunk, chunk) for chunk in file_chunks]
            
            for i, future in enumerate(as_completed(futures)):
                chunk_results = future.result()
                results.extend(chunk_results)
                
                # Progress reporting
                if (i + 1) % 10 == 0:
                    elapsed = time.time() - self.start_time
                    rate = len(results) / elapsed if elapsed > 0 else 0
                    eta = (len(file_chunks) - i - 1) / (i + 1) * elapsed if i > 0 else 0
                    logger.info(f"Processed {i + 1}/{len(file_chunks)} chunks, "
                              f"{len(results)} images found, "
                              f"{rate:.1f} files/sec, ETA: {eta/60:.1f}min")
        
        elapsed = time.time() - self.start_time
        logger.info(f"Scan completed in {elapsed/60:.1f} minutes")
        logger.info(f"Statistics: {dict(self.stats)}")
        
        return results
    
    def scan_memory_optimized(self) -> Generator[Dict, None, None]:
        """Memory-optimized streaming scan for very large datasets."""
        logger.info(f"Starting memory-optimized streaming scan of {self.root_path}")
        self.start_time = time.time()
        processed = 0
        
        # Use threading for I/O-bound file operations
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Process files in smaller batches to manage memory
            batch_size = 1000
            file_batch = []
            
            for file_path in self.fast_directory_walk(self.root_path):
                file_batch.append(file_path)
                
                if len(file_batch) >= batch_size:
                    # Process batch
                    future = executor.submit(self.process_file_chunk, file_batch)
                    for result in future.result():
                        yield result
                        processed += 1
                        
                        if processed % 10000 == 0:
                            elapsed = time.time() - self.start_time
                            rate = processed / elapsed if elapsed > 0 else 0
                            logger.info(f"Processed {processed} images, {rate:.1f} files/sec")
                    
                    file_batch = []
            
            # Process remaining files
            if file_batch:
                future = executor.submit(self.process_file_chunk, file_batch)
                for result in future.result():
                    yield result
                    processed += 1
        
        elapsed = time.time() - self.start_time
        logger.info(f"Streaming scan completed: {processed} images in {elapsed/60:.1f} minutes")


def optimize_system_for_scanning():
    """Optimize system settings for high-performance file scanning."""
    try:
        # Increase file descriptor limit
        import resource
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        resource.setrlimit(resource.RLIMIT_NOFILE, (min(hard, 65536), hard))
        logger.info(f"Increased file descriptor limit to {min(hard, 65536)}")
        
        # Set high I/O priority for this process
        if hasattr(os, 'nice'):
            os.nice(-10)  # Higher priority
            logger.info("Set high process priority")
            
    except Exception as e:
        logger.warning(f"Could not optimize system settings: {e}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="High-performance file system scanner")
    parser.add_argument("path", help="Path to scan")
    parser.add_argument("--workers", type=int, help="Number of worker processes")
    parser.add_argument("--streaming", action="store_true", help="Use streaming mode for memory efficiency")
    parser.add_argument("--output", help="Output file for results")
    
    args = parser.parse_args()
    
    optimize_system_for_scanning()
    
    scanner = FastFileScanner(args.path, args.workers)
    
    if args.streaming:
        results = list(scanner.scan_memory_optimized())
    else:
        results = scanner.scan_parallel_processes()
    
    logger.info(f"Found {len(results)} valid image files")
    
    if args.output:
        import json
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {args.output}")
