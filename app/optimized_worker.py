#!/usr/bin/env python3
"""
FORCEPS Optimized Worker Process

High-performance worker for distributed processing:
- Integrates with optimized embedding computation
- Uses enhanced Redis client with compression
- Implements worker health monitoring and reporting
- Handles graceful shutdown and error recovery
- Memory-efficient processing with backpressure control
"""
import os
import sys
import time
import logging
import signal
import argparse
import yaml
import psutil
import torch
from typing import Dict, List, Any
from dataclasses import asdict
from app.distributed_engine import OptimizedRedisClient, WorkerStats
from app.optimized_embeddings import OptimizedEmbeddingComputer, optimize_gpu_settings
from app.hashers import get_hashers

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class OptimizedWorker:
    """High-performance worker with optimized embedding computation."""
    
    def __init__(self, worker_id: str, config: Dict[str, Any]):
        self.worker_id = worker_id
        self.config = config
        self.running = False
        
        # Initialize Redis client
        redis_config = config['redis']
        self.redis_client = OptimizedRedisClient(
            host=redis_config['host'],
            port=redis_config['port'],
            compression_level=config.get('performance', {}).get('compression_level', 1)
        )
        
        # Initialize embedding computer
        worker_config = config.get('performance', {}).get('worker', {})
        self.embedding_computer = OptimizedEmbeddingComputer(
            max_batch_size=worker_config.get('batch_size', 128),
            use_mixed_precision=worker_config.get('use_mixed_precision', True),
            enable_cuda_streams=worker_config.get('enable_cuda_streams', True)
        )
        
        # Initialize hashers (only for triage mode if needed)
        self.hashers = []
        hashing_config = config.get('hashing', [])
        if hashing_config:
            self.hashers = get_hashers(hashing_config)
        
        # Worker statistics
        self.stats = WorkerStats(worker_id=worker_id)
        
        # Queues
        self.job_queue = redis_config['job_queue']
        self.results_queue = redis_config['results_queue']
        
        # Signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        logger.info(f"Initialized worker {worker_id}")
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        logger.info(f"Worker {self.worker_id} received signal {signum}, shutting down...")
        self.stop()
        sys.exit(0)
    
    def process_job(self, job_data: List[Dict]) -> List[Dict]:
        """Process a job containing multiple images."""
        start_time = time.time()
        
        try:
            # Extract image paths from job data
            image_paths = [item['path'] for item in job_data]
            path_to_data = {item['path']: item for item in job_data}
            
            logger.info(f"Processing job with {len(image_paths)} images")
            
            # Compute embeddings using optimized pipeline
            results = []
            for embedding_result in self.embedding_computer.compute_embeddings_streaming(
                image_paths, 
                batch_size=None  # Use auto-sizing
            ):
                path = embedding_result['path']
                original_data = path_to_data.get(path, {})
                
                # Create result with all required fields
                result = {
                    "path": path,
                    "combined_emb": embedding_result['combined_embedding'],
                    "hashes": original_data.get('hashes', {})
                }
                
                # Add CLIP embedding if available
                if 'clip_embedding' in embedding_result:
                    result["clip_emb"] = embedding_result['clip_embedding']
                
                results.append(result)
            
            processing_time = time.time() - start_time
            self.stats.update_job_stats(len(image_paths), processing_time)
            
            logger.info(f"Completed job: {len(results)} embeddings in {processing_time:.2f}s")
            return results
            
        except Exception as e:
            logger.error(f"Error processing job: {e}", exc_info=True)
            return []
    
    def update_system_stats(self):
        """Update system resource usage statistics."""
        # Memory usage
        process = psutil.Process()
        memory_info = process.memory_info()
        self.stats.memory_usage_mb = memory_info.rss / 1024 / 1024
        
        # GPU memory usage
        if torch.cuda.is_available():
            self.stats.gpu_memory_mb = torch.cuda.memory_allocated() / 1024 / 1024
    
    def report_stats(self):
        """Report worker statistics to Redis."""
        try:
            self.update_system_stats()
            stats_key = f"forceps:worker:{self.worker_id}:stats"
            self.redis_client.set_stats(stats_key, asdict(self.stats), expire=600)
        except Exception as e:
            logger.warning(f"Failed to report stats: {e}")
    
    def run(self):
        """Main worker loop."""
        logger.info(f"Worker {self.worker_id} starting main loop")
        self.running = True
        
        consecutive_empty_polls = 0
        max_empty_polls = 10
        poll_timeout = 5  # seconds
        
        while self.running:
            try:
                # Poll for jobs with timeout
                jobs = self.redis_client.dequeue_batch(
                    self.job_queue, 
                    max_items=1,  # Process one job at a time for better load balancing
                    timeout=poll_timeout
                )
                
                if jobs:
                    consecutive_empty_polls = 0
                    
                    for job_data in jobs:
                        if not self.running:
                            break
                        
                        # Process the job
                        results = self.process_job(job_data)
                        
                        if results:
                            # Push results to Redis
                            self.redis_client.enqueue_batch(self.results_queue, [results])
                            
                            # Update global progress counter
                            try:
                                self.redis_client.redis_client.incrby(
                                    "forceps:stats:embeddings_done", 
                                    len(results)
                                )
                            except Exception as e:
                                logger.warning(f"Failed to update progress counter: {e}")
                    
                    # Report statistics
                    self.report_stats()
                
                else:
                    # No jobs available
                    consecutive_empty_polls += 1
                    
                    if consecutive_empty_polls >= max_empty_polls:
                        # Reduce polling frequency when queue is empty
                        time.sleep(min(30, poll_timeout * consecutive_empty_polls))
                    else:
                        time.sleep(1)
                    
                    # Still report stats periodically even when idle
                    if consecutive_empty_polls % 5 == 0:
                        self.report_stats()
            
            except Exception as e:
                logger.error(f"Error in worker main loop: {e}", exc_info=True)
                time.sleep(5)  # Brief pause before retrying
        
        logger.info(f"Worker {self.worker_id} stopped")
    
    def stop(self):
        """Stop the worker gracefully."""
        logger.info(f"Stopping worker {self.worker_id}")
        self.running = False
        
        # Final stats report
        try:
            self.report_stats()
        except Exception:
            pass


def main():
    """Main entry point for the optimized worker."""
    parser = argparse.ArgumentParser(description="FORCEPS Optimized Worker")
    parser.add_argument("--config", type=str, default="app/config.yaml", 
                       help="Path to configuration file")
    parser.add_argument("--worker_id", type=str, required=True,
                       help="Unique worker identifier")
    parser.add_argument("--redis_host", type=str, default="127.0.0.1",
                       help="Redis host override")
    parser.add_argument("--redis_port", type=int, default=6379,
                       help="Redis port override")
    
    args = parser.parse_args()
    
    # Load configuration
    try:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        logger.error(f"Configuration file not found: {args.config}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        sys.exit(1)
    
    # Override Redis settings if provided
    if args.redis_host != "127.0.0.1":
        config['redis']['host'] = args.redis_host
    if args.redis_port != 6379:
        config['redis']['port'] = args.redis_port
    
    # Optimize system settings
    optimize_gpu_settings()
    
    # Set process title for easier identification
    try:
        import setproctitle
        setproctitle.setproctitle(f"forceps-worker-{args.worker_id}")
    except ImportError:
        pass
    
    # Create and run worker
    worker = OptimizedWorker(args.worker_id, config)
    
    try:
        worker.run()
    except KeyboardInterrupt:
        logger.info("Worker interrupted by user")
    except Exception as e:
        logger.error(f"Worker crashed: {e}", exc_info=True)
        sys.exit(1)
    finally:
        worker.stop()


if __name__ == "__main__":
    main()
