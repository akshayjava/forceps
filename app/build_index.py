#!/usr/bin/env python3
"""
FORCEPS Index Builder Script

Pulls computed embeddings from a Redis queue, builds a FAISS index,
and saves it to disk.
"""
import argparse
import logging
import redis
import json
import numpy as np
import faiss
import pickle
from pathlib import Path
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def build_index_for_embeddings(embeddings, d, n, args):
    """Builds a FAISS index for a given set of embeddings."""
    logger.info(f"Building FAISS index for {n} vectors of dimension {d}.")
    use_gpu = hasattr(faiss, "GpuResources")
    gpu_res = faiss.GpuResources() if use_gpu else None

    pca_ret = None
    d_final = d
    if args.use_pca:
        logger.info(f"Applying PCA to reduce dimension to {args.pca_dim}.")
        eff_pca_dim = max(1, min(args.pca_dim, d, n))
        pca_mat = faiss.PCAMatrix(d, eff_pca_dim)
        pca_mat.train(embeddings[:args.train_samples])
        pca_ret = pca_mat
        d_final = eff_pca_dim

    def choose_m(dim, m_pref):
        candidates = [m for m in [m_pref, 64, 48, 32, 24, 16, 12, 8, 6, 4, 3, 2, 1] if m <= dim and dim % m == 0]
        return candidates[0] if candidates else 1

    nlist = min(args.ivf_nlist, n // 100) # Ensure nlist is reasonable for dataset size
    nlist = max(nlist, 1)
    pq_m = choose_m(d_final, args.pq_m)
    quantizer = faiss.IndexFlatL2(d_final)
    cpu_index = faiss.IndexIVFPQ(quantizer, d_final, nlist, pq_m, 8)

    index_to_train = faiss.index_cpu_to_gpu(gpu_res, 0, cpu_index) if use_gpu else cpu_index
    logger.info(f"Training FAISS index on {'GPU' if use_gpu else 'CPU'} with {min(n, args.train_samples)} samples...")

    train_data = pca_ret.apply_py(embeddings[:args.train_samples]) if args.use_pca else embeddings[:args.train_samples]
    index_to_train.train(train_data)

    logger.info("Adding all embeddings to the index...")
    add_batch_size = args.add_batch
    for off in range(0, n, add_batch_size):
        end = off + add_batch_size
        batch_data = embeddings[off:end]
        if args.use_pca: batch_data = pca_ret.apply_py(batch_data)
        index_to_train.add(batch_data)
        logger.info(f"Added {end}/{n} vectors to index.")

    logger.info("Finalizing index...")
    final_index = faiss.index_gpu_to_cpu(index_to_train) if use_gpu else index_to_train
    final_index.nprobe = min(64, max(1, nlist // 16))
    return final_index, pca_ret

def main():
    parser = argparse.ArgumentParser(description="FORCEPS Index Builder")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the FAISS index and metadata.")

    # Redis args
    parser.add_argument("--redis_host", type=str, default="localhost", help="Redis server host.")
    parser.add_argument("--redis_port", type=int, default=6379, help="Redis server port.")
    parser.add_argument("--results_queue", type=str, default="forceps:results_queue", help="Redis queue for results.")

    # FAISS args (should match the engine's defaults)
    parser.add_argument("--use_pca", action='store_true', help="Enable PCA for dimensionality reduction.")
    parser.add_argument("--pca_dim", type=int, default=384, help="Dimension after PCA.")
    parser.add_argument("--ivf_nlist", type=int, default=4096, help="Number of IVF clusters.")
    parser.add_argument("--pq_m", type=int, default=64, help="Number of sub-quantizers for PQ.")
    parser.add_argument("--train_samples", type=int, default=5000, help="Number of samples to train FAISS index.")
    parser.add_argument("--add_batch", type=int, default=8192, help="Batch size for adding vectors to FAISS index.")

    args = parser.parse_args()

    logger.info("--- FORCEPS Index Builder Starting ---")

    try:
        r = redis.Redis(host=args.redis_host, port=args.redis_port, db=0)
        r.ping()
        logger.info(f"Successfully connected to Redis at {args.redis_host}:{args.redis_port}")
    except redis.exceptions.ConnectionError as e:
        logger.error(f"Could not connect to Redis: {e}")
        return

    # 1. Consume results from the queue
    logger.info(f"Consuming results from queue '{args.results_queue}'...")
    all_results = []
    while True:
        # Pop up to 100 items at a time to be efficient
        pipe = r.pipeline()
        pipe.lrange(args.results_queue, 0, 99)
        pipe.ltrim(args.results_queue, 100, -1)
        items = pipe.execute()[0]

        if not items:
            logger.info("No more results in queue. Waiting for more...")
            time.sleep(10) # Wait for more results
            if not r.lrange(args.results_queue, 0, 0): # Check if still empty
                logger.info("Queue appears empty. Finalizing index.")
                break
            else:
                continue

        for item in items:
            all_results.extend(json.loads(item))
        logger.info(f"Consumed {len(items)} jobs. Total embeddings so far: {len(all_results)}")

    if not all_results:
        logger.warning("No embeddings were found in the results queue. Exiting.")
        return

    # 2. Prepare data for FAISS
    logger.info("Preparing data for indexing...")
    paths = [res["path"] for res in all_results]
    combined_embs = np.array([res["combined_emb"] for res in all_results], dtype=np.float32)

    has_clip = "clip_emb" in all_results[0]
    clip_embs = None
    if has_clip:
        clip_embs = np.array([res["clip_emb"] for res in all_results], dtype=np.float32)

    n, d_comb = combined_embs.shape

    # 3. Build indexes
    index_comb, pca_matrix = build_index_for_embeddings(combined_embs, d_comb, n, args)

    index_clip = None
    if has_clip and clip_embs is not None:
        _, d_clip = clip_embs.shape
        index_clip, _ = build_index_for_embeddings(clip_embs, d_clip, n, args)

    # 4. Save results
    logger.info("Saving final FAISS indexes, PCA matrix, and path mapping...")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    faiss.write_index(index_comb, str(output_dir / "combined.index"))
    if index_clip:
        faiss.write_index(index_clip, str(output_dir / "clip.index"))
    if pca_matrix:
        with open(output_dir / "pca.matrix.pkl", "wb") as f:
            pickle.dump(pca_matrix, f)
    with open(output_dir / "image_paths.json", "w") as f:
        json.dump(paths, f)

    logger.info("--- Index building complete ---")

if __name__ == "__main__":
    main()
