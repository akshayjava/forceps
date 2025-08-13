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
import yaml
import numpy as np
import faiss
import pickle
from pathlib import Path
import time
from app.utils import load_cache, fingerprint
from whoosh.index import create_in
from whoosh.fields import Schema, TEXT, ID
from whoosh.analysis import StemmingAnalyzer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define the schema for the text index.
# We'll store the path as a unique ID to link back to the FAISS index.
# All searchable text will be combined into a single 'content' field for simplicity.
schema = Schema(
    path=ID(stored=True, unique=True),
    content=TEXT(analyzer=StemmingAnalyzer(), stored=False)
)


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
    parser.add_argument("--config", type=str, default="app/config.yaml", help="Path to the configuration file.")
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
    cfg_faiss = config['performance']['faiss']
    cfg_case = config.get('case_details', {'case_name': f'case_{int(time.time())}'})

    # Create a simple namespace object for the build_index_for_embeddings function
    faiss_args = argparse.Namespace(**cfg_faiss)

    logger.info("--- FORCEPS Index Builder Starting ---")
    logger.info(f"Case Name: {cfg_case.get('case_name')}")

    try:
        r = redis.Redis(host=cfg_redis['host'], port=cfg_redis['port'], db=0)
        r.ping()
        logger.info(f"Successfully connected to Redis at {cfg_redis['host']}:{cfg_redis['port']}")
    except redis.exceptions.ConnectionError as e:
        logger.error(f"Could not connect to Redis: {e}")
        return

    # 1. Consume results from the queue
    logger.info(f"Consuming results from queue '{cfg_redis['results_queue']}'...")
    all_results = []
    while True:
        pipe = r.pipeline()
        pipe.lrange(cfg_redis['results_queue'], 0, 99)
        pipe.ltrim(cfg_redis['results_queue'], 100, -1)
        items = pipe.execute()[0]

        if not items:
            logger.info("No more results in queue. Waiting for more...")
            time.sleep(10)
            if not r.lrange(cfg_redis['results_queue'], 0, 0):
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
    manifest_data = [{"path": res["path"], "hashes": res.get("hashes")} for res in all_results]
    combined_embs = np.array([res["combined_emb"] for res in all_results], dtype=np.float32)

    has_clip = "clip_emb" in all_results[0]
    clip_embs = None
    if has_clip:
        clip_embs = np.array([res["clip_emb"] for res in all_results if "clip_emb" in res], dtype=np.float32)

    n, d_comb = combined_embs.shape

    # 3. Build Vector Index (FAISS)
    logger.info("Building vector (FAISS) index...")
    index_comb, pca_matrix = build_index_for_embeddings(combined_embs, d_comb, n, faiss_args)
    index_clip = None
    if has_clip and clip_embs is not None:
        _, d_clip = clip_embs.shape
        index_clip, _ = build_index_for_embeddings(clip_embs, d_clip, n, faiss_args)
    logger.info("Vector index building complete.")

    # 4. Build Text Index (Whoosh)
    logger.info("Building text (Whoosh) index...")
    case_output_dir = Path(cfg_data['output_dir']) / cfg_case['case_name']
    whoosh_dir = case_output_dir / "whoosh_index"
    whoosh_dir.mkdir(parents=True, exist_ok=True)

    whoosh_ix = create_in(whoosh_dir, schema)
    writer = whoosh_ix.writer()

    # We need to get the captions from the cache for each file
    for item in manifest_data:
        path = item['path']
        # This is inefficient, but the cache is the only place captions are stored right now
        # A better architecture would pass captions through the pipeline as well.
        cached_item = load_cache(fingerprint(Path(path))) or {}
        caption = cached_item.get("metadata", {}).get("caption", "")

        # Combine all text fields into one 'content' field for searching
        content = " ".join([
            Path(path).name,
            caption
        ])
        writer.add_document(path=path, content=content)
    writer.commit()
    logger.info("Text index building complete.")

    # 5. Save results
    logger.info("Saving final FAISS indexes, PCA matrix, and manifest...")
    case_output_dir.mkdir(parents=True, exist_ok=True)

    faiss.write_index(index_comb, str(case_output_dir / "combined.index"))
    if index_clip:
        faiss.write_index(index_clip, str(case_output_dir / "clip.index"))
    if pca_matrix:
        with open(case_output_dir / "pca.matrix.pkl", "wb") as f:
            pickle.dump(pca_matrix, f)
    with open(case_output_dir / "manifest.json", "w") as f:
        json.dump(manifest_data, f, indent=2)

    logger.info(f"--- Index building complete for case '{cfg_case['case_name']}' ---")

if __name__ == "__main__":
    main()
