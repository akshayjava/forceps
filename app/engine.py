#!/usr/bin/env python3
"""
FORCEPS Standalone Indexing Engine
"""
import os
import argparse
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

import time
import numpy as np
import faiss
from concurrent.futures import ThreadPoolExecutor, as_completed

from app.embeddings import (
    load_models,
    compute_batch_embeddings,
    preprocess_image_for_vit_clip,
)
from app.utils import (
    is_image_file,
    fingerprint,
    load_cache,
    save_cache,
    compute_perceptual_hashes,
    read_exif,
)
from app.llm_ollama import ollama_installed, generate_caption_ollama, model_available
from torch.utils.data import Dataset
import torch
import cv2
try:
    from turbojpeg import TurboJPEG
    jpeg = TurboJPEG()
    USE_TURBOJPEG = True
except ImportError:
    USE_TURBOJPEG = False


class ForcepsDataset(Dataset):
    def __init__(self, image_paths, preprocess_vit, preprocess_clip):
        self.image_paths = image_paths
        self.preprocess_vit = preprocess_vit
        self.preprocess_clip = preprocess_clip

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path_str = self.image_paths[idx]
        try:
            im = None
            # Fast path for JPEGs
            if USE_TURBOJPEG and path_str.lower().endswith(('.jpg', '.jpeg')):
                with open(path_str, 'rb') as f:
                    im = jpeg.decode(f.read(), pixel_format=turbojpeg.TJPF_RGB)
                im = Image.fromarray(im)
            # Fast path for other formats with OpenCV
            elif path_str.lower().endswith(('.png', '.bmp', '.tiff')):
                img_array = cv2.imread(path_str)
                if img_array is not None:
                    im = Image.fromarray(cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB))

            # Fallback to PIL
            if im is None:
                im = Image.open(path_str).convert("RGB")

            vit_t = self.preprocess_vit(im)
            clip_t = self.preprocess_clip(im) if self.preprocess_clip is not None else torch.tensor([])
            return vit_t, clip_t, path_str
        except Exception as e:
            logger.warning(f"Skipping corrupted image {path_str}: {e}")
            return None, None, path_str # Return None for collate_fn to handle


# Global constants
CACHE_DIR = Path("./cache")
CACHE_DIR.mkdir(exist_ok=True)
DEVICE = "cuda" if __import__("torch").cuda.is_available() else "cpu"


def collate_fn(batch):
    batch = list(filter(lambda x: x is not None and x[0] is not None, batch))
    if not batch:
        return None, None, None
    vit_tensors, clip_tensors, paths = zip(*batch)

    vit_tensors = torch.stack(vit_tensors)
    clip_tensors = torch.stack(clip_tensors) if clip_tensors[0].numel() > 0 else None

    return vit_tensors, clip_tensors, paths

def compute_embeddings_for_job(image_paths, models, args):
    logger.info(f"Processing job of {len(image_paths)} images with ONNX runtime.")

    vit_session = models["vit_session"]
    clip_session = models["clip_session"]
    preprocess_vit = models["preprocess_vit"]
    preprocess_clip = models["preprocess_clip"]
    clip_dim = models["clip_dim"]

    dataset = ForcepsDataset(image_paths, preprocess_vit, preprocess_clip)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.max_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        prefetch_factor=2 if args.max_workers > 0 else None
    )

    results = []
    for vit_tensor_batch, clip_tensor_batch, paths in dataloader:
        if vit_tensor_batch is None: continue

        # Run ViT ONNX model
        vit_input_name = vit_session.get_inputs()[0].name
        vit_output_name = vit_session.get_outputs()[0].name
        emb_vit = vit_session.run([vit_output_name], {vit_input_name: vit_tensor_batch.numpy()})[0]

        # Run CLIP ONNX model if it exists
        emb_clip = None
        if clip_session and clip_tensor_batch is not None:
            clip_input_name = clip_session.get_inputs()[0].name
            clip_output_name = clip_session.get_outputs()[0].name
            emb_clip = clip_session.run([clip_output_name], {clip_input_name: clip_tensor_batch.numpy()})[0]

        # Normalize embeddings
        emb_vit = emb_vit / (np.linalg.norm(emb_vit, axis=1, keepdims=True) + 1e-10)
        if emb_clip is not None:
            emb_clip = emb_clip / (np.linalg.norm(emb_clip, axis=1, keepdims=True) + 1e-10)
            combined_batch = np.concatenate([emb_vit, emb_clip], axis=1)
        else:
            combined_batch = emb_vit

        # Append results
        for i, path in enumerate(paths):
            result = {"path": path, "combined_emb": combined_batch[i].tolist()}
            if clip_dim > 0 and emb_clip is not None:
                result["clip_emb"] = emb_clip[i].tolist()
            results.append(result)

    logger.info(f"Finished ONNX job, produced {len(results)} embeddings.")
    return results

def _caption_one_image(p):
    try:
        fp = fingerprint(Path(p))
        cached = load_cache(fp) or {}
        if cached.get("metadata", {}).get("caption"): return
        cap = generate_caption_ollama(p)
        if cap:
            cached["metadata"] = cached.get("metadata", {})
            cached["metadata"]["caption"] = cap
            save_cache(fp, cached)
    except Exception as e:
        logger.warning(f"Could not caption image {p}: {e}")

def phase2_caption(image_paths, args):
    logger.info("Starting Phase 2: Caption generation.")
    if not ollama_installed() or not model_available("llava"):
        logger.warning("Ollama or llava model not available. Skipping captioning.")
        return

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        list(executor.map(_caption_one_image, image_paths))
    logger.info("Phase 2 complete.")

import pickle

import pickle
import redis
import json
import onnxruntime as ort

def load_onnx_models(model_dir: str):
    logger.info(f"Loading ONNX models from {model_dir}...")
    # It's good practice to specify providers, especially for TensorRT execution
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']

    vit_path = os.path.join(model_dir, "vit.onnx")
    vit_session = ort.InferenceSession(vit_path, providers=providers)

    clip_session = None
    clip_path = os.path.join(model_dir, "clip_visual.onnx")
    if os.path.exists(clip_path):
        clip_session = ort.InferenceSession(clip_path, providers=providers)

    # We still need the preprocessors from the original models
    _, _, preprocess_vit, preprocess_clip, vit_dim, clip_dim = load_models()

    models = {
        "vit_session": vit_session,
        "clip_session": clip_session,
        "preprocess_vit": preprocess_vit,
        "preprocess_clip": preprocess_clip,
        "vit_dim": vit_dim, # This might need to be hardcoded or saved with the model
        "clip_dim": clip_dim
    }
    return models

def main():
    parser = argparse.ArgumentParser(description="FORCEPS Worker Engine")

    # Model & Performance args
    parser.add_argument("--model_dir", type=str, required=True, help="Directory with ONNX models.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for model inference.")
    parser.add_argument("--max_workers", type=int, default=8, help="Max workers for DataLoader.")

    # Redis args
    parser.add_argument("--redis_host", type=str, default="localhost", help="Redis server host.")
    parser.add_argument("--redis_port", type=int, default=6379, help="Redis server port.")
    parser.add_argument("--job_queue", type=str, default="forceps:job_queue", help="Redis queue for jobs.")
    parser.add_argument("--results_queue", type=str, default="forceps:results_queue", help="Redis queue for results.")

    args = parser.parse_args()

    logger.info("--- FORCEPS Worker Engine Starting ---")

    # 1. Connect to Redis
    try:
        r = redis.Redis(host=args.redis_host, port=args.redis_port, db=0)
        r.ping()
        logger.info(f"Successfully connected to Redis at {args.redis_host}:{args.redis_port}")
    except redis.exceptions.ConnectionError as e:
        logger.error(f"Could not connect to Redis: {e}")
        return

    # 2. Load ONNX models once per worker
    logger.info("Loading ONNX models...")
    models = load_onnx_models(args.model_dir)

    # 3. Main worker loop
    logger.info(f"Worker listening for jobs on '{args.job_queue}'...")
    while True:
        try:
            # Blocking pop from the job queue
            _, job_data = r.blpop(args.job_queue)
            image_paths = json.loads(job_data)

            logger.info(f"Received job with {len(image_paths)} images.")

            # Process the job to get embeddings
            results = compute_embeddings_for_job(image_paths, models, args)

            # Push results to the results queue
            if results:
                r.rpush(args.results_queue, json.dumps(results))
                logger.info(f"Pushed {len(results)} embeddings to '{args.results_queue}'.")

            logger.info(f"Finished processing job. Waiting for next job...")

        except KeyboardInterrupt:
            logger.info("Shutdown signal received. Exiting worker.")
            break
        except Exception as e:
            logger.error(f"An error occurred in the worker loop: {e}", exc_info=True)
            time.sleep(5)

if __name__ == "__main__":
    main()
