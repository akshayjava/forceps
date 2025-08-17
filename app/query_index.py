#!/usr/bin/env python3
"""
Query a built FORCEPS index from the command line.

Usage:
  python app/query_index.py --case_dir output_index/local-case --query_image /path/to/image.jpg

If --query_image is omitted, the first image in the manifest will be used.
"""
import argparse
import json
import os
import pickle
from pathlib import Path

import faiss
import numpy as np
from PIL import Image

from app.embeddings import load_models, compute_batch_embeddings


def load_case(case_dir: Path):
	"""Load an index and manifest from either FORCEPS case or vit_indexer output."""
	# FORCEPS case format
	combined_idx_path = case_dir / "combined.index"
	manifest_path = case_dir / "manifest.json"
	pca_path = case_dir / "pca.matrix.pkl"

	# vit_indexer format
	vi_faiss = case_dir / "image_index.faiss"
	vi_paths = case_dir / "image_paths.pkl"

	pca = None

	if combined_idx_path.exists() and manifest_path.exists():
		index = faiss.read_index(str(combined_idx_path))
		with open(manifest_path, "r") as f:
			manifest = json.load(f)
		if pca_path.exists():
			with open(pca_path, "rb") as f:
				pca = pickle.load(f)
		return index, manifest, pca

	if vi_faiss.exists() and vi_paths.exists():
		index = faiss.read_index(str(vi_faiss))
		with open(vi_paths, "rb") as f:
			paths = pickle.load(f)
		manifest = [{"path": p} for p in paths]
		return index, manifest, None

	raise FileNotFoundError(
		f"No supported index found in {case_dir}. Expected (combined.index + manifest.json) or (image_index.faiss + image_paths.pkl)."
	)


def embed_image_vit_only(image_path: Path, vit_model, preprocess_vit):
	img = Image.open(str(image_path)).convert("RGB")
	vit_t = preprocess_vit(img)
	comb, _ = compute_batch_embeddings([vit_t], [], vit_model, None)
	return comb[0].astype(np.float32)


def main():
	parser = argparse.ArgumentParser(description="Query FORCEPS index")
	parser.add_argument("--case_dir", type=str, required=True, help="Path to case directory (contains combined.index, manifest.json)")
	parser.add_argument("--query_image", type=str, default=None, help="Path to a query image (optional)")
	parser.add_argument("--k", type=int, default=10, help="Top-K results")
	args = parser.parse_args()

	case_dir = Path(args.case_dir)
	index, manifest, pca = load_case(case_dir)

	vit_model, clip_model, preprocess_vit, preprocess_clip, vit_dim, clip_dim = load_models()

	if args.query_image:
		query_path = Path(args.query_image)
	else:
		if not manifest:
			raise RuntimeError("Manifest is empty; nothing to query.")
		query_path = Path(manifest[0]["path"])  # use first entry

	print(f"Querying with: {query_path}")
	# Use ViT-only embedding to match default index dimension
	q_emb = embed_image_vit_only(query_path, vit_model, preprocess_vit)

	# Apply PCA if present
	q_vec = q_emb
	if pca is not None:
		q_vec = pca.apply_py(np.array([q_vec]))[0]

	# Adjust dimension if needed (defensive)
	if q_vec.shape[0] != index.d:
		# If our vector is longer, try slicing to index.d
		if q_vec.shape[0] > index.d:
			q_vec = q_vec[: index.d]
		else:
			raise AssertionError(f"Query dim {q_vec.shape[0]} < index dim {index.d}")

	q = np.array([q_vec], dtype=np.float32)

	D, I = index.search(q, args.k)

	print("Top results:")
	for rank, (dist, idx) in enumerate(zip(D[0], I[0]), start=1):
		if idx == -1:
			continue
		path = manifest[idx]["path"] if idx < len(manifest) else "<out-of-range>"
		print(f"{rank:2d}. idx={idx:5d} dist={dist:.4f} path={path}")


if __name__ == "__main__":
	main()


