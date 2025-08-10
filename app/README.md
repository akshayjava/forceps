# FORCEPS — Forensic Optimized Retrieval & Clustering of Evidence via Perceptual Search

## Overview
FORCEPS is a local forensic image similarity and natural-language search tool designed for fast, accurate, and private on-site investigations.

Key features:
- Two-phase indexing (Phase 1: embeddings; Phase 2: NL captions)
- Multi-model embeddings (ViT + CLIP)
- Perceptual hashing (pHash/aHash/dHash) + placeholder for PhotoDNA
- FAISS vector DB (HNSW) for fast similarity
- Optional Ollama integration for captions (local only)
- GPU acceleration when available
- Incremental cache to skip unchanged files
- All processing is local; no data leaves the device

## Quickstart (source)
1. Extract repo.
2. Install + create virtualenv:
```bash
./install.sh
source ./venv_forceps/bin/activate
