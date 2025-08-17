#!/usr/bin/env python3
# FORCEPS CLI — ViT indexing without the UI
#
# Usage examples (run from repo root):
#   PYTHONPATH=. python3 run_cli.py \
#     --image_dir "/path/to/images" \
#     --output_dir index_out_opt \
#     --device auto --batch_size 16
#
# Notes:
# - On CPU/MPS, we disable torch.compile and FP16 by default for stability
# - On CUDA, FP16 is enabled by default and compile is off by default (can be enabled)
# - If process_images_optimized is not available, we fall back to process_images
import os
import sys
import argparse
from pathlib import Path
import torch

def main():
    try:
        # Ensure repo root on sys.path
        repo_root = Path(__file__).resolve().parent
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))
        from app.vit_indexer import ViTIndexer
    except Exception as exc:
        print("Import error:", exc)
        print("Run from repo root or use: PYTHONPATH=. python3 run_cli.py")
        sys.exit(1)

    # CLI args
    parser = argparse.ArgumentParser(description="FORCEPS: ViT indexing CLI")
    parser.add_argument("--image_dir", required=False, default=os.environ.get("FORCEPS_IMAGE_DIR", "/Users/akshayjava/Downloads"), help="Directory containing images")
    parser.add_argument("--output_dir", required=False, default=os.environ.get("FORCEPS_OUTPUT_DIR", "index_out_opt"), help="Output directory for index and metadata")
    parser.add_argument("--device", choices=["auto","cpu","cuda","mps"], default="auto", help="Computation device")
    parser.add_argument("--batch_size", type=int, default=int(os.environ.get("FORCEPS_BATCH", 16)), help="Batch size")
    parser.add_argument("--max_images", type=int, default=None, help="Limit number of images (for testing)")
    parser.add_argument("--fp16", action="store_true", help="Enable FP16 (CUDA only)")
    parser.add_argument("--no-fp16", dest="fp16", action="store_false")
    parser.set_defaults(fp16=None)
    parser.add_argument("--compile", action="store_true", help="Enable torch.compile (CUDA only)")
    parser.add_argument("--no-compile", dest="compile", action="store_false")
    parser.set_defaults(compile=None)
    args = parser.parse_args()

    image_dir = args.image_dir
    output_dir = args.output_dir

    # Resolve device/flags
    if args.device != "auto":
        device = args.device
    else:
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    # Sensible defaults per device
    if device == "cuda":
        use_fp16 = True if args.fp16 is None else args.fp16
        compile_model = False if args.compile is None else args.compile
        bs = args.batch_size if args.batch_size else 32
    elif device == "mps":
        use_fp16 = False
        compile_model = False
        bs = args.batch_size
    else:
        use_fp16 = False
        compile_model = False
        bs = args.batch_size

    indexer = ViTIndexer(model_name="google/vit-base-patch16-224-in21k",
                         device=device, batch_size=bs, use_fp16=use_fp16,
                         compile_model=compile_model)

    # Prefer optimized path; fall back if unavailable
    try:
        indexer.process_images_optimized(image_dir=image_dir, output_dir=output_dir,
                                         max_images=args.max_images, streaming=True)
    except AttributeError:
        indexer.process_images(image_dir=image_dir, output_dir=output_dir,
                               max_images=args.max_images, chunk_size=10000)
    print("Done. FAISS index + metadata written to", output_dir)

if __name__ == "__main__":
    main()
