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
import json
import subprocess
from typing import Dict, Any

# Lightweight EXIF helper
from app.utils import read_exif

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
    parser.add_argument("--captions", action="store_true", help="Generate captions with Ollama llava and save to captions.tsv")
    parser.add_argument("--ollama_model", default="llava", help="Ollama model name for captions")
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
        bs = min(args.batch_size, 8)
    else:
        use_fp16 = False
        compile_model = False
        bs = min(args.batch_size, 8)

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

    # Save EXIF metadata (make/model/datetime) for filters
    try:
        paths = None
        import pickle
        with open(os.path.join(output_dir, "image_paths.pkl"), "rb") as f:
            paths = pickle.load(f)
        exif_map: Dict[str, Any] = {}
        for p in paths:
            try:
                ex = read_exif(Path(p)) or {}
                # serialize time as string
                dt = ex.get("DateTime")
                if dt is not None:
                    import time as _t
                    try:
                        ex["DateTime"] = _t.strftime("%Y-%m-%d %H:%M:%S", dt)
                    except Exception:
                        ex["DateTime"] = str(dt)
                exif_map[p] = ex
            except Exception:
                exif_map[p] = {}
        with open(os.path.join(output_dir, "exif.json"), "w") as f:
            json.dump(exif_map, f)
        print("Saved exif.json")
    except Exception as exc:
        print("EXIF save skipped:", exc)

    # Optional captions via Ollama
    if args.captions:
        try:
            # Check model availability
            out = subprocess.check_output(["ollama", "list"]) if shutil.which("ollama") else b""
            if args.ollama_model.split(":")[0] in out.decode("utf-8", "ignore"):
                print("Generating captions to captions.tsv using", args.ollama_model)
                cap_path = os.path.join(output_dir, "captions.tsv")
                with open(cap_path, "w") as outf:
                    for i, p in enumerate(paths or []):
                        prompt = f"Describe the image in detail (scene, objects, colors, notable items). Image: file://{p}"
                        proc = subprocess.run(["ollama", "run", args.ollama_model], input=prompt.encode("utf-8"), capture_output=True)
                        cap = proc.stdout.decode("utf-8", "ignore").strip().replace("\t", " ")
                        outf.write(f"{p}\t{cap}\n")
                        if (i+1) % 50 == 0:
                            print(f"captions: {i+1}/{len(paths)}")
                print("Saved captions.tsv")
            else:
                print("Ollama model not found; skipping captions")
        except Exception as exc:
            print("Captions skipped:", exc)

if __name__ == "__main__":
    main()
