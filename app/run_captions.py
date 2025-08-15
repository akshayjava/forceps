#!/usr/bin/env python3
"""
Run Phase 2 (captions) over an image set using Ollama if available.

Reads app/config.yaml for input_dir and worker max_workers.
Updates Redis counters (captions_done) via engine._caption_one_image side effects.
"""
import argparse
import yaml
import os
from pathlib import Path

from app.engine import phase2_caption


def discover_images(root_dir: str):
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
    images = []
    for r, _, files in os.walk(root_dir):
        for fn in files:
            if Path(fn).suffix.lower() in exts:
                images.append(os.path.join(r, fn))
    return images


def main():
    ap = argparse.ArgumentParser(description="Run Phase 2 captions")
    ap.add_argument("--config", type=str, default="app/config.yaml", help="Path to config.yaml")
    ap.add_argument("--input_dir", type=str, default=None, help="Override input_dir")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, "r"))
    input_dir = args.input_dir or cfg["data"]["input_dir"]
    max_workers = int(cfg["performance"]["worker"]["max_workers"]) or 4

    class _Args:
        def __init__(self, mw: int):
            self.max_workers = mw

    images = discover_images(input_dir)
    phase2_caption(images, _Args(max_workers))


if __name__ == "__main__":
    main()


