#!/usr/bin/env python3
"""
FORCEPS Model Conversion Script

Converts the PyTorch ViT and CLIP models to the ONNX format,
paving the way for TensorRT compilation.
"""
import argparse
import logging
import torch
from pathlib import Path
import os

# It's better to set the device before importing torch stuff
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from app.embeddings import load_models

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="FORCEPS Model Converter to ONNX")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the ONNX models.")
    parser.add_argument("--opset", type=int, default=14, help="ONNX opset version.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    logger.info("Loading original PyTorch models on CPU for conversion...")

    # Ensure models are loaded on CPU for device-agnostic export
    vit_model, clip_model, _, _, _, _ = load_models()
    vit_model.to("cpu")
    if clip_model:
        clip_model.to("cpu")

    # --- Convert ViT Model ---
    logger.info("Converting ViT model to ONNX...")
    try:
        # Dummy input for tracing the model graph
        dummy_input_vit = torch.randn(1, 3, 224, 224, requires_grad=False)
        vit_model.eval()

        # The HuggingFace model returns a complex output tuple.
        # We need to wrap it to extract only the pooler_output which we use for embeddings.
        class ViTWrapper(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model
            def forward(self, x):
                return self.model(x).pooler_output

        vit_wrapper = ViTWrapper(vit_model)

        torch.onnx.export(
            vit_wrapper,
            dummy_input_vit,
            str(output_dir / "vit.onnx"),
            input_names=['input'],
            output_names=['pooler_output'],
            dynamic_axes={'input': {0: 'batch_size'}, 'pooler_output': {0: 'batch_size'}},
            opset_version=args.opset,
            export_params=True
        )
        logger.info(f"ViT model successfully converted to {output_dir / 'vit.onnx'}")
    except Exception as e:
        logger.error(f"Failed to convert ViT model: {e}", exc_info=True)

    # --- Convert CLIP Image Encoder ---
    if clip_model:
        logger.info("Converting CLIP image encoder to ONNX...")
        try:
            # CLIP's visual model takes a specific input size, (224, 224) for ViT-B/32
            dummy_input_clip = torch.randn(1, 3, 224, 224, requires_grad=False)
            visual_model = clip_model.visual
            visual_model.eval()

            torch.onnx.export(
                visual_model,
                dummy_input_clip,
                str(output_dir / "clip_visual.onnx"),
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
                opset_version=args.opset,
                export_params=True
            )
            logger.info(f"CLIP image encoder successfully converted to {output_dir / 'clip_visual.onnx'}")
        except Exception as e:
            logger.error(f"Failed to convert CLIP image encoder: {e}", exc_info=True)

    logger.info("--- Model conversion complete ---")


if __name__ == "__main__":
    main()
