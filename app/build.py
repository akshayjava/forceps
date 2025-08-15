#!/usr/bin/env python3
"""
Export ViT and CLIP visual encoders to ONNX for the worker engine.
"""
import os
from pathlib import Path
import torch
from transformers import AutoModel

def export_vit(model_name: str, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    dummy = torch.randn(1, 3, 224, 224)
    torch.onnx.export(
        model,
        dummy,
        str(out_dir / "vit.onnx"),
        input_names=["pixel_values"],
        output_names=["last_hidden_state"],
        opset_version=17,
        dynamic_axes={"pixel_values": {0: "batch"}, "last_hidden_state": {0: "batch"}},
    )

def main():
    onnx_dir = Path("/Users/Shared/Projects/foreceps/models/onnx")
    export_vit("google/vit-base-patch16-224-in21k", onnx_dir)
    # CLIP visual export can be added later if needed

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
import shutil, subprocess, sys, os
APP_NAME = "FORCEPS"
ENTRY = "app/main.py"

def clean():
    shutil.rmtree("build", ignore_errors=True)
    shutil.rmtree("dist", ignore_errors=True)
    for f in os.listdir("."):
        if f.endswith(".spec"):
            os.remove(f)

def build():
    print("[build] Running PyInstaller...")
    subprocess.run([sys.executable, "-m", "PyInstaller", "--onefile", "--name", APP_NAME, ENTRY], check=True)
    print("[build] done.")

def make_zip():
    zipname = f"{APP_NAME}_package.zip"
    if os.path.exists(zipname):
        os.remove(zipname)
    shutil.make_archive(APP_NAME + "_package", 'zip', ".")
    print("[zip] created", zipname)

if __name__ == "__main__":
    clean()
    build()
    make_zip()
