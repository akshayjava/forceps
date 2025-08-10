"""
Simple Ollama wrapper for FORCEPS. Assumes 'ollama' CLI is installed locally.
Adjust generate_caption_ollama() to match your local Ollama model invocation.
"""
import os
import shutil
import subprocess
import json

OLLAMA_CLI = shutil.which("ollama")

def ollama_installed():
    return OLLAMA_CLI is not None

def model_available(model_name: str) -> bool:
    if not ollama_installed():
        return False
    try:
        out = subprocess.check_output([OLLAMA_CLI, "list", "--format", "json"], timeout=5)
        j = json.loads(out.decode("utf-8", errors="ignore"))
        names = {item.get("name") for item in (j or [])}
        # Accept exact or prefix match (e.g., "llava" matches "llava:latest")
        return any(n == model_name or (n and n.split(":")[0] == model_name) for n in names)
    except Exception:
        return False

def generate_caption_ollama(image_path, model_name="llava"):
    """
    Best-effort wrapper. Many Ollama multimodal models accept binary image input;
    modify this function to feed the image in the way your local model expects.
    """
    if not ollama_installed():
        return None
    if not model_available(model_name):
        return None
    prompt = f"Describe the image in detail (scene, objects, colors, notable items). Path: {image_path}"
    try:
        timeout_s = float(os.environ.get("CAPTION_TIMEOUT", 15))
        cmd = [OLLAMA_CLI, "run", model_name, "--format", "json"]
        proc = subprocess.run(cmd, input=prompt.encode("utf-8"), capture_output=True, timeout=timeout_s)
        out = proc.stdout.decode(errors="ignore").strip()
        try:
            j = json.loads(out)
            if isinstance(j, dict) and "response" in j:
                return j["response"]
            return out
        except Exception:
            return out
    except Exception:
        return None
