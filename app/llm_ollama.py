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
        # Some Ollama versions do not support --format json
        out = subprocess.check_output([OLLAMA_CLI, "list"], timeout=5)
        text = out.decode("utf-8", errors="ignore")
        lines = [ln.strip() for ln in text.splitlines() if ln.strip() and not ln.lower().startswith("name")]
        for ln in lines:
            first = ln.split()[0]
            base = first.split(":")[0]
            if base == model_name:
                return True
        return False
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
    
    # First, check if image exists
    if not os.path.exists(image_path):
        return None
    
    try:
        # Use stdin prompt; many Ollama builds accept file:// URLs within prompt
        prompt = f"Describe the image in detail (scene, objects, colors, notable items). Image: file://{image_path}"
        timeout_s = float(os.environ.get("CAPTION_TIMEOUT", 30))
        cmd = [OLLAMA_CLI, "run", model_name]
        proc = subprocess.run(cmd, input=prompt.encode("utf-8"), capture_output=True, timeout=timeout_s)
        out = proc.stdout.decode(errors="ignore").strip()
        return out or None
    except Exception:
        return None

def general_ollama_query(query_text: str, model_name: str = "llama2") -> str:
    """
    Performs a general text query to an Ollama LLM.
    Assumes 'ollama' CLI is installed locally.
    """
    if not ollama_installed():
        return None
    if not model_available(model_name):
        return None
    
    try:
        timeout_s = float(os.environ.get("OLLAMA_QUERY_TIMEOUT", 60))
        cmd = [OLLAMA_CLI, "run", model_name]
        proc = subprocess.run(cmd, input=query_text.encode("utf-8"), capture_output=True, timeout=timeout_s)
        out = proc.stdout.decode(errors="ignore").strip()
        return out or None
    except Exception:
        return None
