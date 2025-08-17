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
    
    # First, check if image exists
    if not os.path.exists(image_path):
        return None
    
    try:
        # For llava model, we need to construct the proper JSON with the image encoded as base64
        import base64
        from PIL import Image
        import io
        
        # Read the image file
        with open(image_path, 'rb') as img_file:
            img_data = img_file.read()
        
        # Convert to base64
        img_base64 = base64.b64encode(img_data).decode('utf-8')
        
        # Create the JSON payload that Ollama expects for multimodal models
        prompt = "Describe this image in detail (scene, objects, colors, notable items)."
        payload = json.dumps({
            "prompt": prompt,
            "images": [img_base64]
        })
        
        timeout_s = float(os.environ.get("CAPTION_TIMEOUT", 30))  # Increased timeout for image processing
        cmd = [OLLAMA_CLI, "run", model_name, "--format", "json"]
        proc = subprocess.run(cmd, input=payload.encode("utf-8"), capture_output=True, timeout=timeout_s)
        out = proc.stdout.decode(errors="ignore").strip()
        
        try:
            j = json.loads(out)
            if isinstance(j, dict) and "response" in j:
                return j["response"]
            return out
        except Exception as e:
            print(f"Error parsing Ollama response: {e}")
            return out
    except Exception as e:
        print(f"Error generating caption: {e}")
        return None
