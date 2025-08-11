"""
Embedding helpers for FORCEPS: load ViT and CLIP, preprocess, compute batches.
"""
import torch
from transformers import AutoModel, AutoImageProcessor
from PIL import Image
import numpy as np
import cv2
try:
    from turbojpeg import TurboJPEG
    jpeg = TurboJPEG()
    USE_TURBOJPEG = True
except ImportError:
    USE_TURBOJPEG = False


# optional CLIP import
try:
    import clip
except Exception:
    clip = None

def load_models(vit_model_name="google/vit-base-patch16-224-in21k"):
    proc = AutoImageProcessor.from_pretrained(vit_model_name, use_fast=True)
    vit = AutoModel.from_pretrained(vit_model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        torch.set_num_threads(1)
    except Exception:
        pass
    vit.to(device).eval()
    if device == "cuda":
        vit.half()
    # Use a fixed input size to avoid processor field differences
    size = 224
    from torchvision import transforms
    preprocess_vit = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=proc.image_mean, std=proc.image_std)
    ])
    # CLIP
    if clip is not None:
        clip_model, preprocess_clip = clip.load("ViT-B/32", device=device, jit=False)
        clip_model.eval()
        if device == "cuda":
            clip_model = clip_model.half()
    else:
        clip_model = None
        preprocess_clip = None
    vit_dim = vit.config.hidden_size
    clip_dim = 512 if clip_model is not None else 0
    return vit, clip_model, preprocess_vit, preprocess_clip, vit_dim, clip_dim

def preprocess_image_for_vit_clip(path, preprocess_vit, preprocess_clip):
    try:
        path_str = str(path)
        im = None
        # Fast path for JPEGs
        if USE_TURBOJPEG and path_str.lower().endswith(('.jpg', '.jpeg')):
            with open(path_str, 'rb') as f:
                im = jpeg.decode(f.read(), pixel_format=turbojpeg.TJPF_RGB)
            im = Image.fromarray(im)
        # Fast path for other formats with OpenCV
        elif path_str.lower().endswith(('.png', '.bmp', '.tiff')):
            img_array = cv2.imread(path_str)
            if img_array is not None:
                im = Image.fromarray(cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB))

        # Fallback to PIL
        if im is None:
            im = Image.open(path_str).convert("RGB")

        vit_t = preprocess_vit(im)
        clip_t = preprocess_clip(im) if preprocess_clip is not None else None
        return str(path), im, vit_t, clip_t
    except Exception:
        return None

def compute_batch_embeddings(vit_batch, clip_batch, vit_model, clip_model):
    import torch
    vit_tensor = torch.stack([t for t in vit_batch]).to(next(vit_model.parameters()).device)
    if clip_model is not None and clip_batch:
        clip_tensor = torch.stack([t for t in clip_batch]).to(next(clip_model.parameters()).device)
    else:
        clip_tensor = None
    device = vit_tensor.device
    if device.type == "cuda":
        vit_tensor = vit_tensor.half()
        if clip_tensor is not None:
            clip_tensor = clip_tensor.half()
    with torch.no_grad():
        out_vit = vit_model(vit_tensor)
        emb_vit = out_vit.pooler_output.cpu().float().numpy() if hasattr(out_vit, "pooler_output") and out_vit.pooler_output is not None else out_vit.last_hidden_state[:,0,:].cpu().float().numpy()
        if clip_tensor is not None:
            emb_clip = clip_model.encode_image(clip_tensor).cpu().float().numpy()
        else:
            emb_clip = None
    emb_vit = emb_vit / (np.linalg.norm(emb_vit, axis=1, keepdims=True) + 1e-10)
    if emb_clip is not None:
        emb_clip = emb_clip / (np.linalg.norm(emb_clip, axis=1, keepdims=True) + 1e-10)
        combined = np.concatenate([emb_vit, emb_clip], axis=1)
    else:
        combined = emb_vit
    return combined, emb_clip

def compute_clip_text_embedding(text, clip_model):
    import clip
    import torch
    device = next(clip_model.parameters()).device
    token = clip.tokenize([text]).to(device)
    with torch.no_grad():
        emb = clip_model.encode_text(token).cpu().float().numpy()
    emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-10)
    return emb[0]
