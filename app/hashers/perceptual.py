from pathlib import Path
from typing import Dict, Any
from PIL import Image
import imagehash
from .base import Hasher

class PerceptualHasher(Hasher):
    """Computes perceptual hashes (phash, ahash, dhash)."""
    def __init__(self):
        super().__init__(name="perceptual")

    def compute(self, filepath: Path) -> Dict[str, Any]:
        try:
            im = Image.open(filepath).convert("RGB")
            return {
                "phash": str(imagehash.phash(im)),
                "ahash": str(imagehash.average_hash(im)),
                "dhash": str(imagehash.dhash(im))
            }
        except Exception:
            return {"phash": None, "ahash": None, "dhash": None}

def compute_perceptual_hashes(filepath: Path) -> Dict[str, Any]:
    """
    Standalone utility function to compute perceptual hashes for a single file.
    """
    return PerceptualHasher().compute(filepath)
