import hashlib
from pathlib import Path
from typing import Dict, Any
from PIL import Image
import imagehash
from app.utils import Hasher

class SHA256Hasher(Hasher):
    """Computes the SHA-256 hash of a file."""
    def compute(self, filepath: Path) -> Dict[str, Any]:
        h = hashlib.sha256()
        b = bytearray(128 * 1024)
        mv = memoryview(b)
        try:
            with open(filepath, 'rb', buffering=0) as f:
                while n := f.readinto(mv):
                    h.update(mv[:n])
            return {"sha256": h.hexdigest()}
        except Exception:
            return {"sha256": None}

class PerceptualHasher(Hasher):
    """Computes perceptual hashes (phash, ahash, dhash)."""
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
