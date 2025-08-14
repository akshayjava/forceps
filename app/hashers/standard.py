import hashlib
from pathlib import Path
from typing import Dict, Any
from .base import Hasher

class SHA256Hasher(Hasher):
    """Computes the SHA-256 hash of a file."""
    def __init__(self):
        super().__init__(name="sha256")

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
