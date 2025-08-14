import abc
from pathlib import Path
from typing import Dict, Any

class Hasher(abc.ABC):
    """Abstract base class for all hashers."""
    def __init__(self, name: str):
        self.name = name

    @abc.abstractmethod
    def compute(self, filepath: Path) -> Dict[str, Any]:
        """
        Compute hashes for a given file.

        Args:
            filepath: Path to the file.

        Returns:
            A dictionary where keys are hash names (e.g., "sha256")
            and values are the computed hash values.
        """
        pass
