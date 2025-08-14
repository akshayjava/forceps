from typing import List, Dict, Type
from .base import Hasher
from .standard import SHA256Hasher
from .perceptual import PerceptualHasher, compute_perceptual_hashes
# from .photodna import PhotoDNAHasher # Example for when it's added

# The registry maps a name from the config file to the Hasher class
HASHER_REGISTRY: Dict[str, Type[Hasher]] = {
    "sha256": SHA256Hasher,
    "perceptual": PerceptualHasher,
    # "photodna": PhotoDNAHasher,
}

def get_hashers(config: List[str]) -> List[Hasher]:
    """
    Factory function to get a list of instantiated hasher objects
    based on the configuration.
    """
    hashers = []
    for name in config:
        hasher_class = HASHER_REGISTRY.get(name)
        if hasher_class:
            hashers.append(hasher_class())
        else:
            # In a real app, this should probably log a warning.
            print(f"Warning: Hasher '{name}' not found in registry.")
    return hashers
