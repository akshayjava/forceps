import pytest
from pathlib import Path
from app.hashers import get_hashers
from app.hashers.standard import SHA256Hasher, PerceptualHasher
from tests.test_utils import create_dummy_image

def test_sha256_hasher(tmp_path):
    """
    Tests the SHA256Hasher implementation.
    """
    p = tmp_path / "test.txt"
    content = b"hello world"
    p.write_bytes(content)

    hasher = SHA256Hasher()
    hashes = hasher.compute(p)

    # Pre-computed hash for "hello world"
    expected_hash = "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9"

    assert "sha256" in hashes
    assert hashes["sha256"] == expected_hash

def test_perceptual_hasher(tmp_path):
    """
    Tests the PerceptualHasher implementation by checking for consistency
    and differences between images with patterns.
    """
    # 1. Create test images with patterns
    pattern1 = [([0, 0, 8, 8], "white")]
    pattern2 = [([16, 16, 24, 24], "white")]

    img1_path = create_dummy_image(tmp_path / "img1.png", color="black", pattern=pattern1)
    img2_path = create_dummy_image(tmp_path / "img2.png", color="black", pattern=pattern1) # Identical
    img3_path = create_dummy_image(tmp_path / "img3.png", color="black", pattern=pattern2) # Different

    # 2. Compute hashes
    hasher = PerceptualHasher()
    hashes1 = hasher.compute(img1_path)
    hashes2 = hasher.compute(img2_path)
    hashes3 = hasher.compute(img3_path)

    # 3. Assertions
    # All hashes should be present and be 16-char hex strings
    for h_dict in [hashes1, hashes2, hashes3]:
        assert isinstance(h_dict, dict)
        for hash_name in ["phash", "ahash", "dhash"]:
            assert hash_name in h_dict
            assert isinstance(h_dict[hash_name], str)
            assert len(h_dict[hash_name]) == 16

    # Hashes for identical images should be identical
    assert hashes1 == hashes2

    # Hashes for different images should be different
    assert hashes1 != hashes3

def test_get_hashers_factory():
    """
    Tests that the get_hashers factory returns the correct hasher instances.
    """
    config = ["sha256", "perceptual", "non_existent_hasher"]

    hashers = get_hashers(config)

    assert len(hashers) == 2
    assert isinstance(hashers[0], SHA256Hasher)
    assert isinstance(hashers[1], PerceptualHasher)
