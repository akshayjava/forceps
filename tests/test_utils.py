import pytest
from pathlib import Path
import os
import time
from PIL import Image
from app.utils import fingerprint, is_image_file, compute_perceptual_hashes

def create_dummy_image(path: Path, size=(32, 32), color="red"):
    """Helper function to create a simple image file."""
    img = Image.new("RGB", size, color)
    img.save(path, "PNG")
    return path

def test_fingerprint(tmp_path):
    """
    Tests that the fingerprint function correctly identifies a file
    based on its metadata (size and modification time).
    """
    # 1. Create a dummy file
    d = tmp_path / "sub"
    d.mkdir()
    p = d / "test.txt"
    p.write_text("hello")

    # 2. Get initial stats and fingerprint
    stats1 = p.stat()
    fp1 = fingerprint(p)

    assert fp1 == f"{stats1.st_mtime_ns}-{stats1.st_size}"

    # 3. Modify the file and check that the fingerprint changes
    time.sleep(0.01) # Ensure mtime changes
    p.write_text("world")
    stats2 = p.stat()
    fp2 = fingerprint(p)

    assert fp2 != fp1
    assert fp2 == f"{stats2.st_mtime_ns}-{stats2.st_size}"

def test_is_image_file(tmp_path):
    """
    Tests that is_image_file correctly distinguishes images from other files.
    """
    image_path = create_dummy_image(tmp_path / "test.png")
    text_path = tmp_path / "test.txt"
    text_path.write_text("not an image")

    assert is_image_file(image_path) is True
    assert is_image_file(text_path) is False

def test_compute_perceptual_hashes(tmp_path):
    """
    Tests that perceptual hashes are computed correctly.
    """
    image_path = create_dummy_image(tmp_path / "test.png")
    hashes = compute_perceptual_hashes(image_path)

    assert isinstance(hashes, dict)
    assert "phash" in hashes
    assert "ahash" in hashes
    assert "dhash" in hashes

    # Check that hashes are strings (or None, though they shouldn't be for a valid image)
    assert isinstance(hashes["phash"], str)
    assert isinstance(hashes["ahash"], str)
    assert isinstance(hashes["dhash"], str)

    # Check for expected length of hex hashes
    assert len(hashes["phash"]) == 16
    assert len(hashes["ahash"]) == 16
    assert len(hashes["dhash"]) == 16
