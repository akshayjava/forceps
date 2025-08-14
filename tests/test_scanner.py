import pytest
from pathlib import Path
from app.enqueue_jobs import discover_and_process_files
from app.hashers.base import Hasher

# Mock Hasher for testing purposes
class MockHasher(Hasher):
    def __init__(self, hash_name="mock_hash", hash_value="mock_value"):
        self.hash_name = hash_name
        self.hash_value = hash_value
        super().__init__(name=hash_name)

    def compute(self, file_path):
        return {self.hash_name: self.hash_value}

def test_discover_and_process_files(tmp_path):
    """
    Tests the discover_and_process_files function to ensure it finds all valid images,
    processes them with hashers, and ignores non-image files.
    """
    # Create a nested directory structure
    (tmp_path / "sub1").mkdir()
    (tmp_path / "sub1" / "sub2").mkdir()

    # Create a variety of files
    expected_images = {
        tmp_path / "img1.jpg",
        tmp_path / "img2.png",
        tmp_path / "sub1" / "img3.jpeg",
        tmp_path / "sub1" / "sub2" / "img4.gif",
        tmp_path / "sub1" / "sub2" / "img5.BMP", # Uppercase extension
    }

    other_files = {
        tmp_path / "document.txt",
        tmp_path / "sub1" / "archive.zip",
        tmp_path / "sub1" / "sub2" / "data.json",
        tmp_path / "no_extension_file",
    }

    # Create the files on disk
    for img_path in expected_images:
        img_path.touch()
    for other_path in other_files:
        other_path.touch()

    # Instantiate mock hashers
    mock_hasher1 = MockHasher("hash1", "value1")
    mock_hasher2 = MockHasher("hash2", "value2")
    hashers = [mock_hasher1, mock_hasher2]

    # Run the function
    results = discover_and_process_files(str(tmp_path), hashers, max_workers=2)

    # Assertions
    found_paths = {item['path'] for item in results}

    assert len(results) == len(expected_images)
    assert set(found_paths) == {str(p) for p in expected_images}

    # Check that each result has the correct hashes
    for item in results:
        assert "hashes" in item
        assert item["hashes"]["hash1"] == "value1"
        assert item["hashes"]["hash2"] == "value2"
