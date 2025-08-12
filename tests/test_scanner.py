import pytest
from pathlib import Path
from app.enqueue_jobs import scan_images

def test_scan_images(tmp_path):
    """
    Tests the parallel scan_images function to ensure it finds all valid images
    in a nested directory structure and ignores non-image files.
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

    # Run the scanner
    # Using a small number of workers for the test
    found_paths = scan_images(str(tmp_path), max_workers=2)

    # Assertions
    assert len(found_paths) == len(expected_images)
    assert set(found_paths) == {str(p) for p in expected_images}
