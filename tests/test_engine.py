import pytest
import torch
import numpy as np
import argparse
from unittest.mock import MagicMock, patch

from app.engine import ForcepsDataset, compute_embeddings_for_job, collate_fn
from tests.test_utils import create_dummy_image

# Mock preprocessor functions for testing
def mock_preprocess(img):
    return torch.randn(3, 224, 224)

def test_forceps_dataset(tmp_path):
    """
    Tests that the ForcepsDataset correctly loads an image and applies transforms.
    """
    img_path = create_dummy_image(tmp_path / "test.png")

    dataset = ForcepsDataset([str(img_path)], mock_preprocess, mock_preprocess)
    assert len(dataset) == 1

    vit_tensor, clip_tensor, path = dataset[0]

    assert path == str(img_path)
    assert isinstance(vit_tensor, torch.Tensor)
    assert vit_tensor.shape == (3, 224, 224)
    assert isinstance(clip_tensor, torch.Tensor)
    # The dataset returns an empty tensor if clip preprocess is None, but here it's a mock
    assert clip_tensor.shape == (3, 224, 224)

def test_collate_fn(tmp_path):
    """
    Tests that the collate_fn correctly batches items and handles None values.
    """
    img_path1 = str(create_dummy_image(tmp_path / "test1.png"))
    img_path2 = str(create_dummy_image(tmp_path / "test2.png"))

    # Batch with one valid item and one failed item (represented by None)
    batch = [
        (torch.randn(3, 224, 224), torch.randn(3, 224, 224), img_path1),
        (None, None, "bad_path.png") # Simulate a failed load in the dataset
    ]

    vit_batch, clip_batch, paths = collate_fn(batch)

    assert len(paths) == 1
    assert paths[0] == img_path1
    assert vit_batch.shape == (1, 3, 224, 224)
    assert clip_batch.shape == (1, 3, 224, 224)

def test_compute_embeddings_for_job(tmp_path):
    """
    Tests the main worker function with mocked ONNX sessions.
    """
    # 1. Setup mock models and args
    mock_vit_session = MagicMock()
    mock_clip_session = MagicMock()

    # Configure the 'run' method of the mock sessions to return fake embeddings
    batch_size = 2
    vit_dim, clip_dim = 768, 512
    fake_vit_embs = np.random.rand(batch_size, vit_dim).astype(np.float32)
    fake_clip_embs = np.random.rand(batch_size, clip_dim).astype(np.float32)
    mock_vit_session.run.return_value = [fake_vit_embs]
    mock_clip_session.run.return_value = [fake_clip_embs]

    # Mock the get_inputs and get_outputs methods
    mock_vit_session.get_inputs.return_value = [MagicMock(name="input")]
    mock_vit_session.get_outputs.return_value = [MagicMock(name="output")]
    mock_clip_session.get_inputs.return_value = [MagicMock(name="input")]
    mock_clip_session.get_outputs.return_value = [MagicMock(name="output")]

    models = {
        "vit_session": mock_vit_session,
        "clip_session": mock_clip_session,
        "preprocess_vit": mock_preprocess,
        "preprocess_clip": mock_preprocess,
        "vit_dim": vit_dim,
        "clip_dim": clip_dim
    }

    # Mock args namespace
    args = argparse.Namespace(
        batch_size=batch_size,
        max_workers=0 # Use 0 workers to avoid multiprocessing issues in tests
    )

    # 2. Create dummy image files
    img_paths = [
        str(create_dummy_image(tmp_path / "img1.png")),
        str(create_dummy_image(tmp_path / "img2.png"))
    ]

    # 3. Call the function
    results = compute_embeddings_for_job(img_paths, models, args)

    # 4. Assertions
    assert len(results) == 2

    # Check the first result
    res1 = results[0]
    assert res1["path"] == img_paths[0]
    assert "combined_emb" in res1
    assert "clip_emb" in res1
    assert len(res1["combined_emb"]) == vit_dim + clip_dim
    assert len(res1["clip_emb"]) == clip_dim

    # Check that the mock was called
    mock_vit_session.run.assert_called_once()
    mock_clip_session.run.assert_called_once()
