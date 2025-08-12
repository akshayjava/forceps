import pytest
import numpy as np
import faiss
import argparse

from app.build_index import build_index_for_embeddings

def test_build_index_for_embeddings():
    """
    Tests that the FAISS index building function works correctly.
    """
    # 1. Create mock data and arguments
    num_embeddings = 500
    embedding_dim = 128
    embeddings = np.random.rand(num_embeddings, embedding_dim).astype(np.float32)

    # Mock args for the function
    args = argparse.Namespace(
        use_pca=False,
        pca_dim=64, # Not used if use_pca is False
        ivf_nlist=4, # Small number for a small test dataset
        pq_m=8, # Must be a divisor of embedding_dim
        train_samples=256, # Must be >= 256 for PQ
        add_batch=256
    )

    # 2. Call the function to build the index
    index, pca_matrix = build_index_for_embeddings(embeddings, embedding_dim, num_embeddings, args)

    # 3. Assertions
    assert isinstance(index, faiss.IndexIVFPQ)
    assert index.ntotal == num_embeddings
    assert index.d == embedding_dim
    assert pca_matrix is None # Since use_pca was False

def test_build_index_with_pca(tmp_path):
    """
    Tests that the FAISS index building function works correctly with PCA.
    """
    # 1. Create mock data and arguments
    num_embeddings = 500
    embedding_dim = 128
    pca_dim = 32
    embeddings = np.random.rand(num_embeddings, embedding_dim).astype(np.float32)

    args = argparse.Namespace(
        use_pca=True,
        pca_dim=pca_dim,
        ivf_nlist=4,
        pq_m=16, # Divisor of 32
        train_samples=256, # Must be >= 256 for PQ
        add_batch=256
    )

    # 2. Call the function to build the index
    index, pca_matrix = build_index_for_embeddings(embeddings, embedding_dim, num_embeddings, args)

    # 3. Assertions
    assert isinstance(index, faiss.IndexIVFPQ)
    assert isinstance(pca_matrix, faiss.PCAMatrix)
    assert index.ntotal == num_embeddings
    assert index.d == pca_dim # Dimension should be reduced
    assert pca_matrix.d_out == pca_dim
