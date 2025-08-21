#!/usr/bin/env python3
"""
Debug script to test FORCEPS search functionality
"""
import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import json
import numpy as np
import faiss
from pathlib import Path
from PIL import Image

def debug_search():
    print("🔍 Debugging FORCEPS Search")
    print("=" * 50)
    
    # 1. Check case directory
    case_dir = Path("output_index/local-1755750182")
    if not case_dir.exists():
        print("❌ Case directory not found!")
        return False
    
    print(f"✅ Case directory: {case_dir}")
    
    # 2. Check manifest
    manifest_path = case_dir / "manifest.json"
    if not manifest_path.exists():
        print("❌ Manifest not found!")
        return False
    
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)
    
    print(f"✅ Manifest loaded: {len(manifest)} entries")
    
    # 3. Check FAISS index
    index_path = case_dir / "combined.index"
    if not index_path.exists():
        print("❌ FAISS index not found!")
        return False
    
    try:
        index = faiss.read_index(str(index_path))
        print(f"✅ FAISS index loaded: {index.ntotal} vectors, dimension: {index.d}")
    except Exception as e:
        print(f"❌ Failed to load FAISS index: {e}")
        return False
    
    # 4. Check if image paths exist
    print("\n🔍 Checking image paths...")
    existing_paths = []
    missing_paths = []
    
    for i, item in enumerate(manifest[:10]):  # Check first 10
        path = Path(item['path'])
        if path.exists():
            existing_paths.append(str(path))
        else:
            missing_paths.append(str(path))
    
    print(f"✅ Existing paths: {len(existing_paths)}")
    print(f"❌ Missing paths: {len(missing_paths)}")
    
    if missing_paths:
        print("Sample missing paths:")
        for path in missing_paths[:3]:
            print(f"  - {path}")
    
    # 5. Test FAISS search with dummy query
    print("\n🧠 Testing FAISS search...")
    try:
        # Create a dummy query vector of the right dimension
        dummy_query = np.random.random(index.d).astype(np.float32)
        dummy_query = dummy_query / (np.linalg.norm(dummy_query) + 1e-10)
        
        # Search
        D, I = index.search(np.array([dummy_query]), 10)
        
        print(f"✅ FAISS search successful!")
        print(f"   Distances: {D[0][:5]}")
        print(f"   Indices: {I[0][:5]}")
        
        # Check if indices are valid
        valid_indices = [i for i in I[0] if 0 <= i < len(manifest)]
        print(f"   Valid indices: {len(valid_indices)}/{len(I[0])}")
        
        if valid_indices:
            print("   Sample results:")
            for i, idx in enumerate(valid_indices[:3]):
                result_path = manifest[idx]['path']
                print(f"     {i+1}. {Path(result_path).name}")
        
    except Exception as e:
        print(f"❌ FAISS search failed: {e}")
        import traceback
        traceback.print_exc()
    
    # 6. Check embeddings file
    print("\n📊 Checking embeddings file...")
    embeddings_path = case_dir / "embeddings_combined.mmap"
    if embeddings_path.exists():
        try:
            # Try to load embeddings
            num_embeddings = len(manifest)
            embedding_dim = index.d
            embeddings = np.memmap(embeddings_path, dtype='float32', mode='r', 
                                 shape=(num_embeddings, embedding_dim))
            print(f"✅ Embeddings loaded: {embeddings.shape}")
            
            # Test accessing embeddings
            if len(valid_indices) > 0:
                sample_emb = embeddings[valid_indices[0]]
                print(f"   Sample embedding shape: {sample_emb.shape}")
                print(f"   Sample embedding norm: {np.linalg.norm(sample_emb):.4f}")
        
        except Exception as e:
            print(f"❌ Failed to load embeddings: {e}")
    else:
        print("⚠️  No embeddings file found")
    
    print("\n" + "=" * 50)
    print("📋 Summary:")
    print(f"   Case loaded: ✅")
    print(f"   Manifest: {len(manifest)} entries")
    print(f"   FAISS index: {index.ntotal} vectors")
    print(f"   Image paths exist: {len(existing_paths)}/{len(manifest[:10])}")
    
    if len(existing_paths) == 0:
        print("\n🚨 MAIN ISSUE: All image paths in manifest are missing!")
        print("   This case was built on a different machine with different file paths.")
        print("   Solutions:")
        print("   1. Rebuild the case on this machine")
        print("   2. Copy the actual image files to match the paths")
        print("   3. Update the manifest with correct paths")
    
    return True

if __name__ == "__main__":
    debug_search()
