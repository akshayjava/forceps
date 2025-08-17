#!/usr/bin/env python3
"""
Test script for FORCEPS search functionality
"""
import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import json
import numpy as np
import faiss
from pathlib import Path
from app.embeddings import load_models, compute_batch_embeddings
from PIL import Image
import time

def test_forceps_search():
    print("🔍 Testing FORCEPS Search Functionality")
    print("=" * 50)
    
    # 1. Load a completed case
    case_dir = Path("output_index/local-1755467462")
    if not case_dir.exists():
        print("❌ No completed case found!")
        return False
        
    print(f"📁 Loading case: {case_dir}")
    
    # 2. Load the FAISS index
    index_path = case_dir / "combined.index"
    if not index_path.exists():
        print("❌ No FAISS index found!")
        return False
        
    print("🧠 Loading FAISS index...")
    index = faiss.read_index(str(index_path))
    print(f"✅ Loaded index with {index.ntotal} vectors")
    
    # 3. Load the manifest
    manifest_path = case_dir / "manifest.json" 
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)
    print(f"📝 Loaded manifest with {len(manifest)} entries")
    
    # 4. Load models for query
    print("🤖 Loading models...")
    vit_model, clip_model, preprocess_vit, preprocess_clip, vit_dim, clip_dim = load_models()
    print("✅ Models loaded")
    
    # 5. Test search with the first image as a query
    if manifest:
        test_image_path = manifest[0]['path']
        print(f"🖼️  Testing search with: {Path(test_image_path).name}")
        
        try:
            # Load and preprocess the test image
            img = Image.open(test_image_path).convert("RGB")
            vit_tensor = preprocess_vit(img)
            
            # Compute embedding 
            query_emb, _ = compute_batch_embeddings([vit_tensor], [], vit_model, None)
            query_vector = query_emb[0].astype(np.float32)
            
            # Normalize for cosine similarity
            query_vector = query_vector / (np.linalg.norm(query_vector) + 1e-10)
            
            # Search
            start_time = time.time()
            distances, indices = index.search(np.array([query_vector]), 10)
            search_time = time.time() - start_time
            
            print(f"⚡ Search completed in {search_time:.3f} seconds")
            print(f"🎯 Top results:")
            
            for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
                if idx >= 0 and idx < len(manifest):
                    result_path = manifest[idx]['path']
                    print(f"   {i+1}. {Path(result_path).name} (similarity: {dist:.3f})")
            
            return True
            
        except Exception as e:
            print(f"❌ Search test failed: {e}")
            return False
    
    print("❌ No images in manifest to test with")
    return False

def check_processing_progress():
    """Check the current processing progress"""
    import redis
    
    try:
        r = redis.Redis(host='127.0.0.1', port=6379, db=0)
        total = int(r.get("forceps:stats:total_images") or 0)
        done = int(r.get("forceps:stats:embeddings_done") or 0)
        captions = int(r.get("forceps:stats:captions_done") or 0)
        queue_len = r.llen("forceps:job_queue")
        
        print("\n📊 Current Processing Progress:")
        print(f"   Total images: {total}")
        print(f"   Embeddings done: {done} ({done/total*100:.1f}%)" if total > 0 else "   Embeddings done: 0")
        print(f"   Captions done: {captions}")
        print(f"   Jobs remaining in queue: {queue_len}")
        
        return done, total, queue_len
        
    except Exception as e:
        print(f"❌ Could not check progress: {e}")
        return 0, 0, 0

if __name__ == "__main__":
    # Check progress first
    done, total, queue_len = check_processing_progress()
    
    # Test search functionality
    success = test_forceps_search()
    
    print("\n" + "=" * 50)
    if success:
        print("✅ FORCEPS search functionality is working!")
        print("🌐 Web app is running at: http://localhost:8501")
        print("🎯 You can now:")
        print("   - Upload images for similarity search")
        print("   - Use natural language queries (if CLIP is loaded)")
        print("   - Browse and bookmark results")
        print("   - Export reports")
    else:
        print("❌ Search test failed - check logs above")
        
    if queue_len > 0:
        print(f"\n⏳ Processing is still ongoing ({queue_len} jobs remaining)")
        print("   More images will become searchable as processing completes")
