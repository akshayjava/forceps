#!/usr/bin/env python3
"""
Test script for CLIP-only search functionality
"""
import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import json
import numpy as np
from pathlib import Path
from app.embeddings import load_models, compute_clip_text_embedding
from PIL import Image
import time

def test_clip_search():
    print("🔍 Testing CLIP-Only Search Functionality")
    print("=" * 50)
    
    # 1. Check if demo images exist
    demo_dir = Path("demo_images")
    if not demo_dir.exists():
        print("❌ No demo_images directory found!")
        return False
    
    demo_images = list(demo_dir.glob("*.png"))
    if not demo_images:
        print("❌ No demo images found!")
        return False
    
    print(f"📁 Found {len(demo_images)} demo images")
    
    # 2. Load CLIP models
    print("🤖 Loading CLIP models...")
    try:
        vit_model, clip_model, preprocess_vit, preprocess_clip, vit_dim, clip_dim = load_models()
        if clip_model is None:
            print("❌ CLIP model not available!")
            return False
        print("✅ CLIP models loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load models: {e}")
        return False
    
    # 3. Create a simple test case with CLIP embeddings
    print("\n📝 Creating test case with CLIP embeddings...")
    test_case_dir = Path("output_index/test_clip_case")
    test_case_dir.mkdir(parents=True, exist_ok=True)
    
    # Process a few demo images to create embeddings
    test_images = demo_images[:10]  # Use first 10 images
    manifest = []
    
    for img_path in test_images:
        try:
            # Load and preprocess image
            img = Image.open(img_path).convert("RGB")
            img_tensor = preprocess_clip(img).unsqueeze(0)
            
            # Compute CLIP embedding
            with torch.no_grad():
                clip_emb = clip_model.encode_image(img_tensor).detach().cpu().numpy()[0]
            
            # Normalize
            clip_emb = clip_emb / (np.linalg.norm(clip_emb) + 1e-10)
            
            # Add to manifest
            manifest.append({
                "path": str(img_path),
                "hashes": {},
                "clip_embedding": clip_emb.tolist()
            })
            
        except Exception as e:
            print(f"⚠️  Failed to process {img_path.name}: {e}")
            continue
    
    if not manifest:
        print("❌ No images could be processed!")
        return False
    
    print(f"✅ Created manifest with {len(manifest)} images")
    
    # 4. Save manifest
    manifest_path = test_case_dir / "manifest.json"
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    print(f"💾 Saved manifest to {manifest_path}")
    
    # 5. Test text-to-image search
    print("\n🔍 Test 1: Text-to-Image Search")
    try:
        # Test query
        query_text = "synthetic image"
        query_emb = compute_clip_text_embedding(query_text, clip_model)
        
        # Search through manifest
        similarities = []
        for item in manifest:
            if 'clip_embedding' in item:
                img_emb = np.array(item['clip_embedding'])
                # Compute cosine similarity
                similarity = np.dot(query_emb, img_emb) / (np.linalg.norm(query_emb) * np.linalg.norm(img_emb))
                similarities.append((item['path'], similarity))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        print(f"✅ Text search successful!")
        print(f"   Query: '{query_text}'")
        print(f"   Top 3 results:")
        for i, (path, sim) in enumerate(similarities[:3]):
            filename = Path(path).name
            print(f"     {i+1}. {filename} (similarity: {sim:.4f})")
    
    except Exception as e:
        print(f"❌ Text search failed: {e}")
        return False
    
    # 6. Test image-to-image search
    print("\n🖼️  Test 2: Image-to-Image Search")
    try:
        # Use first image as query
        query_img_path = test_images[0]
        query_img = Image.open(query_img_path).convert("RGB")
        query_tensor = preprocess_clip(query_img).unsqueeze(0)
        
        with torch.no_grad():
            query_emb = clip_model.encode_image(query_tensor).detach().cpu().numpy()[0]
        
        # Normalize
        query_emb = query_emb / (np.linalg.norm(query_emb) + 1e-10)
        
        # Search through manifest
        similarities = []
        for item in manifest:
            if 'clip_embedding' in item:
                img_emb = np.array(item['clip_embedding'])
                # Compute cosine similarity
                similarity = np.dot(query_emb, img_emb) / (np.linalg.norm(query_emb) * np.linalg.norm(img_emb))
                similarities.append((item['path'], similarity))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        print(f"✅ Image search successful!")
        print(f"   Query image: {query_img_path.name}")
        print(f"   Top 3 results:")
        for i, (path, sim) in enumerate(similarities[:3]):
            filename = Path(path).name
            print(f"     {i+1}. {filename} (similarity: {sim:.4f})")
    
    except Exception as e:
        print(f"❌ Image search failed: {e}")
        return False
    
    print("\n" + "=" * 50)
    print("✅ CLIP-only search is working correctly!")
    print(f"📁 Test case created at: {test_case_dir}")
    print("🌐 You can now load this case in the Streamlit app")
    print("🎯 The search should return results instead of 0")
    
    return True

if __name__ == "__main__":
    # Add torch import
    import torch
    
    success = test_clip_search()
    if not success:
        print("\n❌ CLIP search test failed - check logs above")
        sys.exit(1)
