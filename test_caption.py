#!/usr/bin/env python3

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

from llm_ollama import generate_caption_ollama, ollama_installed, model_available

def test_caption():
    print("Testing Ollama caption generation...")
    
    if not ollama_installed():
        print("❌ Ollama not installed")
        return False
        
    print("✅ Ollama is installed")
    
    if not model_available("llava"):
        print("❌ llava model not available")
        return False
        
    print("✅ llava model is available")
    
    # Try to find a sample image
    demo_dir = os.path.join(os.path.dirname(__file__), "demo_images")
    sample_image = None
    
    if os.path.exists(demo_dir):
        for file in os.listdir(demo_dir):
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                sample_image = os.path.join(demo_dir, file)
                break
    
    if not sample_image:
        print("❌ No sample image found in demo_images directory")
        return False
        
    print(f"🖼️ Testing with image: {sample_image}")
    
    caption = generate_caption_ollama(sample_image)
    
    if caption:
        print(f"✅ Caption generated successfully:")
        print(f"📝 {caption}")
        return True
    else:
        print("❌ Failed to generate caption")
        return False

if __name__ == "__main__":
    success = test_caption()
    sys.exit(0 if success else 1)
