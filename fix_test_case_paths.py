#!/usr/bin/env python3
"""
Fix test case paths to be absolute
"""
import json
from pathlib import Path

def fix_test_case_paths():
    # Load the test case manifest
    test_case_dir = Path("output_index/test_clip_case")
    manifest_path = test_case_dir / "manifest.json"
    
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)
    
    # Convert relative paths to absolute paths
    for item in manifest:
        if item['path'].startswith('demo_images/'):
            # Convert to absolute path
            item['path'] = str(Path.cwd() / item['path'])
    
    # Save the updated manifest
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    print(f"✅ Updated {len(manifest)} paths to absolute")
    print(f"📁 Manifest saved to {manifest_path}")

if __name__ == "__main__":
    fix_test_case_paths()
