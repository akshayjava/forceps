import streamlit as st
import os
import sys
import json
import yaml
import time
import numpy as np
import pickle
from pathlib import Path
import logging
import redis
from PIL import Image
import faiss
import torch

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add app directory to sys.path
sys.path.insert(0, os.path.abspath('./app'))

# --- Safe Module Loading ---
def safe_import_modules():
    """Safely import optional modules and set availability flags"""
    global HAS_HASHERS, HAS_EMBEDDINGS, HAS_OLLAMA, HAS_UTILS
    global get_hashers, load_models, compute_batch_embeddings
    global ollama_installed, generate_caption_ollama, model_available, general_ollama_query
    global fingerprint, load_cache, save_cache, read_exif
    
    # Initialize flags
    HAS_HASHERS = False
    HAS_EMBEDDINGS = False
    HAS_OLLAMA = False
    HAS_UTILS = False
    
    # Initialize function references
    get_hashers = None
    load_models = None
    compute_batch_embeddings = None
    ollama_installed = None
    generate_caption_ollama = None
    model_available = None
    general_ollama_query = None
    fingerprint = None
    load_cache = None
    save_cache = None
    read_exif = None
    
    # Try to import hashers
    try:
        from app.hashers import get_hashers
        HAS_HASHERS = True
        st.sidebar.success("✅ Hashers module loaded")
    except ImportError as e:
        st.sidebar.warning(f"❌ Hashers module: {e}")
    
    # Try to import embeddings
    try:
        from app.embeddings import load_models, compute_batch_embeddings
        HAS_EMBEDDINGS = True
        st.sidebar.success("✅ Embeddings module loaded")
    except ImportError as e:
        st.sidebar.warning(f"❌ Embeddings module: {e}")
    
    # Try to import Ollama
    try:
        from app.llm_ollama import ollama_installed, generate_caption_ollama, model_available, general_ollama_query
        HAS_OLLAMA = True
        st.sidebar.success("✅ Ollama module loaded")
    except ImportError as e:
        st.sidebar.warning(f"❌ Ollama module: {e}")
    
    # Try to import utils
    try:
        from app.utils import fingerprint, load_cache, save_cache, read_exif
        HAS_UTILS = True
        st.sidebar.success("✅ Utils module loaded")
    except ImportError as e:
        st.sidebar.warning(f"❌ Utils module: {e}")

# --- Configuration Loading ---
@st.cache_data
def load_config(config_path="app/config.yaml"):
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        st.error(f"Configuration file not found at: {config_path}")
        return None
    except Exception as e:
        st.error(f"Error reading configuration file: {e}")
        return None

config = load_config()
if not config:
    st.stop()

# --- Enhanced Functions ---
def enhanced_image_browser(input_dir):
    """Enhanced image browser with metadata extraction"""
    if not os.path.isdir(input_dir):
        return []
    
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".webp"}
    image_files = []
    
    try:
        for root, dirs, files in os.walk(input_dir):
            for file in files:
                if Path(file).suffix.lower() in image_extensions:
                    file_path = os.path.join(root, file)
                    # Get basic file info
                    try:
                        file_size = os.path.getsize(file_path)
                        mod_time = os.path.getmtime(file_path)
                        image_files.append({
                            'path': file_path,
                            'name': file,
                            'size': file_size,
                            'modified': mod_time
                        })
                    except Exception as e:
                        logger.warning(f"Could not get info for {file_path}: {e}")
                        image_files.append({'path': file_path, 'name': file, 'size': 0, 'modified': 0})
    except Exception as e:
        st.error(f"Error scanning directory: {e}")
    
    return image_files

def enhanced_file_processor(input_dir, output_dir, use_hashers=True, use_embeddings=True):
    """Enhanced file processor with optional ML features"""
    st.info(f"Processing files from {input_dir} to {output_dir}")
    
    image_files = enhanced_image_browser(input_dir)
    if not image_files:
        st.warning("No image files found")
        return None
    
    st.success(f"Found {len(image_files)} images")
    
    # Create output directory
    case_name = f'case_{int(time.time())}'
    case_output_dir = Path(output_dir) / case_name
    case_output_dir.mkdir(parents=True, exist_ok=True)
    
    processed_data = []
    
    # Process with hashers if available
    if use_hashers and HAS_HASHERS:
        st.info("Computing image hashes...")
        try:
            hashers_to_run = config.get('hashing', [])
            hashers = get_hashers(hashers_to_run)
            if hashers:
                for img_info in image_files:
                    try:
                        all_hashes = {}
                        for hasher in hashers:
                            all_hashes.update(hasher.compute(img_info['path']))
                        img_info['hashes'] = all_hashes
                    except Exception as e:
                        logger.warning(f"Could not compute hashes for {img_info['path']}: {e}")
                        img_info['hashes'] = {}
        except Exception as e:
            st.warning(f"Hash computation failed: {e}")
    
    # Process with embeddings if available
    if use_embeddings and HAS_EMBEDDINGS:
        st.info("Computing image embeddings...")
        try:
            vit_model, clip_model, preprocess_vit, preprocess_clip, vit_dim, clip_dim = load_models()
            
            # Process images in batches
            batch_size = 32
            for i in range(0, len(image_files), batch_size):
                batch = image_files[i:i + batch_size]
                batch_paths = [img['path'] for img in batch]
                
                try:
                    # Preprocess images
                    vit_tensors = []
                    for img_path in batch_paths:
                        try:
                            img = Image.open(img_path).convert("RGB")
                            vit_tensor = preprocess_vit(img)
                            vit_tensors.append(vit_tensor)
                        except Exception as e:
                            logger.warning(f"Could not preprocess {img_path}: {e}")
                            vit_tensors.append(None)
                    
                    # Compute embeddings for valid tensors
                    valid_tensors = [t for t in vit_tensors if t is not None]
                    if valid_tensors:
                        query_emb, _ = compute_batch_embeddings(valid_tensors, [], vit_model, None)
                        
                        # Assign embeddings back to image files
                        emb_idx = 0
                        for j, img_info in enumerate(batch):
                            if vit_tensors[j] is not None:
                                img_info['embedding'] = query_emb[emb_idx].astype(np.float32)
                                emb_idx += 1
                
                except Exception as e:
                    st.warning(f"Embedding computation failed for batch {i//batch_size + 1}: {e}")
                
                st.progress((i + batch_size) / len(image_files), text=f"Processed {min(i + batch_size, len(image_files))} embeddings...")
        
        except Exception as e:
            st.warning(f"Embedding computation failed: {e}")
    
    # Save processed data
    try:
        # Save image paths
        image_paths_only = [img['path'] for img in image_files]
        with open(case_output_dir / "image_paths.pkl", "wb") as f:
            pickle.dump(image_paths_only, f)
        
        # Save full metadata
        with open(case_output_dir / "metadata.pkl", "wb") as f:
            pickle.dump(image_files, f)
        
        # Create manifest
        manifest_data = []
        for img in image_files:
            manifest_entry = {
                "path": img['path'],
                "name": img['name'],
                "size": img['size'],
                "modified": img['modified']
            }
            if 'hashes' in img:
                manifest_entry['hashes'] = img['hashes']
            if 'embedding' in img:
                manifest_entry['has_embedding'] = True
            manifest_data.append(manifest_entry)
        
        with open(case_output_dir / "manifest.json", "w") as f:
            json.dump(manifest_data, f, indent=2)
        
        # Build FAISS index if embeddings are available
        if use_embeddings and HAS_EMBEDDINGS:
            embeddings_list = [img['embedding'] for img in image_files if 'embedding' in img]
            if embeddings_list:
                st.info("Building FAISS index...")
                try:
                    embeddings_array = np.array(embeddings_list, dtype=np.float32)
                    n, d = embeddings_array.shape
                    
                    # Create simple flat index
                    index = faiss.IndexFlatL2(d)
                    index.add(embeddings_array)
                    
                    # Save index
                    faiss.write_index(index, str(case_output_dir / "image_index.faiss"))
                    st.success(f"FAISS index built with {n} vectors of dimension {d}")
                except Exception as e:
                    st.warning(f"FAISS index building failed: {e}")
        
        st.success(f"Processed {len(image_files)} files to {case_output_dir}")
        return case_output_dir
        
    except Exception as e:
        st.error(f"Error saving files: {e}")
        return None

def simple_redis_check(host, port):
    """Simple Redis connection check"""
    try:
        r = redis.Redis(host=host, port=port, db=0, decode_responses=True)
        r.ping()
        return True, r
    except Exception as e:
        return False, str(e)

def enhanced_image_search(query_image, case_dir):
    """Enhanced image search using FAISS index"""
    if not HAS_EMBEDDINGS:
        st.error("Embeddings module not available. Cannot perform image search.")
        return
    
    try:
        # Load index and metadata
        index_path = case_dir / "image_index.faiss"
        metadata_path = case_dir / "metadata.pkl"
        
        if not index_path.exists():
            st.error("No FAISS index found. Please rebuild the case with embeddings enabled.")
            return
        
        with open(metadata_path, "rb") as f:
            image_metadata = pickle.load(f)
        
        index = faiss.read_index(str(index_path))
        st.info(f"Loaded index with {index.ntotal} vectors.")
        
        # Process query image
        try:
            vit_model, clip_model, preprocess_vit, preprocess_clip, vit_dim, clip_dim = load_models()
            
            # Preprocess query image
            img = Image.open(query_image).convert("RGB")
            vit_tensor = preprocess_vit(img)
            query_emb, _ = compute_batch_embeddings([vit_tensor], [], vit_model, None)
            query_vector = query_emb[0].astype(np.float32)
            
            # Normalize query vector
            query_vector = query_vector / (np.linalg.norm(query_vector) + 1e-10)
            
            # Perform search
            k = min(10, index.ntotal)  # Top k results
            distances, indices = index.search(np.array([query_vector]), k)
            
            st.subheader(f"Top {k} Similar Images:")
            for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
                if idx >= 0 and idx < len(image_metadata):
                    result_img = image_metadata[idx]
                    result_path = result_img['path']
                    
                    col1, col2 = st.columns([1, 3])
                    with col1:
                        try:
                            st.image(result_path, caption=f"Rank {i+1}", width=150)
                        except Exception as e:
                            st.write(f"Error loading image: {e}")
                    
                    with col2:
                        st.write(f"**Rank {i+1}** - Similarity: {dist:.3f}")
                        st.write(f"**File:** {Path(result_path).name}")
                        st.write(f"**Size:** {result_img.get('size', 'Unknown'):,} bytes")
                        if 'hashes' in result_img and result_img['hashes']:
                            st.write(f"**Hashes:** {len(result_img['hashes'])} computed")
                        if 'embedding' in result_img:
                            st.write("**Embedding:** ✅ Available")
                else:
                    st.write(f"**Rank {i+1}** - Invalid index: {idx}")
        
        except Exception as e:
            st.error(f"Error processing query image: {e}")
    
    except Exception as e:
        st.error(f"Error during image search: {e}")

def enhanced_text_search(query_text, case_dir):
    """Enhanced text search through image metadata"""
    if not HAS_UTILS:
        st.error("Utils module not available. Cannot perform text search.")
        return
    
    try:
        metadata_path = case_dir / "metadata.pkl"
        with open(metadata_path, "rb") as f:
            image_metadata = pickle.load(f)
        
        st.info(f"Searching through {len(image_metadata)} images for: '{query_text}'")
        
        # Simple text search through file names and paths
        query_lower = query_text.lower()
        found_results = []
        
        for img in image_metadata:
            # Search in filename
            if query_lower in img['name'].lower():
                found_results.append(img)
                continue
            
            # Search in full path
            if query_lower in img['path'].lower():
                found_results.append(img)
                continue
            
            # Search in cached metadata if available
            try:
                fp = fingerprint(Path(img['path']))
                cached_data = load_cache(fp)
                if cached_data and "metadata" in cached_data:
                    metadata = cached_data["metadata"]
                    # Search in caption
                    if "caption" in metadata and query_lower in metadata["caption"].lower():
                        found_results.append(img)
                        continue
                    # Search in other metadata fields
                    for key, value in metadata.items():
                        if isinstance(value, str) and query_lower in value.lower():
                            found_results.append(img)
                            break
            except Exception:
                pass
        
        if found_results:
            st.subheader(f"Found {len(found_results)} matching images:")
            cols = st.columns(3)
            for i, result in enumerate(found_results[:9]):  # Show first 9
                col_idx = i % 3
                with cols[col_idx]:
                    try:
                        st.image(result['path'], caption=result['name'], width=150)
                        st.write(f"**{result['name']}**")
                        if 'hashes' in result and result['hashes']:
                            st.write(f"Hashes: {len(result['hashes'])}")
                    except Exception as e:
                        st.write(f"Error loading {result['name']}: {e}")
        else:
            st.info("No matching images found.")
    
    except Exception as e:
        st.error(f"Error during text search: {e}")

# --- Streamlit UI ---
st.set_page_config(layout="wide", page_title="FORCEPS Enhanced")

st.title("FORCEPS Enhanced - With Querying")
st.info("Enhanced version with ML capabilities and querying functionality.")

# Safely import modules after Streamlit is initialized
with st.spinner("Loading modules safely..."):
    safe_import_modules()

st.sidebar.header("Configuration")
input_dir_default = config['data']['input_dir']
input_dir = st.sidebar.text_input("Image Directory", value=input_dir_default)
output_dir_default = config['data']['output_dir']
output_dir = st.sidebar.text_input("Output Directory", value=output_dir_default)

st.sidebar.markdown("---")
st.sidebar.header("System Status")

# Check Redis
redis_host = config['redis']['host']
redis_port = config['redis']['port']
redis_ok, redis_result = simple_redis_check(redis_host, redis_port)

if redis_ok:
    st.sidebar.success(f"✅ Redis: {redis_host}:{redis_port}")
    try:
        total_images = redis_result.get("forceps:stats:total_images") or 0
        st.sidebar.info(f"Total Images: {total_images}")
    except:
        pass
else:
    st.sidebar.error(f"❌ Redis: {redis_result}")

# Check directories
if os.path.isdir(input_dir):
    st.sidebar.success(f"✅ Input: {input_dir}")
else:
    st.sidebar.error(f"❌ Input: {input_dir}")

if os.path.isdir(output_dir):
    st.sidebar.success(f"✅ Output: {output_dir}")
else:
    st.sidebar.error(f"❌ Output: {output_dir}")

st.sidebar.markdown("---")
st.sidebar.header("Module Status")
st.sidebar.write(f"**Hashers**: {'✅' if HAS_HASHERS else '❌'}")
st.sidebar.write(f"**Embeddings**: {'✅' if HAS_EMBEDDINGS else '❌'}")
st.sidebar.write(f"**Ollama**: {'✅' if HAS_OLLAMA else '❌'}")
st.sidebar.write(f"**Utils**: {'✅' if HAS_UTILS else '❌'}")

st.header("1. Enhanced File Processing")
st.write("Process images with optional ML features (hashing, embeddings).")

col1, col2 = st.columns(2)
with col1:
    use_hashers = st.checkbox("Use Hashers", value=HAS_HASHERS, disabled=not HAS_HASHERS)
    if not HAS_HASHERS:
        st.info("Hashers module not available")
with col2:
    use_embeddings = st.checkbox("Use Embeddings", value=HAS_EMBEDDINGS, disabled=not HAS_EMBEDDINGS)
    if not HAS_EMBEDDINGS:
        st.info("Embeddings module not available")

if st.button("Process Images with ML Features"):
    if not os.path.isdir(input_dir):
        st.error(f"Input directory '{input_dir}' does not exist.")
    else:
        with st.spinner("Processing images with ML features..."):
            case_dir = enhanced_file_processor(input_dir, output_dir, use_hashers, use_embeddings)
            if case_dir:
                st.session_state.case_dir = case_dir
                st.success("Enhanced processing completed!")

st.header("2. Image Search")
st.write("Search for similar images using the FAISS index.")

if 'case_dir' in st.session_state and st.session_state.case_dir:
    case_dir = st.session_state.case_dir
    
    # Check if index exists
    index_path = case_dir / "image_index.faiss"
    if index_path.exists():
        st.success(f"FAISS index available: {index_path}")
        
        uploaded_file = st.file_uploader("Choose an image to search for...", type=["jpg", "jpeg", "png", "bmp", "gif", "tiff", "webp"])
        
        if uploaded_file is not None:
            st.image(uploaded_file, caption='Query Image', use_container_width=True)
            
            if st.button("Search for Similar Images"):
                enhanced_image_search(uploaded_file, case_dir)
    else:
        st.warning("No FAISS index found. Please process images with embeddings enabled.")
else:
    st.info("No case loaded. Process images first to enable search functionality.")

st.header("3. Text Search")
st.write("Search through image metadata and cached information.")

if 'case_dir' in st.session_state and st.session_state.case_dir:
    case_dir = st.session_state.case_dir
    
    text_query = st.text_input("Enter search terms:")
    if text_query:
        if st.button("Search"):
            enhanced_text_search(text_query, case_dir)
else:
    st.info("No case loaded. Process images first to enable text search.")

st.header("4. Image Browser")
st.write("Browse and view images in your input directory.")

if os.path.isdir(input_dir):
    image_files = enhanced_image_browser(input_dir)
    
    if image_files:
        st.success(f"Found {len(image_files)} images")
        
        # Show file list with metadata
        if st.checkbox("Show detailed file list"):
            for i, img in enumerate(image_files[:20]):  # Show first 20
                st.write(f"{i+1}. {img['name']} ({img['size']:,} bytes)")
        
        # Show sample images
        if st.button("Show Sample Images"):
            cols = st.columns(3)
            for i, img in enumerate(image_files[:9]):  # Show first 9
                col_idx = i % 3
                with cols[col_idx]:
                    try:
                        st.image(img['path'], caption=img['name'], width=150)
                        st.write(f"**{img['name']}**")
                        st.write(f"Size: {img['size']:,} bytes")
                    except Exception as e:
                        st.write(f"Error loading {img['name']}: {e}")
    else:
        st.info("No image files found in the input directory.")
else:
    st.warning(f"Input directory '{input_dir}' does not exist.")

st.header("5. System Information")
st.write("Current system status and capabilities.")

col1, col2 = st.columns(2)
with col1:
    st.subheader("Python Environment")
    st.write(f"**Python Version**: {sys.version}")
    st.write(f"**Working Directory**: {os.getcwd()}")
    st.write(f"**Input Directory**: {input_dir}")
    st.write(f"**Output Directory**: {output_dir}")
    
with col2:
    st.subheader("Available Libraries")
    st.write(f"**FAISS**: {'Available' if faiss else 'Not Available'}")
    st.write(f"**PyTorch**: {'Available' if torch else 'Not Available'}")
    st.write("✅ PIL (Pillow)")
    st.write("✅ Redis")
    st.write("✅ NumPy")
    st.write("✅ Pathlib")

st.sidebar.markdown("---")
if st.sidebar.button("Exit App"):
    st.stop()
