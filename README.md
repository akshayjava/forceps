# FORCEPS — Forensic Optimized Retrieval & Clustering of Evidence via Perceptual Search

## Overview
FORCEPS is a forensic image similarity and natural-language search tool designed for massive-scale, high-speed, and private on-site investigations. It is built on a scalable, distributed architecture to handle terabyte-sized datasets.

### Key Features
- **Distributed Architecture:** Uses a Redis queue to distribute work to multiple parallel workers for extreme scalability.
- **Case Management:** Organizes analysis into distinct cases, with all output stored in case-specific directories.
- **Forensic Hashing:** Employs a pluggable framework to compute multiple cryptographic (SHA-256) and perceptual (pHash, aHash, dHash) hashes for each piece of evidence.
- **Hybrid Search:** Combines semantic vector search (FAISS) with traditional keyword search (Whoosh) for highly accurate and relevant results.
- **Automatic Clustering:** Leverages machine learning to automatically group search results into visual themes, helping investigators quickly identify patterns.
- **Optimized Models:** Supports GPU-accelerated ONNX models for high-speed embedding computation.
- **Comprehensive Reporting:** Generates detailed PDF and CSV reports for bookmarked evidence, including all hashes, metadata, and user-provided notes.
- **All processing is local:** No data leaves your environment.

## User Interface Mock-up
```
+----------------------------------------------------------------------------------------------------------------------+
| [FORCEPS] Forensic Optimized Retrieval & Clustering of Evidence via Perceptual Search                                |
+----------------------------------------------------------------------------------------------------------------------+
| [ Sidebar ]                                        | [ Search & Results ]  [ Reporting ]                               |
|                                                    |-------------------------------------------------------------------+
| [FORCEPS Indexing Controls]                        | [Search]                                                          |
|  Folder to index: [ /path/to/images      ]         |  Natural language query: [ a dog near a red car          ]         |
|  [ Start Indexing Job ]                            |  [ Run Search ]                                                   |
|                                                    |                                                                   |
| [Backend Controls]                                 | [Top results]                                                     |
|  Redis Host: [ localhost ]                         |  +----------------+  +----------------+  +----------------+       |
|  Redis Port: [ 6379      ]                         |  | [Image]        |  | [Image]        |  | [Image]        |       |
|                                                    |  | img_001.jpg    |  | img_002.jpg    |  | ...            |       |
| [Load Case for Searching]                          |  | [Show Details]v|  | [Show Details]v|  | [Show Details]v|       |
|  Cases Directory: [ output_index ]                 |  | [Manage Bkmk]v |  | [Manage Bkmk]v |  | [Manage Bkmk]v |       |
|  Select Case: [ case-001                v]         |  +----------------+  +----------------+  +----------------+       |
|  [ Load Selected Case ]                            |                                                                   |
|                                                    | ---                                                               |
|                                                    | [Cluster Analysis]                                                |
|                                                    |  Number of Clusters: [ 2 ----o--------------- 50 ] (10)           |
|                                                    |  [ Cluster Displayed Results ]                                    |
+----------------------------------------------------------------------------------------------------------------------+
```

## How It Works: Architecture
The system is composed of several key components that work together:

1.  **Redis:** A message broker that manages the queue of images to be processed.
2.  **Enqueuer (`enqueue_jobs.py`):** A script that scans a directory, computes forensic hashes, and populates the Redis job queue.
3.  **Worker (`engine.py`):** The core processing engine. You can run many workers in parallel. Each worker computes embeddings for images and pushes the results to a results queue.
4.  **Index Builder (`build_index.py`):** A script that consumes results, aggregates all embeddings, and builds the final search indexes (FAISS for vectors, Whoosh for text).
5.  **User Interface (`main.py`):** A Streamlit application for controlling the backend and analyzing results.

## Installation & Setup

### Option 1: Running with Docker Compose (Recommended for Production)
The easiest way to run the entire FORCEPS stack is with `docker-compose`.

#### 1. Prerequisites
- **Install Docker and Docker Compose.**
- **NVIDIA GPU Users:** Ensure you have the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) installed.
- **Prepare your data:** Create host directories for your images (e.g., `./images`) and for the output (e.g., `./output_index`). The `docker-compose.yml` file maps these to the correct paths inside the containers.
- **(Optional) Convert models to ONNX:** For maximum performance, run `python app/convert_models.py --output_dir models/onnx`.

#### 2. Launch the Application
From the root of the project directory, run:
```bash
docker-compose up --build
```
To run multiple workers for faster processing, use the `--scale` flag:
```bash
docker-compose up --build --scale worker=4
```

### Option 2: Command Line Setup (for Development/Testing)

#### 1. Prerequisites
- **Python 3.9+**
- **Redis Server** installed and running
- **Git** for version control
- **Required libraries** (see requirements.txt)

#### 2. Clone the Repository
```bash
git clone https://github.com/akshayjava/forceps.git
cd foreceps
```

#### 3. Create a Virtual Environment
```bash
python -m venv venv_forceps
source venv_forceps/bin/activate  # On Windows: venv_forceps\Scripts\activate
```

#### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

#### 5. Setup Redis
Ensure Redis is running on localhost:6379 (default):
```bash
# On macOS with Homebrew
brew install redis
brew services start redis

# On Ubuntu/Debian
sudo apt install redis-server
sudo systemctl start redis-server

# Verify Redis is running
redis-cli ping  # Should return PONG
```

## Command-Line End-to-End Workflow
This section describes the recommended workflow for indexing and querying images entirely from the command line on a single machine. This is the simplest and fastest way to get started with FORCEPS.

### Step 1: Indexing Images (`run_cli.py`)
The `run_cli.py` script is a standalone tool for quickly creating a searchable index from a directory of images. It computes image embeddings, builds a FAISS index, and saves metadata.

**Usage:**
```bash
# Ensure you are in the project's root directory
# Make sure required packages are installed:
# pip install torch torchvision transformers faiss-cpu pillow tqdm opencv-python

PYTHONPATH=. python3 run_cli.py \
  --image_dir "/path/to/your/images" \
  --output_dir "index_output" \
  --device auto \
  --batch_size 16
```

**Optional: Generate Text Captions**
If you have `ollama` installed with the `llava` model, you can automatically generate descriptive captions for each image. This is required for text-based search.
```bash
# Add the --captions flag to the command above
PYTHONPATH=. python3 run_cli.py \
  --image_dir "/path/to/your/images" \
  --output_dir "index_output" \
  --captions
```

This command will create the `index_output` directory (or the name you specified) containing the following files:
- `image_index.faiss`: The FAISS vector index for similarity search.
- `image_paths.pkl`: A list of the indexed image paths.
- `metadata.pkl`: Index metadata.
- `exif.json`: EXIF data extracted from images, useful for filtering.
- `captions.tsv` (if `--captions` is used): A tab-separated file of image paths and their text descriptions.

For performance tuning options (e.g., using a GPU), see the *Optimizing the Command-Line Indexer* section in `OPTIMIZATION_GUIDE.md`.

### Step 2: Querying by Image
Once the index is built, you can find the top 10 most visually similar images to a query image using the following command.

**Usage:**
Replace `index_output` with your output directory and `/path/to/query.jpg` with the path to your query image.
```bash
python3 - <<'PY'
import sys, pickle, numpy as np, torch
from transformers import ViTImageProcessor, ViTModel
from PIL import Image
import faiss, os
# --- Configuration ---
outdir = "index_output"  # <-- Set your output directory here
query_img = sys.argv[1]   # <-- Query image path is passed as an argument
# --- Script ---
if not os.path.isdir(outdir) or not os.path.isfile(query_img):
    print(f"Error: Ensure output directory '{outdir}' exists and query image '{query_img}' is valid.")
    sys.exit(1)
paths = pickle.load(open(os.path.join(outdir,"image_paths.pkl"),"rb"))
index = faiss.read_index(os.path.join(outdir,"image_index.faiss"))
proc = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k"); model.eval()
im = Image.open(query_img).convert("RGB")
inp = proc(images=im, return_tensors="pt")
with torch.no_grad():
    feat = model(**inp).last_hidden_state[:,0].cpu().float().numpy()
feat /= (np.linalg.norm(feat,axis=1,keepdims=True)+1e-8)
D,I = index.search(feat.astype("float32"), 10)
print(f"Top 10 most similar images to '{query_img}':")
for r,(d,idx) in enumerate(zip(D[0],I[0]),1):
    if idx==-1: continue
    print(f"{r:02d}\t{paths[idx]}")
PY "/path/to/query.jpg"
```

### Step 3: Querying by Text
If you generated captions during indexing (Step 1), you can perform a text-based search using `grep`. This command searches the `captions.tsv` file for a specific phrase and returns the paths of the matching images.

**Usage:**
Replace `index_output` with your output directory and `"your phrase here"` with your search query.
```bash
grep -i "your phrase here" index_output/captions.tsv | cut -f1 | head -n 10
```

## Using the Web Application (for Distributed Processing)

This section describes the original, more advanced workflow that uses a web interface to manage a distributed processing backend. This is suitable for processing massive datasets across multiple machines.

### 1. Configure Your Case
Edit the configuration file (`app/config.yaml` or `app/config_optimized.yaml`) to specify your input directory, case name, and Redis connection details.

### 2. Start the Backend Infrastructure
This workflow requires a Redis server to be running.
```bash
# Start Redis (if not already running)
redis-server
```

### 3. Start the Web UI and Workers
You will need multiple terminals.

**Terminal 1: Start the Web Application**
```bash
# Ensure PYTHONPATH includes the project root
PYTHONPATH=. streamlit run app/main.py
```
This launches the UI at `http://localhost:8501`.

**Terminal 2 (and others): Start Background Workers**
For optimal performance, start multiple worker processes to process images in parallel.
```bash
PYTHONPATH=. python app/optimized_worker.py --config app/config_optimized.yaml
```

### 4. Enqueue Images for Processing
Use the `enqueue_jobs.py` script to scan a directory and add jobs to the Redis queue for the workers to process.
```bash
PYTHONPATH=. python app/enqueue_jobs.py --config app/config_optimized.yaml --input_dir /path/to/evidence
```

### 5. Monitor and Analyze in the UI
Use the web interface to:
- Monitor the progress of the indexing job.
- Load the completed case for analysis.
- Search, filter, cluster, and bookmark results.
- Generate reports.

## Performance Optimization Tips

- **Use the optimized configuration**: The `app/config_optimized.yaml` file contains settings optimized for high-performance processing.
- **Run multiple workers**: Start 4-8 worker processes for optimal performance on a typical workstation.
- **Adjust batch sizes**: For machines with limited memory, decrease `batch_size` in the config. For powerful machines, increase it.
- **Enable GPU acceleration**: Set `use_gpu: true` in the config file when running on a CUDA-capable GPU.
- **Tune Redis**: For very large datasets, consider adjusting Redis configuration for higher memory limits.
- **Disk I/O**: For best performance, place input images and output directory on fast SSDs.
