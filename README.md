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

## Running FORCEPS from the Command Line

### 1. Configure Your Case
Edit the configuration file to specify your input directory and other settings:

```bash
# Standard configuration
vim app/config.yaml

# Or use the optimized configuration for better performance
vim app/config_optimized.yaml
```

Key settings to update:
- `data.input_dir`: Path to your forensic images directory
- `data.output_dir`: Where to store the output (default: `./output_index`)
- `case_details.case_name`: Name for your investigation case

### 2. Run CLI Indexing (no UI)

Quickly index a directory from the command line using the ViT indexer:

```bash
cd /path/to/foreceps
python3 -m pip install --user --upgrade torch torchvision transformers faiss-cpu pillow tqdm opencv-python
PYTHONPATH=. python3 run_cli.py \
  --image_dir /path/to/images \
  --output_dir index_out_opt \
  --device auto --batch_size 16
```

By default, FP16/compile are disabled on CPU/MPS for stability and enabled (FP16 only) on CUDA. Use `--fp16/--no-fp16` and `--compile/--no-compile` to override.

### 3. Start the Web Application (UI)

The Streamlit web interface provides control over the entire system:

```bash
# Ensure PYTHONPATH includes the project root
PYTHONPATH=/path/to/foreceps streamlit run app/main.py --server.port 8501 --server.address localhost
```

This will launch the web application at http://localhost:8501

### 4. Start Background Processing Workers

For optimal performance, start multiple worker processes to process images in parallel:

```bash
# Start multiple optimized workers (in separate terminals)
PYTHONPATH=/path/to/foreceps python app/optimized_worker.py --config app/config_optimized.yaml
PYTHONPATH=/path/to/foreceps python app/optimized_worker.py --config app/config_optimized.yaml
PYTHONPATH=/path/to/foreceps python app/optimized_worker.py --config app/config_optimized.yaml

# Optionally run the auto-scaling worker manager
PYTHONPATH=/path/to/foreceps python app/auto_scale_workers.py --config app/config_optimized.yaml
```

### 5. Process a Directory of Images

Queue a directory of images for processing with the enqueuer script:

```bash
# Process a specific directory
PYTHONPATH=/path/to/foreceps python app/enqueue_jobs.py --config app/config_optimized.yaml --input_dir /path/to/evidence --job_batch_size 256

# Use the directory specified in the config file
PYTHONPATH=/path/to/foreceps python app/enqueue_jobs.py --config app/config_optimized.yaml
```

### 6. Monitor Processing Progress

Check the current processing status:

```bash
# View queue statistics
redis-cli mget "forceps:stats:total_images" "forceps:stats:embeddings_done" "forceps:stats:captions_done"

# View remaining jobs in queue
redis-cli llen forceps:job_queue
```

### 7. Test Search Functionality

Verify the system is working correctly with the test script:

```bash
python test_search.py
```

### 8. Running the Complete Pipeline

Here's a complete sequence to run FORCEPS from scratch:

```bash
# Terminal 1: Start Redis (if not already running)
redis-server

# Terminal 2: Start the web interface
PYTHONPATH=/path/to/foreceps streamlit run app/main.py

# Terminal 3: Start worker processes
PYTHONPATH=/path/to/foreceps python app/optimized_worker.py --config app/config_optimized.yaml

# Terminal 4: Enqueue images for processing
PYTHONPATH=/path/to/foreceps python app/enqueue_jobs.py --config app/config_optimized.yaml --input_dir /path/to/evidence
```

## Using the Application

### 1. Configure Your Case
Before starting, edit `app/config.yaml` (or `app/config.docker.yaml` if using Docker) to set your `case_name` and `input_dir`.

### 2. Start the Backend
If running manually, start the `worker` and `indexer` scripts in separate terminals as shown in the command line instructions above. If using Docker, `docker-compose up` handles this for you.

### 3. Start the Indexing Job
Run the `enqueuer` script manually as shown above, or click the "Start Indexing Job" button in the UI. This will begin populating the Redis queue. You can monitor the progress in the UI's "Backend Monitoring" section.

### 4. Analyze the Results
Once processing is underway, you can immediately start using the UI (`http://localhost:8501`):
1.  **Load the Case:** In the sidebar, select your case from the dropdown and click "Load Selected Case".
2.  **Search:** Use the search bar to perform a hybrid keyword and semantic search.
3.  **Inspect:** Click "Show Details" on any image to see its full path, all computed hashes, and EXIF metadata.
4.  **Cluster:** Use the "Cluster Analysis" section to group results visually and find patterns.
5.  **Bookmark:** Use the "Manage Bookmark" expander to save items of interest, add tags, and write detailed notes.
6.  **Report:** Go to the "Reporting" tab to review all bookmarked items and generate a comprehensive PDF report for your case file.

## Performance Optimization Tips

- **Use the optimized configuration**: The `app/config_optimized.yaml` file contains settings optimized for high-performance processing.
- **Run multiple workers**: Start 4-8 worker processes for optimal performance on a typical workstation.
- **Adjust batch sizes**: For machines with limited memory, decrease `batch_size` in the config. For powerful machines, increase it.
- **Enable GPU acceleration**: Set `use_gpu: true` in the config file when running on a CUDA-capable GPU.
- **Tune Redis**: For very large datasets, consider adjusting Redis configuration for higher memory limits.
- **Disk I/O**: For best performance, place input images and output directory on fast SSDs.
