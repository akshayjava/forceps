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

## Running with Docker Compose (Recommended)
The easiest way to run the entire FORCEPS stack is with `docker-compose`.

### 1. Prerequisites
- **Install Docker and Docker Compose.**
- **NVIDIA GPU Users:** Ensure you have the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) installed.
- **Prepare your data:** Create host directories for your images (e.g., `./images`) and for the output (e.g., `./output_index`). The `docker-compose.yml` file maps these to the correct paths inside the containers.
- **(Optional) Convert models to ONNX:** For maximum performance, run `python app/convert_models.py --output_dir models/onnx`.

### 2. Launch the Application
From the root of the project directory, run:
```bash
docker-compose up --build
```
To run multiple workers for faster processing, use the `--scale` flag:
```bash
docker-compose up --build --scale worker=4
```

## Using the Application

### 1. Configure Your Case
Before starting, edit `app/config.yaml` (or `app/config.docker.yaml` if using Docker) to set your `case_name` and `input_dir`.

### 2. Start the Backend
If running manually, start the `worker` and `indexer` scripts in separate terminals. If using Docker, `docker-compose up` handles this for you.

### 3. Start the Indexing Job
Run the `enqueuer` script manually, or click the "Start Indexing Job" button in the UI. This will begin populating the Redis queue. You can monitor the progress in the UI's "Backend Monitoring" section.

### 4. Analyze the Results
Once the `indexer` script finishes, go to the UI (`http://localhost:8501`):
1.  **Load the Case:** In the sidebar, select your case from the dropdown and click "Load Selected Case".
2.  **Search:** Use the search bar to perform a hybrid keyword and semantic search.
3.  **Inspect:** Click "Show Details" on any image to see its full path, all computed hashes, and EXIF metadata.
4.  **Cluster:** Use the "Cluster Analysis" section to group results visually and find patterns.
5.  **Bookmark:** Use the "Manage Bookmark" expander to save items of interest, add tags, and write detailed notes.
6.  **Report:** Go to the "Reporting" tab to review all bookmarked items and generate a comprehensive PDF report for your case file.
