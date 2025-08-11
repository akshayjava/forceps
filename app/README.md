# FORCEPS — Forensic Optimized Retrieval & Clustering of Evidence via Perceptual Search

## Overview
FORCEPS is a forensic image similarity and natural-language search tool designed for massive-scale, high-speed, and private on-site investigations. It is built on a scalable, distributed architecture to handle terabyte-sized datasets.

Key features:
- **Distributed Architecture:** Uses a Redis queue to distribute work to multiple parallel workers for extreme scalability.
- **Optimized Models:** Supports GPU-accelerated ONNX models for high-speed embedding computation, with a clear path to TensorRT.
- **Two-phase indexing:** (Phase 1: embeddings; Phase 2: NL captions)
- **Multi-model embeddings:** (ViT + CLIP)
- **FAISS Vector DB:** Builds a highly optimized FAISS index for fast similarity search.
- **Optional Ollama Integration:** For local natural language captioning of images.
- **All processing is local:** No data leaves your environment.

## Architecture
The system is composed of several key components that work together:

1.  **Redis:** A message broker that manages the queue of images to be processed and the queue of results.
2.  **Enqueuer (`enqueue_jobs.py`):** A script that scans the image directory and populates the Redis job queue.
3.  **Worker (`engine.py`):** The core processing engine. You can run many workers in parallel. Each worker pulls a job from the queue, computes the embeddings for the images, and pushes the results to the results queue.
4.  **Index Builder (`build_index.py`):** A script that consumes the results from the workers, aggregates all embeddings, and builds the final, compressed FAISS index.
5.  **User Interface (`main.py`):** A Streamlit application that can be used to start the enqueueing process and to search the final, completed index.

## Quickstart (Distributed Indexing)

This guide explains how to run the full distributed pipeline.

### 1. Prerequisites
- **Install Python dependencies:**
  ```bash
  ./install.sh
  source ./venv_forceps/bin/activate
  ```
- **Install and run Redis:**
  - On macOS: `brew install redis && brew services start redis`
  - On Linux: `sudo apt-get install redis-server && sudo systemctl start redis-server`
- **(Optional) Convert models to ONNX:** For maximum performance, convert the PyTorch models to ONNX.
  ```bash
  python app/convert_models.py --output_dir models/onnx
  ```

### 2. Run the Backend Services
Open three separate terminal windows. In each one, activate the virtual environment (`source venv_forceps/bin/activate`).

- **Terminal 1: Start the Worker(s):**
  The worker processes the images. You can run multiple instances of this script on different machines for parallel processing.
  ```bash
  python app/engine.py --model_dir models/onnx
  ```

- **Terminal 2: Start the Index Builder:**
  The index builder waits for results from the workers and builds the final index file.
  ```bash
  python app/build_index.py --output_dir output_index
  ```

### 3. Start the Indexing Job
- **Terminal 3: Enqueue the Jobs:**
  This script scans your image folder and pushes jobs to the Redis queue for the workers to pick up.
  ```bash
  python app/enqueue_jobs.py --input_dir /path/to/your/images
  ```

The workers and index builder will now process the entire dataset. You can monitor the progress in their terminal windows.

### 4. Use the User Interface
Once the `build_index.py` script reports that it is complete, you can launch the Streamlit UI to search your index.

- **Launch the UI:**
  ```bash
  streamlit run app/main.py
  ```
- In the UI's sidebar, set the "Index Directory" to the folder you specified in the `build_index.py` command (e.g., `output_index`).
- Click "Load Index from Directory".
- You can now use the search bar to find similar images.
- You can also use the UI to kick off a new indexing job by specifying a folder and clicking "Start Two-Phase Indexing". This simply runs the `enqueue_jobs.py` script for you.
