FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04

WORKDIR /workspace
RUN apt-get update && apt-get install -y python3 python3-venv python3-pip git libgl1-mesa-glx libglib2.0-0 ffmpeg libmagic1 && rm -rf /var/lib/apt/lists/*

COPY . /workspace
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --upgrade pip
RUN pip install -r app/requirements.txt

LABEL maintainer="your.email@example.com"
LABEL description="FORCEPS: Forensic Optimized Retrieval & Clustering of Evidence via Perceptual Search"

EXPOSE 8501
CMD ["streamlit", "run", "app/main.py", "--server.address=0.0.0.0", "--server.port=8501"]
