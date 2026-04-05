FROM nvidia/cuda:12.6.2-cudnn-runtime-ubuntu24.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    git \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /repo
COPY . /repo

CMD ["python3", "-m", "ibm650_it.cli", "prepare-sft", "--dataset-index", "/repo/artifacts/datasets/index.jsonl", "--output", "/repo/artifacts/datasets/train_sft.jsonl"]
