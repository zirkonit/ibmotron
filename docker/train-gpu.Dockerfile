FROM pytorch/pytorch:2.8.0-cuda12.9-cudnn9-runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    git \
    openssh-client \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /repo
COPY . /repo

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -e ".[train]"

# Nemotron-specific binary wheels (`mamba-ssm` and `causal-conv1d`) are
# installed by the Runpod workflow so they can be matched to the pod's
# exact Python / Torch / CUDA stack.

CMD ["python", "-m", "ibm650_it.cli", "smoke-train-eval", "--dataset-root", "/repo/artifacts/datasets/pilot_1000", "--output", "/repo/artifacts/eval_reports/m4_smoke", "--limit", "5"]
