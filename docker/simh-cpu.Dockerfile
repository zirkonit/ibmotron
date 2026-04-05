FROM ubuntu:24.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    curl \
    git \
    make \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /repo
COPY . /repo

RUN python3 scripts/fetch_sources.py && ./scripts/build_simh.sh

CMD ["python3", "-m", "ibm650_it.cli", "smoke-examples", "--output", "/repo/artifacts/docker_smoke"]
