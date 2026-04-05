# Runpod Workflow

The repository includes a Runpod launcher script for remote GPU training and evaluation:

```bash
python3 scripts/runpod_train_eval.py \
  --dataset-name pilot_1000 \
  --backend transformers_qlora \
  --model-name nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16 \
  --epochs 1 \
  --max-examples 128 \
  --limit 5
```

Expected local setup:

- `RUNPOD_API_KEY` stored in `.env`
- a local SSH public key at `~/.ssh/id_ed25519.pub`
- `runpodctl` installed locally

The launcher will:

1. ensure the SSH key is registered with Runpod
2. create or reuse a pod
3. package the repo plus the selected dataset
4. upload the archive to the pod
5. install training dependencies remotely
6. build SIMH on the pod
7. run `ibm650_it.cli train-eval`
8. copy the output artifacts back locally
9. terminate the pod unless `--keep-pod` is set

The default GPU target is `NVIDIA RTX A6000` on community cloud. Adjust `--gpu-id` if inventory changes.

For the Nemotron backend, the launcher does not rely on `pip install mamba-ssm` from source. It installs CUDA / Torch / Python-matched binary wheels for `mamba-ssm` and `causal-conv1d` so the remote environment stays reproducible on Runpod's `torch 2.8 / CUDA 12.8 / Python 3.12` image.
