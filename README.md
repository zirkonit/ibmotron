# IBM 650 IT -> PIT Pipeline

This repository builds a stage-separated, reproducible pipeline for IBM 650 Internal Translator research with the learned target fixed to `IT -> PIT`.

The deterministic oracle remains:

1. IT translation to raw PIT punch output.
2. Historical reservation-card reconstruction for P1.
3. Patched SOAP II assembly to SPIT.
4. SIMH execution for functional validation.

The repository preserves raw punched artifacts, derives canonical training targets, and keeps SIMH execution isolated per job so dataset generation can run in parallel without clobbering fixed filenames used by the restored workflow.

## Quick start

```bash
python3.13 scripts/fetch_sources.py
./scripts/build_simh.sh
./scripts/smoke_examples.sh
make test
```

## CLI

```bash
ibm650-it translate --source third_party/simh/I650/sw/it/it_example_2_src.txt --output artifacts/example2_translate
ibm650-it split-reservations --pit-raw artifacts/example2_translate/pit_raw.dck --output artifacts/example2_translate
ibm650-it build-phase2-p1 --reservation-cards artifacts/example2_translate/reservation_cards.dck --translation-body artifacts/example2_translate/translation_body.dck --output artifacts/example2_phase2/pit_phase2_input_p1.dck
ibm650-it assemble --pit-phase2-input-p1 artifacts/example2_phase2/pit_phase2_input_p1.dck --output artifacts/example2_assemble
ibm650-it build-spit --soap-output artifacts/example2_assemble/soap_output.dck --output artifacts/example2_run/spit_p1.dck
ibm650-it run-spit --spit-p1 artifacts/example2_run/spit_p1.dck --output artifacts/example2_run
ibm650-it pipeline --source third_party/simh/I650/sw/it/it_example_2_src.txt --output artifacts/example2_pipeline
ibm650-it generate-accepted --band B1 --count 10 --output artifacts/datasets/b1_pilot
ibm650-it build-pilot-corpus --output artifacts/datasets/pilot_1000 --total-count 1000 --workers 8
ibm650-it build-stage-corpus --stage 2k --output artifacts/datasets/stage_2k --workers 8
ibm650-it prepare-sft --dataset-index artifacts/datasets/pilot_1000/splits/synthetic_train.jsonl --output artifacts/datasets/pilot_1000/sft/train.jsonl --limit 128 --band-repeat-preset b45_focus
ibm650-it train-model --sft-jsonl artifacts/datasets/pilot_1000/sft/train.jsonl --output artifacts/models/m4_smoke
ibm650-it run-inference --reference-index artifacts/datasets/pilot_1000/splits/synthetic_dev.jsonl --mode fine_tuned --model artifacts/models/m4_smoke --output artifacts/eval_reports/m4_fine
ibm650-it reevaluate-predictions --reference-index artifacts/datasets/pilot_1000/splits/synthetic_dev.jsonl --prediction-index artifacts/eval_reports/m4_fine/predictions.jsonl --output artifacts/eval_reports/m4_fine
ibm650-it review-b1-failures --reference-index artifacts/datasets/pilot_remote_128_20/splits/synthetic_dev.jsonl --prediction-index artifacts/eval_reports/sweeps/subset_128_20_a40_20260405_1137/e5_lr0p0002/predictions/fine_tuned/predictions.jsonl --output artifacts/eval_reports/reviews/b1_best_run
ibm650-it eval-report --reference-index artifacts/datasets/pilot_1000/splits/synthetic_dev.jsonl --prediction-index artifacts/eval_reports/m4_fine/predictions.jsonl
ibm650-it train-eval --dataset-root artifacts/datasets/pilot_remote_128_20 --output artifacts/eval_reports/baseline_real --limit 20
ibm650-it smoke-train-eval --dataset-root artifacts/datasets/pilot_1000 --output artifacts/eval_reports/m4_smoke --limit 5
ibm650-it overfit-sanity --dataset-index artifacts/datasets/pilot_remote_128_20/splits/synthetic_train.jsonl --output artifacts/eval_reports/overfit_real --example-count 32
python3 scripts/runpod_train_eval.py --dataset-name pilot_remote_128_20 --limit 20
python3 scripts/runpod_sweep.py --dataset-name pilot_remote_128_20 --gpu-id "NVIDIA A40" --cloud-type SECURE --reuse-single-pod
ibm650-it dashboard --host 127.0.0.1 --port 8765 --refresh-seconds 10
ibm650-it smoke-examples --output artifacts/smoke_examples
```

## Layout

The package follows the implementation layout from the spec:

- `ibm650_it/simh`: stage-separated SIMH wrapper and deck handling.
- `ibm650_it/source`: AST, header/bounds analysis, and canonical IT rendering.
- `ibm650_it/pit`: PIT normalization and comparison helpers.
- `ibm650_it/generate`: synthetic program generation bands.
- `ibm650_it/dataset`: record creation, dedupe, and splits.
- `ibm650_it/training`: SFT data preparation and training stubs.
- `ibm650_it/eval`: exact match, assemblability, and functional evaluation.

## Current scope

The implemented baseline covers:

- M0 source locking and SIMH build support.
- M1 stage-separated reference pipeline with preserved artifacts.
- M2 B0-B5-capable AST, bounds, renderer, and synthetic sample generator.
- M3 pilot corpus builder with provenance, dedupe, and alpha-hash splits.
- M4 SFT preparation with higher-band repeat presets, smoke/QLoRA training backends, and zero-shot/few-shot/fine-tuned evaluation runs.
- Runpod orchestration for remote bootstrap, training, artifact sync, and pod teardown.

Full language coverage, larger runtime packages, and direct machine-code targets remain out of scope for v1.

`B5` input-dependent samples use a simulator-backed helper program to punch valid IT `READ` cards, so the corpus can exercise `READ` without hand-maintaining a parallel raw card codec.

## Smoke Training Note

The current M4 training path is a deterministic smoke backend that stores retrieval examples and exercises the training and evaluation interfaces end-to-end. It is intentionally lightweight and reproducible; it is not the final GPU Unsloth fine-tuning path.

## Prompt Contract

SFT examples now use a strict wrapped completion format:

```text
System:
Compile the following IBM 650 IT program to canonical PIT deck output.
Return only a <PIT>...</PIT> block containing the PIT deck, one card per line, with no explanation.

User:
<IT>
... canonical it_text_v1 ...
</IT>

Assistant:
<PIT>
... pit_raw_canonical ...
</PIT>
```

The inference path strips the wrapper before assemblability and functional checks, and generation stops on `</PIT>` when the tokenizer exposes that token sequence.

## Local Reevaluation

GPU jobs now generate PIT candidates remotely and defer SOAP/SIMH validation to the local CPU environment. The `reevaluate-predictions` command recomputes assembly and functional metrics from a saved `predictions.jsonl`, rewrites prediction records with full diagnostics, regenerates the report, and refreshes the failure archive.

## Dashboard

Use the local dashboard to monitor active Runpod wrappers, active pods, and recent finished runs from a browser:

```bash
ibm650-it dashboard --host 127.0.0.1 --port 8765 --refresh-seconds 10
```

Then open:

```text
http://127.0.0.1:8765
```

The page shows:

- active local launcher processes
- active or orphan Runpod pods
- explicit job phases: `remote_bootstrap`, `remote_train`, `remote_generate`, `local_reevaluate`, `complete`, `failed`
- remote generation progress for `zero_shot`, `few_shot`, and `fine_tuned`
- GPU utilization reported by `nvidia-smi`
- recent local run summaries and top-line metrics

## Runpod Sweep

Use the sweep runner to execute the planned same-slice `128/20` LR continuation grid around the frozen `5 epochs @ 2e-4` baseline:

```bash
python3 scripts/runpod_sweep.py \
  --dataset-name pilot_remote_128_20 \
  --gpu-id "NVIDIA A40" \
  --cloud-type SECURE \
  --reuse-single-pod \
  --epochs 5 \
  --learning-rates 3e-4 4e-4
```

The script writes a sweep manifest at `artifacts/eval_reports/sweeps/<name>/manifest.json`, compares each candidate run against the current `2e-4` baseline, records Gate A status, and chooses a winner using:

1. higher functional equivalence
2. then higher exact match
3. then higher per-card exact
4. with `assemblability >= 95%`

Each run keeps its own full local output directory under that sweep root.

## B1 Failure Review

Use the B1 reviewer on the winning held-out run before scaling the corpus:

```bash
ibm650-it review-b1-failures \
  --reference-index artifacts/datasets/pilot_remote_128_20/splits/synthetic_dev.jsonl \
  --prediction-index artifacts/eval_reports/sweeps/subset_128_20_a40_20260405_1137/e5_lr0p0002/predictions/fine_tuned/predictions.jsonl \
  --output artifacts/eval_reports/reviews/b1_best_run
```

The review writes:

- `cases.jsonl` with one classification per non-exact `B1` example
- `summary.json` with category counts
- `review.md` with a concise operator-facing summary
- `selected_failures/` with 10 representative archived cases

## Overfit Sanity Check

Before running larger GPU jobs, use the overfit sanity command on a tiny slice of the training set. If the backend cannot memorize `16-32` examples with strong exact and assemblability scores on that same slice, scaling up the dataset is premature.

## GPU Training Note

The repository now also includes a real `transformers_qlora` backend plus a Runpod launcher script. That path is intended for actual adapter training on rented GPUs, while the smoke backend remains the fast local verification path.

For Nemotron on Runpod, the launcher installs CUDA/Torch-matched `mamba-ssm` and `causal-conv1d` binary wheels explicitly instead of relying on source builds.
