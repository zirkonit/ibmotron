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
ibm650-it prepare-sft --dataset-index artifacts/datasets/pilot_1000/splits/synthetic_train.jsonl --output artifacts/datasets/pilot_1000/sft/train.jsonl --limit 128
ibm650-it train-model --sft-jsonl artifacts/datasets/pilot_1000/sft/train.jsonl --output artifacts/models/m4_smoke
ibm650-it run-inference --reference-index artifacts/datasets/pilot_1000/splits/synthetic_dev.jsonl --mode fine_tuned --model artifacts/models/m4_smoke --output artifacts/eval_reports/m4_fine
ibm650-it eval-report --reference-index artifacts/datasets/pilot_1000/splits/synthetic_dev.jsonl --prediction-index artifacts/eval_reports/m4_fine/predictions.jsonl
ibm650-it train-eval --dataset-root artifacts/datasets/pilot_1000 --output artifacts/eval_reports/full_smoke --backend smoke --limit 5
ibm650-it smoke-train-eval --dataset-root artifacts/datasets/pilot_1000 --output artifacts/eval_reports/m4_smoke --limit 5
ibm650-it overfit-sanity --dataset-index artifacts/datasets/pilot_1000/splits/synthetic_train.jsonl --output artifacts/eval_reports/overfit_smoke --example-count 16 --backend smoke
python3 scripts/runpod_train_eval.py --dataset-name pilot_remote_quick --backend transformers_qlora --max-examples 4 --limit 1
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
- M2 B0/B1-capable AST, bounds, renderer, and sample generator.
- M3 pilot corpus builder with provenance, dedupe, and alpha-hash splits.
- M4 smoke SFT preparation, smoke training backend, and zero-shot/few-shot/fine-tuned evaluation runs.
- Runpod orchestration for remote bootstrap, training, artifact sync, and pod teardown.

Full language coverage, larger runtime packages, and direct machine-code targets remain out of scope for v1.

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

## Overfit Sanity Check

Before running larger GPU jobs, use the overfit sanity command on a tiny slice of the training set. If the backend cannot memorize `16-32` examples with strong exact and assemblability scores on that same slice, scaling up the dataset is premature.

## GPU Training Note

The repository now also includes a real `transformers_qlora` backend plus a Runpod launcher script. That path is intended for actual adapter training on rented GPUs, while the smoke backend remains the fast local verification path.

For Nemotron on Runpod, the launcher installs CUDA/Torch-matched `mamba-ssm` and `causal-conv1d` binary wheels explicitly instead of relying on source builds.
