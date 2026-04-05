# Implementation Spec: Reproducible LLM-Based IBM 650 IT Translation Pipeline

## 1. Objective

Build a reproducible research pipeline in which the learned component performs the IBM 650 Internal Translator’s **translation phase**, not the full historical toolchain. Historically, IT compilation proceeds in two phases: IT source is translated into **PIT**, a symbolic program in standard alphanumeric SOAP format, and SOAP then assembles PIT into the machine-coded **SPIT** program. PIT is not just a short mnemonic listing. It includes the main symbolic program, a statement dictionary, constants, and ten reservation cards. The v1 learned target MUST therefore be `IT -> PIT`, with SOAP II remaining the deterministic assembler and address-placement engine. ([Internet Archive][1])

The implementation MUST optimize for historical correctness, reproducibility, and debuggability. It MUST preserve raw reference artifacts from the restored SIMH workflow, but it MAY also derive normalized training targets and evaluation views from those raw artifacts. The model is being tested on learned symbolic translation, not on drum-timing arithmetic or machine-address selection. SOAP continues to own those responsibilities in v1. ([Internet Archive][1])

## 2. Non-goals

v1 MUST NOT attempt any of the following as baseline deliverables:

* Direct `IT -> machine code` generation.
* Learned drum optimization or learned machine address assignment.
* Full-language IT coverage, including every extension, runtime package, and segmentation feature.
* A handwritten reimplementation of the historical IT compiler.
* A user-facing demo before the deterministic reference pipeline, dataset pipeline, and evaluation harness are stable.

Those are stretch goals only.

## 3. Locked external sources and required fetched artifacts

The coding agent MUST fetch and lock the following external assets in `sources.lock.json`, including source identifier, exact commit hash or archive checksum, fetch date, and local path. The restored SIMH IBM 650 package already contains the critical decks, scripts, patches, runtime packages, and sample programs needed to build a reference oracle. ([Open SIMH][2])

Required source families:

1. `open-simh/simh`

   * `I650/sw/run_it.ini`
   * `I650/sw/run_soap.ini`
   * `I650/sw/it/it_compiler.dck`
   * `I650/sw/it/soapII.dck`
   * `I650/sw/it/soapII_patch.dck`
   * `I650/sw/it/it_reservation_p1.dck`
   * `I650/sw/it/it_package_p1.dck`
   * `I650/sw/it/it_example_1_src.txt`
   * `I650/sw/it/it_example_1_data.txt`
   * `I650/sw/it/it_example_2_src.txt`

2. Primary historical documentation

   * `CarnegieInternalTranslator.pdf`
   * IBM 650 operation manual / programming manual material sufficient to confirm instruction format and special addresses

3. Training stack

   * `nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16`
   * current Unsloth Nemotron 3 documentation
   * current Unsloth requirements page

The repository SHOULD NOT vendor large PDF manuals. Store fetch metadata, checksums, citations, and any locally produced parsed notes instead.

## 4. Historical and machine constraints that the implementation MUST honor

IBM 650 instructions are ten-digit decimal words. Digits 10-9 are the opcode, digits 8-5 are the data address, and digits 4-1 are the next-instruction address. Addressable locations include general storage `0000-1999` and the special locations `8000` through `8003` for the console storage-entry switches, distributor, lower accumulator, and upper accumulator. The simulator documentation and shipped scripts default to a `2k` machine, and the restored IT/SOAP workflows assume that configuration. ([Computer History Archive][3])

In IT itself, statement numbers are labels rather than execution order. Historical execution follows the physical order of statements unless altered by transfers. Statement numbers are non-negative and historically bounded by 626. `I` variables are fixed-point index variables. `Y` and `C` variables are floating-point problem variables. Composite variables are allowed, and their parenthesized or indexed quantities must be fixed-point expressions. Output statements can punch one to four variables. Iteration statements are historically nestable up to depth four. Statement cards are limited to 14 characters excluding the statement number, and every statement must terminate with `F`, with the final statement terminated by `FF`. ([Internet Archive][1])

The header card exists because IT needs ahead-of-time storage sizing. Historically it must provide the maximal subscript numbers for the variable classes, the maximal statement number, and the total storage required by extensions or runtime packages. In the SIMH line-oriented examples, this appears in simplified form as a leading control line such as `+ 2 0 50 10 1672`; the prime example explains that `1672 = 1999 - (2 + 0 + 50 + 10 + 265)` when using P1. The implementation MUST compute this field correctly and MUST treat it as a semantic part of source generation. ([Internet Archive][1])

The restored SIMH environment uses runtime package **P1** by default. The restoration notes state that P1 provides floating-point `+`, `-`, `/`, `*`, plus `PUNCH` and `READ`, and explicitly notes that exponentiation is not available there, so using the power operator will crash the object program at runtime. Those notes also document the SOAP I to SOAP II mnemonic substitutions used in the restoration and the requirement to apply an IT-specific `soapII_patch` because SOAP I itself is unavailable. v1 MUST therefore exclude exponentiation and MUST always assemble IT-produced output through patched SOAP II. ([GitHub][4])

## 5. Chosen implementation boundary and target artifacts

The implementation MUST distinguish these artifacts:

1. `it_text_v1`

   * Canonical, human-readable, line-oriented IT source used for generation, training input, and model inference.

2. `pit_raw`

   * The raw deck punched by the IT translation phase, preserved exactly as emitted.

3. `pit_raw_canonical`

   * A normalized one-card-per-line view of `pit_raw`, used as the **primary v1 model target**.

4. `pit_phase2_input_p1`

   * A derived deck formed by moving the translation phase’s last ten reservation cards to the front, then inserting `it_reservation_p1.dck`, then appending the remainder of the translation output. This is the historical phase-two SOAP input for P1, reconstructed deterministically. ([Internet Archive][1])

5. `spit_p1`

   * The assembled, runnable deck formed by prepending `it_package_p1.dck` to the patched SOAP assembly output. This is **not** the primary training target in v1. ([GitHub][5])

6. `run_output`

   * The punched output deck produced by executing `spit_p1` in SIMH, optionally with an attached data deck.

The primary supervised target for v1 MUST be `pit_raw_canonical`, not `spit_p1` and not direct machine code.

## 6. Repository layout

```text
repo/
  README.md
  Makefile
  pyproject.toml
  sources.lock.json

  third_party/
    simh/                        # pinned checkout or vendored snapshot

  docker/
    simh-cpu.Dockerfile
    train-gpu.Dockerfile

  docs/
    source_manifest.md
    lexical_notes.md
    dataset_schema.md
    eval_protocol.md

  scripts/
    fetch_sources.py
    build_simh.sh
    smoke_examples.sh

  ibm650_it/
    __init__.py

    simh/
      runner.py                  # subprocess wrapper
      ini_templates/
        translate_only.ini.j2
        assemble_pit.ini.j2
        run_spit.ini.j2
      workdir.py                 # per-job isolation
      deckio.py                  # split/join/canonicalize decks

    source/
      ast.py
      bounds.py                  # interval analysis for header sizing
      render_it_text.py
      render_it_card80.py        # optional archival renderer
      normalize_it.py

    pit/
      normalize_pit.py
      diff.py
      parsers.py                 # minimal parser only if needed

    generate/
      templates.py
      bands.py
      sample_program.py
      sample_data.py             # v1.1+
      shrink.py

    dataset/
      schema.py
      build_records.py
      dedupe.py
      split.py

    training/
      prepare_sft.py
      prompt_templates.py
      train_unsloth.py
      infer.py

    eval/
      exact_match.py
      assemble_check.py
      functional.py
      failure_taxonomy.py
      report.py

  tests/
    golden/
      it_example_1/
      it_example_2/
    unit/
    integration/

  artifacts/
    logs/
    datasets/
    models/
    eval_reports/
    failures/
```

## 7. Environment and execution model

Use Linux first. Data generation, assembly, and execution validation SHOULD run CPU-only. Training SHOULD run in a separate GPU environment. Keep the two environments isolated so that SIMH reproducibility does not depend on CUDA or model-serving packages.

Every SIMH job MUST execute in an isolated temporary working directory. The shipped `.ini` scripts write fixed filenames such as `deck_out.dck`, `deck_pit.dck`, `deck_spit.dck`, `deck_soap.dck`, `deck_res.dck`, and `deck_out_run.dck`. `run_it.ini` also deletes intermediates at the end. Running those flows in a shared directory will create collisions and silent corruption under parallel dataset generation. Either templatize the `.ini` files with unique filenames or copy the necessary decks into a unique per-job work directory and run there. ([GitHub][5])

The implementation MUST preserve every intermediate artifact needed for debugging: console log, printed listing, raw punched output, canonicalized deck views, and assembly/run logs.

## 8. Deterministic reference pipeline

Do **not** call the shipped `run_it.ini` unchanged as the only workflow. It bundles translation, reservation-card rearrangement, assembly, package joining, execution, and cleanup. Instead, implement the same logic as separate preserved stages. `run_it.ini` is the reference behavior, not the final pipeline interface. ([GitHub][5])

Required deterministic stages:

1. **Translation-only**

   * Load `it_compiler.dck`.
   * Attach source deck.
   * Run the IT compiler.
   * Capture the raw punched output deck.
   * Treat non-zero upper accumulator as compilation failure and preserve the failure code and logs. ([GitHub][5])

2. **Reservation split**

   * Split the final ten cards of the translation output into `reservation_cards`.
   * Preserve the remainder as `translation_body`.
   * This split is historically significant. The manual identifies the last ten cards as reservation cards, and the shipped SIMH script reconstructs phase-two input by moving them. ([Internet Archive][1])

3. **PIT phase-two input build for P1**

   * Join `reservation_cards`
   * then `it_reservation_p1.dck`
   * then `translation_body`
   * write `pit_phase2_input_p1.dck` ([Internet Archive][1])

4. **Assembly**

   * Load `soapII.dck`
   * apply `soapII_patch.dck`
   * assemble `pit_phase2_input_p1.dck`
   * preserve raw SOAP punch output and log ([GitHub][5])

5. **SPIT build**

   * Join `it_package_p1.dck`
   * then assembly output
   * write `spit_p1.dck` ([GitHub][5])

6. **Execution**

   * Load `spit_p1.dck`
   * set program address to `1999`
   * attach optional input deck
   * attach output punch deck
   * run with a wall-clock timeout and a simulator-step budget
   * preserve punched output deck and console log ([GitHub][5])

The implementation MUST expose these as callable Python APIs and CLI entry points. Each stage MUST be testable in isolation.

## 9. Canonical text representations

### 9.1 `it_text_v1`

`it_text_v1` is the canonical input representation.

Rules:

* First line: simplified header in the form `+ nI nY nC nS N`
* One statement per line
* Statement numbers MUST be zero-padded to four digits
* Use lower-case source tokens
* Use exactly one ASCII space between tokens
* Canonical source MUST omit human comments
* Canonical source MUST end with a final statement whose terminator is `ff`
* Canonical source MUST contain no post-`ff` commentary, even though the SIMH prime example notes that comments after the final `ff` are safely ignored by the IBM 650 compiler. ([GitHub][6])

`it_text_v1` is **not** the archival 80-column card format. It is the human-readable SIMH-style source form. A separate optional renderer, `it_card80_raw`, SHOULD render true historical statement cards, including the blank card after the header, for archival validation only. The archival renderer MUST enforce the historical per-statement length limits and `F`/`FF` placement rules. ([Internet Archive][1])

### 9.2 `pit_raw` and `pit_raw_canonical`

`pit_raw` MUST preserve exactly what the translation phase punched. Do not delete cards, reorder cards, or inject package decks into this raw artifact.

`pit_raw_canonical` is the primary training target and MUST:

* preserve one punched card per line
* preserve card order
* preserve the statement dictionary, constants, and reservation cards
* strip trailing whitespace
* normalize line endings to LF
* optionally collapse repeated internal spaces only if the normalizer is proven idempotent on the golden examples
* preserve SOAP comment text if present
* drop leading/trailing blank load cards **only** in the canonical derivative, never in the raw artifact, because the historical operating instructions explicitly identify edge blank load cards as discardable output artifacts. ([Internet Archive][1])

### 9.3 `pit_phase2_input_p1`

This is a deterministic derived artifact, not the primary translation target. It exists so patched SOAP II can assemble the output. Build it exactly as described in Section 8 and never ask the model to hallucinate `it_reservation_p1.dck`. ([Internet Archive][1])

## 10. Program AST, bound analysis, and header computation

The source generator MUST work from an AST, not from string templates alone.

Minimum AST:

```text
Program
  header
  statements: [Statement]

Statement
  Assign(target, expr)
  Goto(target_stmt)
  IfGoto(target_stmt, lhs, relation, rhs)
  Punch(vars)
  Halt()
  Iterate(end_stmt, loop_var, start_expr, step_expr, stop_expr)
  Read(vars)                    # v1.1+

Expr
  IntConst
  FloatConst
  Var
  Neg(expr)
  Add(lhs, rhs)
  Sub(lhs, rhs)
  Mul(lhs, rhs)
  Div(lhs, rhs)

Var
  I(index_expr)
  Y(index_expr)
  C(index_expr)
```

The generator MUST include an interval or affine-bound analysis for fixed-point expressions so it can compute `nI`, `nY`, `nC`, and the final free-storage field `N`. The key rule is that header sizing uses **maximal reachable subscript values**, not just the largest literal subscript text seen in source. The SIMH prime example shows this explicitly: `ci1` contributes to `nC = 50` because `i1` ranges over `1..50`, producing header `+ 2 0 50 10 1672`. ([Internet Archive][1])

Header computation MUST follow this algorithm for v1:

```text
nI   = maximal reachable fixed-point subscript for class I
nY   = maximal reachable subscript for class Y
nC   = maximal reachable subscript for class C
nS   = maximal statement number used in the source
nPkg = runtime package footprint (P1 = 265 for v1)
N    = 1999 - (nI + nY + nC + nS + nPkg)
assert N > 0
```

The generator MUST reject any program whose bounds cannot be proven finite and positive under this simple analysis. In v1, index expressions SHOULD therefore be restricted to affine fixed-point forms with statically known bounds.

## 11. Supported language subset

The restored documentation and examples confirm the broader historical language, but the shipped implementation SHOULD stage support. The following subset MUST ship first. ([Internet Archive][1])

### v1.0 required subset

* Assignment / substitution statements
* Unconditional transfer
* Conditional transfer
* Halt
* `T` output statements
* Iteration statements with nesting depth capped at **1** in the generator
* Variables from classes `I`, `Y`, `C`
* Simple bounded composite or indexed references once the bound analyzer is working
* Integer constants and floating-point constants
* Unary minus
* `+`, `-`, `*`, `/`
* Parenthesized expressions only where required by the renderer
* One explicit halting path
* One explicit observable output path

### v1.0 exclusions

* Exponentiation
* Extension statements
* Segmentation
* Matrix variables
* Zero-numbered statements
* Continuation-card statements
* Loop nesting depth > 1
* Programs without guaranteed halting behavior
* Programs whose semantics cannot be observed through punched output

### v1.1

Add `READ` support only after the card-data codec has been validated against the shipped sample material and golden tests. Until then, generate self-contained programs that compute from constants and internal state and expose their final values via `T`.

### v2

Add:

* P2/P3/P4 runtime packages
* exponentiation
* zero-numbered statements
* continuation cards
* extension statements
* deeper loop nesting
* matrix variables

## 12. Generator strategy and corpus scaling

Do not begin with “50,000 pairs” as a blind fixed target. Begin with a validated pilot corpus and expand only while structural novelty remains healthy.

Required curriculum bands:

1. **B0: straight-line scalar programs**

   * 1 to 3 assignments
   * no composite indexing
   * one final `T`
   * one final `H`

2. **B1: longer straight-line programs**

   * 4 to 8 statements
   * mixed `I`, `Y`, `C`
   * arithmetic chains and intermediate reuse

3. **B2: transfer-heavy programs**

   * unconditional and conditional transfers
   * still guaranteed halting
   * no iterations yet

4. **B3: iteration programs**

   * single iteration statement
   * bounded loop variable
   * optional bounded composite indexing

5. **B4: mixed programs**

   * 8 to 20 statements
   * transfers plus one iteration
   * multiple outputs

6. **B5: input-dependent programs**

   * enabled only after `READ` codec validation

Admission rules for any generated sample:

* header sizing succeeds
* source renderer succeeds
* reference translation succeeds
* patched SOAP assembly succeeds
* execution succeeds within time and step budgets if the sample is in the functional-evaluation set
* no duplicate `alpha_hash`
* no unsupported lexical construct appears
* observability requirement satisfied

Scaling plan:

* smoke: 100 accepted samples total
* pilot: 1,000 accepted samples
* first full corpus: 10,000 accepted unique samples
* second full corpus: 20,000 to 25,000 accepted unique samples
* stretch to 50,000 only if trailing 1,000-sample `alpha_hash` novelty stays above 5 percent and trailing `shape_hash` novelty stays above 1 percent

The implementation MUST record both accepted and rejected-generation statistics so the generator can be tuned intentionally rather than by guesswork.

## 13. Dataset schema and artifact retention

Use file-backed artifacts plus a JSONL index. Large deck text SHOULD live in files, with JSONL storing relative paths and hashes. Small pilot corpora MAY inline text for convenience.

Minimum record schema:

```json
{
  "id": "uuid-or-stable-hash",
  "band": "B3",
  "seed": 123456,
  "runtime_package": "P1",
  "source_format": "it_text_v1",

  "source": {
    "it_text_v1": "path/or-inline",
    "it_card80_raw": "optional path",
    "header": {
      "n_i": 2,
      "n_y": 0,
      "n_c": 50,
      "n_s": 10,
      "n_pkg": 265,
      "N": 1672
    }
  },

  "hashes": {
    "surface_hash": "sha256",
    "alpha_hash": "sha256",
    "shape_hash": "sha256",
    "pit_hash": "sha256"
  },

  "reference": {
    "translate": {
      "status": "ok",
      "upper_acc": 0,
      "pit_raw": "path",
      "pit_raw_canonical": "path",
      "reservation_cards": "path",
      "translation_body": "path",
      "console_log": "path"
    },
    "assemble": {
      "status": "ok",
      "pit_phase2_input_p1": "path",
      "soap_output": "path",
      "console_log": "path"
    },
    "run": {
      "status": "ok",
      "input_deck": "optional path",
      "spit_p1": "path",
      "output_deck": "path",
      "console_log": "path"
    }
  },

  "generator": {
    "ast_json": "path",
    "bounds_json": "path",
    "features": ["if_goto", "iterate", "indexed_c"]
  },

  "provenance": {
    "simh_source": "locked source id",
    "simh_commit_or_checksum": "value",
    "generator_version": "git sha",
    "normalizer_version": "git sha"
  }
}
```

Every record included in supervised training MUST have a successful translation. Every record included in functional evaluation MUST additionally have successful assembly and successful execution.

## 14. Canonicalization, deduplication, and data splitting

The implementation MUST use at least three hashes:

* `surface_hash`: exact canonical source text
* `alpha_hash`: source canonicalized by renaming statement numbers and variables by order of first appearance within each variable class
* `shape_hash`: a coarser AST structure hash that ignores superficial naming

Data splits MUST be performed at least by `alpha_hash`, not by raw text alone. Otherwise the train/test boundary will leak near-identical source skeletons with only renumbered labels or variables.

The split pipeline SHOULD produce:

* `historical_golden`

  * shipped example programs and any hand-entered manual examples
* `synthetic_train`
* `synthetic_dev`
* `synthetic_test`
* `adversarial_test`

  * edge-of-limit statement lengths
  * bound-tight headers
  * sparse statement numbers
  * high branch density
  * composite indexing near bound edges

## 15. Golden tests and validation requirements

The coding agent MUST ship these tests before large-scale generation:

1. **SIMH build test**

   * build IBM 650 simulator successfully

2. **Shipped example reproduction**

   * run the shipped IT example workflows successfully

3. **Translate-only equivalence**

   * custom stage-separated pipeline reproduces the same downstream behavior as the shipped `run_it.ini` example flow, modulo explicitly documented canonicalization differences such as discardable edge blank cards. ([Open SIMH][7])

4. **Header math test**

   * prime example computes `N = 1672` from `2, 0, 50, 10, 265` exactly. ([GitHub][6])

5. **P1 restriction test**

   * exponentiation is rejected or excluded from v1 generation, because P1 lacks the required runtime support. ([GitHub][4])

6. **Reservation-card split test**

   * translation output final ten cards are preserved and reassembled into `pit_phase2_input_p1` exactly as specified

7. **Archival renderer test**

   * if `it_card80_raw` is implemented, statement-length and `F`/`FF` placement rules are enforced exactly. ([Internet Archive][1])

8. **Optional parser invariant**

   * if a PIT parser is implemented later, assert that the first instruction of the first statement lands at symbolic location corresponding to historical location 1992, as described by the manual. ([Internet Archive][1])

9. **Pilot dataset integrity**

   * generate at least 100 accepted end-to-end samples with complete artifact records and no broken references

## 16. Training data preparation and prompt contracts

Use supervised fine-tuning first. Do not start with RL, DPO, or any preference pipeline.

Primary prompt contract:

```text
System:
Compile the following IBM 650 IT program to canonical PIT deck output.
Return only the PIT deck, one card per line, with no explanation.

User:
<IT>
... canonical it_text_v1 ...
</IT>

Assistant:
<PIT>
... pit_raw_canonical ...
</PIT>
```

Rules:

* The assistant completion MUST contain only target deck text
* No natural-language explanation
* No comments added by the modern model
* Preserve exact line order
* Use the primary target `pit_raw_canonical`

Optional auxiliary dataset variants:

1. `pit_phase2_input_p1` target
2. structured scratchpad target, but **only** in deterministic machine-readable form, for example:

   * computed header fields
   * variable bound summary
   * branch target map
   * final PIT deck

Do not use freeform natural-language chain-of-thought as the baseline target. If a reasoning-preservation experiment is desired, make it an explicit ablation, not the default.

## 17. Model and fine-tuning plan

The default base model is `nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16`. The current model card identifies it as a Nemotron-Hybrid model compressed from `NVIDIA-Nemotron-Nano-9B-v2`, with `3.97 x 10^9` parameters, release date **2026-03-16**, and support for contexts up to 262K tokens. That maximum context is irrelevant to this workload. Training SHOULD cap context far lower, typically 4K to 8K, because IT and PIT programs are short and shorter sequences reduce cost and instability. ([Hugging Face][8])

Use QLoRA first. Keep the model/training wrapper modular so a later swap to a smaller or different model does not require rewriting the dataset and evaluation stack. Unsloth currently documents support for fine-tuning Nemotron 3 models and notes that 16-bit LoRA on Nemotron 3 Nano uses around 60 GB VRAM. Its requirements page also describes its published VRAM table as **absolute minimums**, not comfortable operating points. As a conservative engineering choice, use 24 GB or more of GPU memory for reproducible training runs, while allowing smaller-memory smoke tests if they prove stable. ([Unsloth - Train and Run Models Locally][9])

If a structured reasoning experiment is added, prefer deterministic scratchpads. Unsloth’s current Nemotron guidance says reasoning retention can be helped by mixing direct-answer and chain-of-thought examples and suggests a 75/25 mix; treat that as an optional ablation only. It is not a required baseline for this deterministic translation task. ([Unsloth - Train and Run Models Locally][9])

Recommended starting hyperparameters:

* QLoRA 4-bit
* rank 32
* alpha 64
* dropout 0.05
* learning rate `1e-4`
* cosine decay
* warmup ratio 0.03
* epochs 3
* effective batch size tuned by gradient accumulation
* greedy or near-greedy evaluation decoding
* gradient checkpointing enabled

The training code MUST support:

* zero-shot evaluation
* few-shot evaluation
* fine-tuned evaluation
* resumed training from checkpoints
* export of adapters and merged weights where supported

## 18. Evaluation protocol

The evaluation harness MUST report all of the following:

1. **Exact deck match**

   * full exact match on `pit_raw_canonical`

2. **Per-card exact match**

   * percentage of cards matched at identical positions

3. **Normalized edit distance**

   * sequence-level

4. **Assemblability**

   * percent of model outputs that patched SOAP II accepts after deterministic phase-two deck construction

5. **Functional equivalence**

   * percent of assembled model outputs whose execution output matches the reference output exactly

6. **Bucketed scores**

   * by curriculum band
   * by statement count
   * by expression depth
   * by indexed-variable usage
   * by loop presence

7. **Failure taxonomy**

   * malformed source echo in output
   * malformed PIT card
   * wrong symbolic instruction
   * wrong constant
   * wrong transfer target
   * wrong reservation handling
   * unassemblable output
   * assembles but misexecutes
   * timeout / non-halting behavior

8. **Baseline delta**

   * zero-shot base model
   * few-shot base model
   * fine-tuned model

For any model output that is not an exact PIT match but still assembles and executes equivalently, record it as a **functional success / exact failure**. That distinction matters.

The implementation SHOULD also include a shrinker that reduces a failing source program to a minimal counterexample while preserving the observed failure mode.

## 19. Required milestones and acceptance criteria

### M0. Source locking and build

Done when:

* all external sources are fetched
* `sources.lock.json` exists
* SIMH builds reproducibly

### M1. Reference pipeline

Done when:

* custom stage-separated pipeline runs
* shipped examples succeed
* all intermediate artifacts are preserved

### M2. Generator and bounds

Done when:

* AST renderer exists
* header computation exists
* generator can emit accepted B0 and B1 programs

### M3. Pilot corpus

Done when:

* 1,000 accepted records exist
* dedupe and split logic works
* every accepted record has complete provenance

### M4. Training smoke

Done when:

* SFT data preparation runs
* a smoke fine-tune completes
* zero-shot, few-shot, and fine-tuned evaluation scripts all run end-to-end

### M5. Full v1 corpus

Done when:

* at least 10,000 accepted unique records exist
* evaluation reports break down by band and feature
* failure archive is populated with counterexamples

### M6. First research-quality result

Done when:

* a full report exists with exact, assemblability, and functional metrics
* the report includes baseline comparisons and failure taxonomy
* all commands are reproducible from a fresh checkout

## 20. Stretch goals

After v1 is stable:

* add `READ` and full input-deck generation
* add P2/P3/P4 package support
* add exponentiation
* add extension statements
* train on `pit_phase2_input_p1` as an auxiliary target
* train direct `IT -> SPIT` or direct machine-code output
* add a comment-stripped semantic PIT target
* compare against smaller models
* extend to RUNCIBLE
* build an interactive demo only after the deterministic oracle and evaluation harness are solid

## 21. Final implementation rules

* Preserve all raw artifacts.
* Make all transforms idempotent and versioned.
* Never let the model invent deterministic scaffolding that can be supplied by the pipeline.
* Do not collapse historical stages for convenience.
* Keep the learned task narrow, measurable, and historically faithful.
* Ship the oracle, the corpus builder, the trainer, and the evaluator as one coherent reproducible system.

[1]: https://archive.org/stream/bitsavers_ibm650Carntor_16304233/CarnegieInternalTranslator_djvu.txt "https://archive.org/stream/bitsavers_ibm650Carntor_16304233/CarnegieInternalTranslator_djvu.txt"
[2]: https://opensimh.org/simdocs/i650_doc.html?utm_source=chatgpt.com "IBM 650 Simulator User’s Guide | Open SIMH"
[3]: https://archive.computerhistory.org/resources/access/text/2012/07/102726995-05-01-acc.pdf "https://archive.computerhistory.org/resources/access/text/2012/07/102726995-05-01-acc.pdf"
[4]: https://github.com/open-simh/simh/raw/refs/heads/master/I650/sw/it/00_readme.txt "https://github.com/open-simh/simh/raw/refs/heads/master/I650/sw/it/00_readme.txt"
[5]: https://raw.githubusercontent.com/open-simh/simh/refs/heads/master/I650/sw/run_it.ini "https://raw.githubusercontent.com/open-simh/simh/refs/heads/master/I650/sw/run_it.ini"
[6]: https://github.com/open-simh/simh/blob/master/I650/sw/it/it_example_2_src.txt "https://github.com/open-simh/simh/blob/master/I650/sw/it/it_example_2_src.txt"
[7]: https://opensimh.org/simdocs/i650_doc.html "https://opensimh.org/simdocs/i650_doc.html"
[8]: https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16 "https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16"
[9]: https://unsloth.ai/docs/models/nemotron-3 "https://unsloth.ai/docs/models/nemotron-3"