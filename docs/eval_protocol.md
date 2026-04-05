# Eval Protocol

The baseline evaluator reports:

- exact PIT deck match
- per-card exact match
- normalized edit distance
- assemblability through patched SOAP II
- functional equivalence through SIMH execution
- failure taxonomy
- bucketed scores by band, statement count, expression depth, loop presence, indexed usage, and feature tags

The repository also includes:

- failure archives with source, reference PIT, candidate PIT, and preserved logs
- zero-shot, few-shot, and fine-tuned mode comparison with deltas versus zero-shot

Model outputs that are not exact PIT matches but still assemble and execute equivalently should be retained as `functional_success_exact_failure`.
