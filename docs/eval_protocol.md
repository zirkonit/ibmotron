# Eval Protocol

The baseline evaluator reports:

- exact PIT deck match
- per-card exact match
- normalized edit distance

The repository also includes assembly and functional hooks so evaluation can expand to:

- assemblability through patched SOAP II
- functional equivalence through SIMH execution
- bucketed scores by band and source features
- failure taxonomy labels

Model outputs that are not exact PIT matches but still assemble and execute equivalently should be retained as `functional_success_exact_failure`.
