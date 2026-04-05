# Dataset Schema

The dataset is file-backed with a JSONL index. Large deck bodies remain in files; records store paths and hashes.

Minimum record fields implemented:

- `id`, `band`, `seed`, `runtime_package`, `source_format`
- `source`
- `hashes`
- `reference.translate`
- `reference.assemble`
- `reference.run`
- `generator`
- `provenance`

Primary supervised target:

- `reference.translate.pit_raw_canonical`

Split boundary:

- split by `alpha_hash`, not by raw surface text alone
