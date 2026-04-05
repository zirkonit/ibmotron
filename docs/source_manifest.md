# Source Manifest

`sources.lock.json` is the authoritative machine-readable source lock.

Locked source families:

- `open-simh/simh`: pinned simulator checkout plus the exact IT/SOAP decks and example files used by the oracle.
- Historical docs: `CarnegieInternalTranslator.pdf` and IBM 650 manual material used to confirm machine format and storage addresses.
- Training stack: the current Hugging Face model card/API metadata for `nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16`, plus current Unsloth Nemotron 3 and requirements pages.

The lock generator lives at [scripts/fetch_sources.py](/Users/i/Library/CloudStorage/Dropbox/claude-sandbox/ibmotron/scripts/fetch_sources.py).
