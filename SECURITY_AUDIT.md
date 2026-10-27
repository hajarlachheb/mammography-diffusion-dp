# Security and privacy audit

Date 2026-10-27. Review covers the initial public release and subsequent safety/documentation updates; see `git log`.

## What was scanned

- **Coverage.** Every file in the candidate, including code, comments, docstrings, configuration, CSV, LaTeX and the
  embedded metadata of figure PDFs and PNGs. The scan reads file **contents**, not only names.
- **Where the scanner lives.** It is kept outside the repository with the project's private build records, because its
  pattern list necessarily names private paths and hosts.

| Category | Method | Result |
|---|---|---|
| Internal paths, usernames, host names, mount names | Regex for absolute paths under user-home, media and mount directories, internal mount and host names, venv paths, temporary directories | **0** outside an intentional, generic pattern list in `tests/test_paths_and_cli.py` |
| IP addresses, e-mail addresses | Regex | **0** |
| API keys, tokens, passwords | Regex (GitHub, Hugging Face, AWS, OpenAI-style, Slack tokens; `password/secret/token = "..."` assignments) | **0** |
| SSH / VPN material | Private-key headers, `ssh-rsa`, WireGuard / OpenVPN markers | **0** |
| Patient identifiers | CBIS `P_xxxxx`, BCDR `patientbcdr__n`, EUCanImage subject patterns | Only invented test-fixture IDs (`P_9000x`, `P_99999`, `patientbcdr__80x`) |
| Medical images | Magic bytes (PNG, JPEG, DICOM, NIfTI suffixes) | Only the 5 publication-figure PNGs (plots and schematic; no image content) |
| Checkpoints / serialized objects | Suffixes and magic bytes (`.pth`, `.pt`, `.ckpt`, `.safetensors`, pickle, zip) | **0** |
| Training logs | `.log` files and log-like content | **0**. Run logs, which include server paths, are excluded; only aggregate CSVs are released |
| Large files | > 1 MB | **0**. Largest file < 0.6 MB |
| Python bytecode caches | `__pycache__/*.pyc` | **Found and removed** during the audit: they embed absolute build paths. They are ignored by `.gitignore` and absent from the commit and the ZIP |

## Code-level review

- **No network calls** in repository code, other than model downloads by third-party libraries:
  - torchvision Swin-T weights;
  - Hugging Face SD-2.1 for CC-NET;
  - pytorch-fid Inception weights.
- **No credentials** are read or required. `.env.example` contains only relative directory paths.
- **Checkpoint loading uses `torch.load(weights_only=True)`** to restrict deserialization to tensors and supported
  primitive data, avoiding arbitrary pickle-code execution. Do not load untrusted checkpoints: malformed files can
  still exhaust resources, and legacy checkpoints containing custom Python objects may fail to load. Convert legacy
  files only in a trusted, isolated environment.
- **Opacus secure RNG is disabled** (`secure_mode=False`) to match the reported experimental runs. This uses a standard
  PyTorch RNG, not a cryptographically secure RNG. It is not appropriate for a production privacy release; see
  `docs/privacy.md` for the required secure-mode retraining guidance.
- **Path checks.** `run_experiment.py`, `build_training_folders.py` and `organize_mcgan.py` refuse to overwrite existing
  directories.

## Git staging check

- **Tracked content.** The published tree contains no ignored or sensitive path.
- **`.gitignore` behaviour.** Tested with `git check-ignore` on sample paths: `.env`, `data/`, `runs/`, `*.pth`,
  `*.png` outside `figures/publication/`, `*.log`, `*.npz`, `__pycache__/`.
- **History.** Review the current commit and its changes with `git log`; release updates are kept in the normal
  project history.
