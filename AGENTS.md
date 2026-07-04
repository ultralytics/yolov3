# AGENTS.md

This file provides guidance to AI coding agents (Claude Code, etc.) when working with code in this repository. CLAUDE.md is a symlink to this file.

## Core Principles (CRITICAL)

Respecting these principles is critical for every PR.

**Less is more. The simplest solution is the best solution.**

The action hierarchy for every change: **Delete > Replace > Add**. The best code change is a deletion. The second best is modifying what exists. Adding new code is the last resort.

1. **Minimal**: The simplest solution that works. Do not over-engineer, over-abstract, or add code just in case. Three similar lines beat a premature abstraction. Avoid error handling for impossible states, feature flags, compatibility shims, or policy scaffolding unless they are truly required.
2. **Solve at the source**: Do not hack fixes. Solve problems at their root. If something is broken, fix or remove the broken thing. Never patch over a broken abstraction, add workarounds, or add synchronization code for state that should not be duplicated.
3. **Delete ruthlessly**: When replacing code, delete what it replaced. Remove unused imports, functions, types, files, and commented-out code. Git preserves history. Run the repo's relevant dead-code or cleanup check when available.
4. **Replace > Add**: Modify existing code over adding new code. Edit existing files, extend existing components or functions with minimal parameters, and reuse existing utilities. If creating a new file, first prove it cannot fit cleanly in an existing file.
5. **Check existing**: Search the entire repo before creating anything new. If a feature, component, helper, responder, workflow, or utility already solves a similar problem, reuse or adapt it and delete the duplicate path.
6. **Deduplicate**: Do not duplicate existing code when updating the repo. Consolidate or refactor duplicates you find when it is in scope and low risk.
7. **Zero Regression**: Do not break existing features or workflows unless the PR intentionally removes them with evidence.
8. **Production ready**: All changes must be thoroughly debugged, validated, and production ready.

**When fixing bugs, ask: "What can I delete?" before "What can I replace?" before "What should I add?"**

## PR Workflow

After opening a PR:

1. Wait for the automated PR review and auto-format commit from Ultralytics Actions (`format.yml`), then pull and address every finding.
2. Launch an independent adversarial review agent with cold context (just the PR diff and this file) to hunt for bugs, regressions, and Core Principles violations — use the Codex CLI, one fresh `codex exec` run per round. Fix, push, and repeat until a fresh run reports LGTM.
3. Never fight other commits: Ultralytics Actions pushes auto-format and header commits, and multiple users may work on the same PR. `git pull --rebase` before pushing; never force-push, reset, or revert commits you did not author.
4. After the PR merges, clean up: remove local worktrees and branches for it, then `git checkout main && git pull`.

## Commands

```bash
uv pip install -r requirements.txt # install (never bare pip install)

# There is no pytest suite. CI smoke-tests the real scripts; the full loop (val/detect on both
# official and trained weights, plus torch.hub custom-load traces) is in .github/workflows/ci-testing.yml.
# Fast local subset (--name smoke --exist-ok pins the save dir; without it train auto-increments runs/train/exp<N>):
python train.py --imgsz 64 --batch 32 --weights yolov3-tiny.pt --cfg yolov3-tiny.yaml --epochs 1 --device cpu --name smoke --exist-ok
python val.py --imgsz 64 --batch 32 --weights runs/train/smoke/weights/best.pt --device cpu
python detect.py --imgsz 64 --weights yolov3-tiny.pt --device cpu
python export.py --weights yolov3-tiny.pt --img 64 --include torchscript
python models/yolo.py --cfg yolov3-tiny.yaml # build model from YAML
python hubconf.py --model yolov3-tiny        # PyTorch Hub load test

ruff format . && ruff check --fix . # format/lint (line-length 120, source of truth: pyproject.toml [tool.ruff])
```

CI (`ci-testing.yml`) runs the smoke tests on ubuntu-latest and windows-latest with latest-stable Python, plus ubuntu jobs on Python 3.11 and on the Python 3.8 + torch 1.8.0 floor — keep code compatible with Python>=3.8 and PyTorch>=1.8, and never assume newer APIs without a version gate.

## Architecture

This is a YOLOv5-lineage training/inference codebase packaging the three classic YOLOv3 **detection-only** models (yolov3, yolov3-spp, yolov3-tiny) — no segmentation, classification, or YOLOv5 weights exist here. Entry points are flat scripts at the repo root: `train.py`, `val.py`, `detect.py`, `export.py`, `benchmarks.py`, plus `hubconf.py` exposing `yolov3`/`yolov3_spp`/`yolov3_tiny`/`custom` for `torch.hub.load`. Models are defined declaratively in `models/*.yaml` and built by `parse_model()` in `models/yolo.py`; `models/common.py` holds the layer zoo and the `DetectMultiBackend` multi-format inference wrapper; `utils/` holds dataloaders, loss, metrics, plotting, and loggers.

The repo depends on the `ultralytics` pip package and re-exports many helpers from it (see `utils/general.py`, `utils/torch_utils.py`); functions annotated `Keep local (do not dedup)` differ deliberately from their upstream namesakes (return arity, rounding, objectness channel) — do not "deduplicate" them. TensorFlow _export_ was removed, but the TF rows in `export.py:export_formats()` are load-bearing: they are positionally coupled to `DetectMultiBackend` suffix detection and `benchmarks.py`.

Pretrained weights download from the GitHub release `v9.6.0` assets via `utils/downloads.py:attempt_download`. There is no PyPI publish workflow; releases are GitHub tags carrying `.pt` assets. `docker.yml` builds and pushes `ultralytics/yolov3:{latest,latest-cpu,latest-arm64}` to Docker Hub on every push to `master` (gated to the `ultralytics/yolov3` repo).

## Conventions

- The default branch is `master`, not `main` — read `main` as `master` in the PR Workflow above, and target PRs at `master`.
- Ultralytics Actions (`format.yml`) auto-formats PRs (Ruff, docformatter, codespell, prettier) and adds the `# Ultralytics 🚀 AGPL-3.0 License` header — never add or revert headers or formatting manually.
- Google-style docstrings, 120-char lines (Ruff/isort/docformatter all configured in `pyproject.toml`); every larger class and function needs a Google-style docstring (Args/Returns sections), while a one-line summary suffices for small helpers.
- The CI smoke tests hit the live network: they download `yolov3-tiny.pt` from the v9.6.0 release and the coco128 dataset from `github.com/ultralytics/assets`.
- Keep `requirements.txt` and `pyproject.toml` dependency floors aligned — Dependabot bumps both (monthly pip, weekly github-actions).
- Links to `github.com/ultralytics/yolov5/(issues|pull|discussions)/<N>` are intentional upstream provenance — do not rewrite them to `yolov3` (those numbers 404 there). Bare yolov5 repo/tree/releases links were already rebranded.
- README, docstrings, and tutorial content must stay evergreen and YOLOv3-focused: historical facts are fine, but no "latest/NEW/SOTA" promo for other models — reference the broader family only via version-less `github.com/ultralytics/ultralytics` pointers.
