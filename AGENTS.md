# AGENTS.md

This file provides guidance to AI coding agents (Claude Code, etc.) when working with code in this repository. CLAUDE.md is a symlink to this file.

## Core Principles (CRITICAL)

**Less is more. The simplest solution is the best solution.** The action hierarchy for every change: **Delete > Replace > Add**.

1. **Solve at the owner**: Put behavior in the code path that owns or observes it. For fixes, never guard a symptom with a staleness check, initialization flag, skip-first-call branch, or `try/except` around broken logic; relocate the trigger and delete the wrong path. For features, extend the existing owner rather than creating a parallel abstraction.
2. **Search and reuse first**: Search the whole repository before creating a feature, component, helper, workflow, or utility. Reuse or adapt what exists, consolidate in-scope duplication in the shared owner, and delete duplicate paths. Three similar lines beat a helper nobody else calls.
3. **Delete and modify existing code before creating new code**: Bugfixes are net-negative by default unless deletion and relocation are demonstrably impossible. A new file must first prove it cannot fit cleanly in an existing owner.
4. **Keep scope minimal**: Implement only the simplest complete solution. Avoid impossible-state handling, speculative flags, compatibility shims, policy scaffolding, and unrelated cleanup. Tests are out of scope by default — rely on existing coverage and focused validation; only an uncovered, high-risk regression path justifies minimal new test code.
5. **Ship zero-regression, production-ready changes**: Understand what you remove instead of retaining broken code as insurance. Remove unused imports, functions, types, files, and comments; run relevant cleanup checks; and thoroughly debug and validate the changed owner. Do not break existing features or workflows unless the PR intentionally removes them with evidence.

**Review gate:** for every addition, the reviewer decides whether deleting or changing existing code would have fixed the problem instead — if it would, that is a blocking finding. A missing or thin PR description is never itself a finding.

NEVER push to `main`. NEVER force push. Always start work in a new git worktree (`git worktree add`) on a feature branch and open a PR — never edit the primary checkout directly, it may hold in-flight work.

## PR Workflow

After opening a PR:

1. Wait for the automated PR review and auto-format commit from Ultralytics Actions (`format.yml`), then pull and address every finding.
2. Review the full diff in-session against the Core Principles, performance, and the review gate above, then batch the fixes into one commit and push. After each round of bot or human commits, pull and resume the same reviewer on `<last-reviewed-sha>..HEAD` plus anything that delta could have invalidated. Repeat until the local head matches the live head.
3. Hand off or merge only on a clean final pass: one cold full-diff review returning LGTM with no findings, on a head that is still live at merge time.
4. Never fight other commits: Ultralytics Actions pushes auto-format and header commits, and multiple users may work on the same PR. `git pull --rebase` before pushing; never reset or revert commits you did not author.
5. After the PR merges, clean up: remove local worktrees and branches for it, then `git checkout main && git pull`.

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

- The default branch is `master`, not `main` — target PRs at `master`.
- Ultralytics Actions (`format.yml`) auto-formats PRs (Ruff, docformatter, codespell, prettier) and adds the `# Ultralytics 🚀 AGPL-3.0 License` header — never add or revert headers or formatting manually.
- Google-style docstrings, 120-char lines (Ruff/isort/docformatter all configured in `pyproject.toml`); every larger class and function needs a Google-style docstring (Args/Returns sections), while a one-line summary suffices for small helpers.
- The CI smoke tests hit the live network: they download `yolov3-tiny.pt` from the v9.6.0 release and the coco128 dataset from `github.com/ultralytics/assets`.
- Keep `requirements.txt` and `pyproject.toml` dependency floors aligned — Dependabot bumps both (monthly pip, weekly github-actions).
- Links to `github.com/ultralytics/yolov5/(issues|pull|discussions)/<N>` are intentional upstream provenance — do not rewrite them to `yolov3` (those numbers 404 there). Bare yolov5 repo/tree/releases links were already rebranded.
- README, docstrings, and tutorial content must stay evergreen and YOLOv3-focused: historical facts are fine, but no "latest/NEW/SOTA" promo for other models — reference the broader family only via version-less `github.com/ultralytics/ultralytics` pointers.
