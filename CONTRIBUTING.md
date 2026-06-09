# Contributing to Ultralytics YOLOv3 🚀

We love your input! We want to make contributing to Ultralytics YOLOv3 as easy and transparent as possible, whether you're:

- Reporting a bug
- Discussing the current state of the code
- Submitting a fix
- Proposing a new feature
- Becoming a maintainer

Ultralytics YOLO models are successful thanks to the collective efforts of our community. Every improvement you contribute helps advance the possibilities of AI and computer vision! 😃

## 🚀 Submitting a Pull Request (PR)

We greatly appreciate contributions in the form of [pull requests](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/about-pull-requests). To make the review process as smooth as possible, please follow these steps:

1. **[Fork the repository](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/fork-a-repo):** Fork [ultralytics/yolov3](https://github.com/ultralytics/yolov3) to your GitHub account.
2. **[Create a branch](https://docs.github.com/en/desktop/making-changes-in-a-branch/managing-branches-in-github-desktop):** Create a branch in your fork with a clear, descriptive name (e.g., `fix-issue-123`, `add-feature-xyz`).
3. **Make your changes:** Keep them minimal and focused on a single bug fix or feature. Ensure your code follows the project's style and doesn't introduce new errors or warnings.
4. **Test your changes:** There is no pytest suite — the [CI workflow](https://github.com/ultralytics/yolov3/blob/master/.github/workflows/ci-testing.yml) smoke-tests the real scripts. Run a fast local equivalent before submitting:
   ```bash
   python train.py --imgsz 64 --batch 32 --weights yolov3-tiny.pt --cfg yolov3-tiny.yaml --epochs 1 --device cpu
   python val.py --imgsz 64 --batch 32 --weights runs/train/exp/weights/best.pt --device cpu
   python detect.py --imgsz 64 --weights yolov3-tiny.pt --device cpu
   ```
5. **[Create a pull request](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request):** Open a PR from your branch to the `master` branch of [ultralytics/yolov3](https://github.com/ultralytics/yolov3). Provide a clear title and a description explaining the purpose and scope of your changes.

### PR Best Practices

To ensure your contribution is integrated smoothly, please:

- ✅ Keep your PR **up-to-date** with the `master` branch. If it falls behind, click the 'Update branch' button or merge `master` locally.
- ✅ Confirm that all **Continuous Integration (CI) checks pass**.
- ✅ Limit changes to the **minimum required** for your bug fix or feature.
  _"It is not daily increase but daily decrease, hack away the unessential. The closer to the source, the less wastage there is."_ — Bruce Lee

## 🎨 Code Style and Formatting

Ultralytics YOLOv3 is formatted with [Ruff](https://github.com/astral-sh/ruff) using a line length of 120 (configured in `pyproject.toml`). When you open a PR, the [Ultralytics Actions](https://github.com/ultralytics/actions) bot automatically applies formatting — Ruff, docformatter, codespell, and prettier — so there is no need to fight its style. New functions and classes should include [Google-style docstrings](https://google.github.io/styleguide/pyguide.html) so the codebase stays readable and maintainable.

## 📝 CLA Signing

Before we can merge your pull request, you must sign our [Contributor License Agreement (CLA)](https://docs.ultralytics.com/help/CLA). This legal agreement ensures that your contributions are properly licensed, allowing the project to continue being distributed under the [AGPL-3.0 license](https://www.ultralytics.com/legal/agpl-3-0-software-license).

After you submit your PR, the CLA bot will guide you through the signing process. To sign, add a comment in your PR stating:

```text
I have read the CLA Document and I sign the CLA
```

## 🐛 Submitting a Bug Report

If you encounter an issue with Ultralytics YOLOv3, please submit a bug report!

To help us investigate, please provide a [minimum reproducible example](https://docs.ultralytics.com/help/minimum-reproducible-example). Your code should be:

- ✅ **Minimal** – Use as little code as possible that still produces the issue.
- ✅ **Complete** – Include all parts needed for someone else to reproduce the problem.
- ✅ **Reproducible** – Test your code to ensure it reliably triggers the issue.

Additionally, for [Ultralytics](https://www.ultralytics.com/) to assist, your code should be:

- ✅ **Current** – Verify the problem persists on the latest [`master` branch](https://github.com/ultralytics/yolov3/tree/master). Use `git pull` or `git clone` to get the latest version.
- ✅ **Unmodified** – The problem must be reproducible without custom modifications. [Ultralytics](https://www.ultralytics.com/) does not provide support for custom code.

If your issue meets these criteria, please open a new issue using the 🐛 **Bug Report** [template](https://github.com/ultralytics/yolov3/issues/new/choose), including your [minimum reproducible example](https://docs.ultralytics.com/help/minimum-reproducible-example) to help us diagnose and resolve the problem.

## 📜 License

By contributing, you agree that your submissions will be licensed under the [AGPL-3.0 license](https://www.ultralytics.com/legal/agpl-3-0-software-license).

---

Thank you for helping improve Ultralytics YOLOv3! Your contributions make a difference. For more on open-source best practices, see [GitHub's open source guides](https://opensource.guide/how-to-contribute/).
