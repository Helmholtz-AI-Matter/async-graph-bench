# Installation Guide

This framework can be installed either directly from GitHub or via PyPI.

## Installation Options

### 1. From GitHub

```bash
pip install git+https://github.com/Helmholtz-AI-Matter/async-graph-bench
```

### 2. From PyPI

```bash
pip install PACKAGE_NAME TODO
```
---

## Requirements

* **Python 3.11+** is required.

Before installation, ensure you have a compatible Python environment (e.g., using `venv`, `uv` or `conda`).

---

## Optional Dependencies

Some features require additional packages depending on the models or visualization tools you plan to use:

| Feature                 | Required Packages | Purpose                                     |
| ----------------------- | ----------------- |---------------------------------------------|
| **vLLM model support**  | `vllm`, `torch`   | Required for running vLLM-based models      |
| **OpenAI API support**  | `openai`          | Required for querying OpenAI API endpoints  |
| **Graph visualization** | `graphviz`        | Required for rendering execution graphs     |

The package itself does not install vLLM. Install the optional vLLM extra as needed:

```bash
pip install -e ".[vllm]"
```

For development and CPU-only integration tests, install the development extra from
the CPU vLLM wheel index:

```bash
python -m pip install uv
uv pip install --extra-index-url https://wheels.vllm.ai/0.27.1/cpu --torch-backend cpu -e ".[dev]"
```

Other optional packages can be installed manually as needed, for example:

```bash
pip install vllm torch
pip install openai
pip install graphviz
```

[//]: # (Alternatively, install all optional dependencies with:)

[//]: # ()
[//]: # (```bash)

[//]: # (pip install PACKAGE_NAME[all])

[//]: # (```)
