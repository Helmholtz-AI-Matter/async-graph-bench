# Async Graph Bench

Asynchronous Benchmarking of Large Language Models and other Models

## What is this?

The `async_graph_bench` framework provides a flexible, modular way to design and run benchmarks as **directed acyclic
graphs (DAGs)**. It is designed for multistep computations, where intermediate computation steps are encapsulated by a
`Node` that defines its required and provided dependencies. The framework manages execution order, resource scheduling, and caching automatically.

## Core Concepts

* **Graph-Based Execution**

    * The `BenchmarkManager` builds a **directed acyclic graph (DAG)** of nodes based on their declared dependencies.
    * Each node represents a calculation, transformation, or metric applied to incoming data.
    * The graph ensures results are computed in the correct order and reused where possible.

* **Asynchronous Execution**

    * Nodes are executed asynchronously, allowing the framework to make efficient use of external resources (e.g., GPUs, APIs, or databases).
    * Although mainly designed for concurrent programming, custom resources using multiprocessing can be leveraged to run computations in parallel locally.

* **Data Flow**

    * A `DataSource` emits items (Python `dict`s).
    * These items travel through the graph and are progressively extended with additional dependencies computed by nodes.

* **Stepwise Execution**

    * For resource-limited environments, benchmarks can be executed in **multiple steps**.
    * Nodes are activated iteratively so that graphs too large to compute in one pass can still be processed in a single benchmark script.

* **Iterative Runs & Sampling**

    * Benchmarks can run multiple iterations over the same dataset.
    * This enables the computation of scores or statistics across these multiple iterations using sampling strategies, which is especially interesting for scenarios where resources are involved that carry inherent randomness (e.g. LLM inference).

* **Node Utilities via Configuration**

    * Nodes can be extended with optional features such as:

        * **Caching/Result persistence**: Caching of outputs from nodes, which may be reused across benchmark runs to avoid redundant computation.
        * **Batching**: Process items in groups for efficiency when working with expensive resources.

## Installation

Install locally using `pip install -e .`. vLLM is optional and can be installed with
`pip install -e ".[vllm]"`.

## Examples

You can find examples for different benchmarking applications in `examples/`.

## Documentation

The documentation is mainted in `docs/`. These are sphinx-generated pages. We are currently working to host as webpages. In the meantime, please see `docs/README.md` for details on how to build them locally.

## Development

Install development dependencies with the CPU-only vLLM build:

```bash
python -m pip install uv
uv pip install --extra-index-url https://wheels.vllm.ai/0.27.1/cpu --torch-backend cpu -e ".[dev]"
```

For a regular vLLM installation, use `pip install -e ".[vllm]"` and select the
appropriate vLLM and PyTorch wheels for your hardware (with `uv`, consider to use the `--torch-backend` CLI parameter).
For more vllm installation hints, see [here](https://vllm.ai/).

**Code Hygiene**

Consider to execute the checks below to keep the code you contribute clean. All tooling will be installed with the `dev` installation target.

| Check | Command |
|---|---|
| Run unit tests | `pytest` |
| Lint code | `ruff check` |
| Auto-fix lint issues | `ruff check --fix` |
| Format code | `ruff format` |

**Example Tests**

Each use case in `examples/` has its own pytest tests under a local `tests/`
directory. The tests are designed to validate the examples without requiring
external LLM services during fast checks.

Run the fast tests for one example with:

```bash
pytest examples/<example>/tests -m "not slow"
```

Run the slow integration tests with:

```bash
pytest examples/<example>/tests -m slow
```

Fast tests use deterministic mock resources and also exercise the CLI help
paths where an example provides a CLI. Slow tests use the tiny random vLLM
model for the LLM examples. The `min_working_example` slow test uses its real
`DummyNoiseResource` instead.

GitHub Actions creates a separate virtual environment for each example and
installs the main library into it. Fast example tests run for pull requests;
slow example tests run once daily or through `workflow_dispatch`. Slow tests
require the vLLM dependencies and may download the tiny model into the
Hugging Face cache.

### Managing Example Test Environments

Use an individual virtual environment for each example. This keeps example
dependencies isolated and verifies that the example works with the installed
main library rather than relying on packages from another environment.

From the repository root, replace `resource_benchmark` with the example you
want to test:

```bash
example=resource_benchmark
python -m venv "examples/$example/.venv"
source "examples/$example/.venv/bin/activate"
python -m pip install --upgrade pip
python -m pip install -e .
if [ -f "examples/$example/requirements.txt" ]; then
    python -m pip install -r "examples/$example/requirements.txt"
fi
python -m pip install pytest pytest-asyncio
pytest "examples/$example/tests" -m "not slow"
deactivate
```

For PowerShell, activate the environment with:

```powershell
examples\resource_benchmark\.venv\Scripts\Activate.ps1
```

The `.venv` directories are ignored by Git. GitHub Actions creates these
individual environments automatically for the example test jobs.
