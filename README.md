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

Install locally using `pip install -e .`

## Examples

You can find examples for different benchmarking applications in `examples/`.

## Documentation

The documentation is mainted in `docs/`. These are sphinx-generated pages. We are currently working to host as webpages. In the meantime, please see `docs/README.md` for details on how to build them locally.
