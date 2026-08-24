# Multi-Resource Test Example

This example demonstrates running the benchmark against one or more
OpenAI-compatible model resources. It expects endpoint configuration in the
environment and selects resources through the command-line interface.

## Run

From this directory, select one, two, or both configured resources:

```bash
python run.py --models one
python run.py --models two
python run.py --models both
```

The example uses the `BLABLADOR_*` and `SCADS_*` environment variables for the
two resources. The commands make external model requests and are not used by
the fast CI tests.

## Tests

The tests are in `tests/` and can be run from the repository root:

```bash
pytest examples/multi_resource_test/tests -m "not slow"
```

Fast tests validate `QueryModel` with a deterministic mock model and exercise
the CLI help path without making model requests. The slow integration test
uses the tiny random vLLM model with one short prompt.

In GitHub Actions, each example is tested in its own virtual environment after
installing the main library. Fast tests run on pull requests. Slow tests run
once daily or can be started manually with `workflow_dispatch`.
