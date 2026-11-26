# CI workflow notes

This file documents the non-obvious parts of `.github/workflows/tests.yml`.
Last updated: 2026-11-26.

## Jobs at a glance

| Job | Runner | Runs on | Notes |
| --- | --- | --- | --- |
| `test` | `ubuntu-24.04` | PRs, schedule, manual dispatch | Matrix over Python 3.11/3.12/3.13. Slow tests run only on 3.12 for scheduled/manual runs. |
| `examples-fast` | `ubuntu-24.04` | PRs | Runs `pytest -m "not slow"` for each example. |
| `examples-slow` | `ubuntu-24.04` | Schedule, manual dispatch | Runs `pytest -m slow` for each example; only examples that install vLLM start the engine. |

## Why the runner is pinned to `ubuntu-24.04`

**Symptom** (observed in the scheduled run on 2026-11-26):

```text
WARNING ... Failed to import from vllm._C:
ImportError('.../vllm/_C.abi3.so: cannot enable executable stack as shared object requires: Invalid argument')
...
AttributeError: '_OpNamespace' '_C' object has no attribute 'init_cpu_memory_env'
...
RuntimeError: Engine core initialization failed.
```

**Cause:**

1. GitHub migrated `ubuntu-latest` from Ubuntu 24.04 to Ubuntu 26.04 between
   2026-10-19 and 2026-11-19. The failing run used `Image: ubuntu-26.04`
   with glibc 2.43.
2. Since glibc 2.41, `dlopen` no longer makes the stack executable for a shared
   object that requests it. Loading such an object now fails instead. See the
   glibc 2.41 announcement in the references below.
3. The vLLM 0.27.1 CPU wheel ships `vllm/_C.abi3.so` with
   `PT_GNU_STACK = RWE` (executable stack). The extension therefore fails to
   load, vLLM continues without `_C`, and the CPU worker dies at
   `torch.ops._C.init_cpu_memory_env`.
4. Ubuntu 24.04 has glibc 2.39, which still permits the executable stack, so
   the same tests passed before the `ubuntu-latest` migration. This is **not**
   related to the Python version or to the test code.

The pin is the primary compatibility fix. Keep it until either GitHub retires
the `ubuntu-24.04` runner image or a vLLM release ships CPU wheels linked with
`-z noexecstack`. Do not switch these jobs back to `ubuntu-latest` without
running the patch step below and verifying the slow jobs.

## Executable-stack patch (defense in depth)

Even with the runner pin, the workflow clears the flag after installing
dependencies. This makes the workflows forward-compatible with Ubuntu 26.04+
and with any future change to the runner image.

The helper is
[`.github/workflows/patch_vllm_execstack.py`](./patch_vllm_execstack.py). It:

1. Walks the installed `vllm` package and finds every `*.so*` file.
2. Reads each file's ELF program headers directly (no `readelf` dependency).
3. Runs `patchelf --clear-execstack` on every 64-bit ELF object whose
   `PT_GNU_STACK` header has the executable bit set.
4. Re-checks all extensions and exits non-zero if any executable-stack flag
   remains.

It is idempotent and only touches ELF files. The flag is normally a build
leftover; clearing it is safe for libraries that do not execute code on the
stack.

### Where it runs

- `test` job: immediately before the slow tests, under the same
  `schedule`/`workflow_dispatch` + Python 3.12 condition. The fast suite does
  not need it because it ignores the vLLM engine tests.
- `examples-slow` job: before the slow example tests. The step first checks
  whether `vllm` is importable; examples such as `min_working_example` do not
  install vLLM and are skipped.

The patch also allowed re-enabling `multi_resource_test` in the
`examples-slow` matrix. It had been removed because its only slow test starts
the same vLLM CPU engine; its fast tests already run in `examples-fast`.

### Requirements and maintenance

- `patchelf` is installed with `apt-get` in the workflow. It must support
  `--clear-execstack` (Ubuntu 24.04/26.04 ship 0.18.0, which does).
- The patch must be reapplied after every reinstall of vLLM; the workflow does
  this automatically because it runs after dependency installation.
- Once vLLM ships a fixed CPU wheel, the patch becomes a no-op and can be
  removed together with the `patchelf` install steps.
- Manual verification:

  ```bash
  SITE="$(python -c 'import pathlib, vllm; print(pathlib.Path(vllm.__file__).parent)')"
  readelf -lW "$SITE/_C.abi3.so" | grep GNU_STACK
  # Expected: GNU_STACK ... RW ...  (not RWE)
  ```

### Alternative that is intentionally not used

Setting `GLIBC_TUNABLES=glibc.rtld.execstack=2` also makes the library load,
but it enables an executable stack for the entire process. The `patchelf`
approach is targeted and does not weaken the process-wide hardening.

## Slow-test selection and pytest exit code 5

Before this change, the slow step was:

```bash
pytest -m slow --ignore=tests/test_vllm_model_integration.py
```

The only `@pytest.mark.slow` tests under `tests/` are the 8 tests in
`tests/test_vllm_model_integration.py`. Ignoring that file therefore selected
zero tests, and pytest exited with code **5** ("no tests collected"), which CI
reported as a failure:

```text
collected 410 items / 410 deselected / 0 selected
```

The current behaviour is:

- **Fast step** keeps
  `--ignore=tests/test_vllm_model_integration.py`. That file also contains
  non-slow tests (`test_init_*`, `test_disable_reasoning_parser`) that start a
  vLLM engine; keeping them out of the fast suite keeps PR feedback quick.
- **Slow step** no longer ignores the file and runs `pytest -m slow`, so the 8
  slow integration tests are selected and executed on Python 3.12 after the
  executable-stack patch.

If the slow tests should ever not run in CI, remove or guard the step rather
than leaving an empty selection in place (pytest treats "no tests collected"
as an error).

## Reproducing locally

On Ubuntu 26.04+ (or any glibc >= 2.41 system):

```bash
sudo apt-get update
sudo apt-get install --no-install-recommends patchelf

SITE="$(python -c 'import pathlib, vllm; print(pathlib.Path(vllm.__file__).parent)')"
python .github/workflows/patch_vllm_execstack.py "$SITE"

pytest -m "not slow" --ignore=tests/test_vllm_model_integration.py
pytest -m slow
```

For an example virtual environment:

```bash
PYTHON=examples/random_number/.venv/bin/python
SITE="$("$PYTHON" -c 'import pathlib, vllm; print(pathlib.Path(vllm.__file__).parent)')"
"$PYTHON" .github/workflows/patch_vllm_execstack.py "$SITE"
"$PYTHON" -m pytest examples/random_number/tests -m slow
```

## pytest-asyncio loop scope

`pyproject.toml` now sets `asyncio_default_fixture_loop_scope = "function"`
under `[tool.pytest.ini_options]`. This matches pytest-asyncio's current
default and silences its deprecation warning
(`The configuration option "asyncio_default_fixture_loop_scope" is unset`).
It is unrelated to the executable-stack and slow-test issues above.

## References

- glibc 2.41 release announcement (executable-stack change):
  <https://lists.gnu.org/archive/html/info-gnu/2025-01/msg00014.html>
- Ubuntu 26.04 / glibc 2.41+ write-up:
  <https://lesstep.com/github-actions/errors/executable-stack-ubuntu-26-04/>
- `patchelf` approach and background:
  <https://honestkit.github.io/guides/cannot-enable-executable-stack-as-shared-object-requires/>
- vLLM CPU wheel and `patchelf --clear-execstack`:
  <https://docs.mulgadc.com/hardware/cisco/llm-serving>
- Previous CI workaround (commit `8a25ce7`, 2026-11-05) that added
  `--ignore=tests/test_vllm_model_integration.py`: it addressed the
  executable-stack failure for the fast suite but left the slow step with an
  empty selection.
