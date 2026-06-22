# AGENTS.md

## 🎯 Project Goal

Implementing a DAG based modular benchmarking framework that executes benchmark tasks asynchronously.

## 🛠 Scientific Software Standards
To ensure scientific integrity and reproducibility, all contributions must adhere to:

1.  **Reproducible Environment**: 
    *   No naked `pip install`. Use `pyproject.toml` or `requirements.txt`.
    *   Prefer explicit version pinning for scientific libraries (e.g., `numpy==1.26.0`) to prevent breaking changes in numerical results.
2.  **Rigorous Testing**:
    *   Every new feature or bug fix must include a corresponding unit test in `/tests`.
    *   Focus on **edge cases** and **numerical stability** (use `pytest.approx` for floating point comparisons).
3.  **CI/CD Integration**:
    * All changes must be compatible with the GitHub Actions workflow.
    * The agent should verify that the test suite passes before declaring a task "complete."
4.  **Documentation**:
    *   Docstrings must follow the NumPy/Google format, explicitly stating the physical units of inputs and outputs.
5.  **Linting & Formatting**:
    * Run `ruff check` to lint code. Use `ruff check --fix` for auto-fixable issues.
    * Run `ruff format` to format code.

## 🤖 Agent Operating Procedures

### 1. Parsimonious Edits
*   **Surgical Changes**: Avoid rewriting entire files to change a single function. Use targeted edits to preserve original comments and formatting.
*   **Avoid Refactor-Creep**: Do not refactor unrelated code unless explicitly requested. If you see a better way to do something elsewhere, note it in the chat rather than changing it silently.
*   **Modularity**: Respect the established modularity of the project. Suggest only code edits which keep the code modular and maintainable.

### 2. Human-in-the-Loop (HITL)
*   **Checkpointing**: Before performing destructive operations (e.g., deleting files, large-scale migrations), propose the plan and wait for user approval.
*   **Ambiguity Protocol**: If a scientific requirement is ambiguous (e.g., "should this be a mean or a median?"), **stop and ask**. Do not guess the science.
*   **Verification**: After a complex change, provide a brief "Verification Summary":
    *   *What was changed.*
    *   *Which test proves it works.*
    *   *Any new dependencies added.*
*   **Never execute without permission**: Only execute shell or python commands if prompted by the user to do so. 
*   **Never execute run anything on the commandline without permission**: Only run shell or python commands with permission by user!

## 📂 Directory Structure
*   `/src`: Core logic.
*   `/tests`: Pytest suite.
*   `.github/workflows`: CI configurations.
*   `pyproject.toml`: Dependency and build definitions.
