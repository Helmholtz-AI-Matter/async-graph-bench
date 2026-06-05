# Documentation

## Dependencies 

- gnu make
- python and the following modules
```bash
pip install sphinx myst-parser sphinx_external_toc sphinx-autodoc-typehints sphinx-rtd-theme
```

## Building the Docs locally

First, build the library, then issue the following command from the repo root:

```bash
sphinx-build -b html docs/source docs/build
```

## Generating API Docs

Modify `generate_api.py` to contain all nodes that are to be included in the api section, then run the script. This will replace already existing documentation markdown files under `api/`
