# AGENTS.md

Micrograd-inspired scalar autograd engine, written as a learning exercise. There is no CI, linter, or formatter installed — verify changes by running the unit tests, or the entry scripts and checking their output.

## Commands

uv-managed project, Python 3.14 pinned via `.python-version` (pyproject declares `requires-python = ">=3.14"`). Use `uv run`, not bare `python`. Run everything from the repo root.

- `uv run python xor.py` — XOR benchmark, finishes in seconds.
- `uv run python moons.py` — two-moons training; a full 200-epoch run takes ~1–2 minutes. Pass `--no-gc` to disable the garbage collector (faster).
- `uv run python -m unittest` — run the unit tests in `tests/` (stdlib `unittest`, no extra dependencies, finishes in seconds).
- `uv run python -m unittest tests.test_engine` — one test module; `uv run python -m unittest -k relu` matches test names.

## Layout

- `smolgrad/engine.py` — `Value` autograd node (data, grad, `prev` set, per-op `_backward` closures; `backward()` walks a topological order).
- `smolgrad/nn.py` — `Module`/`Neuron`/`Layer`/`MLP` (hidden layers ReLU, final layer linear).
- `smolgrad/dataset/two_moons.py` — dataset with seeded train/test split (seed 42).
- `smolgrad/utils/visualizer.py` — `ascii_mlp()` debug rendering.
- Root `xor.py` / `moons.py` are the runnable entry scripts, not part of the package; `inspect_dataset.ipynb` is a root notebook.
- `tests/` — stdlib `unittest` suite covering `engine.py` and `nn.py`.
- `smolgrad` is not installed (no `[build-system]` in pyproject.toml); imports resolve because the scripts sit next to the package dir, so run everything from the repo root.

`smolgrad/__init__.py` re-exports only `Value`, `two_moons`, `ascii_mlp` (`__all__`); `MLP` must be imported from `smolgrad.nn` directly.

## Gotchas

- Submodules (`nn.py`, `two_moons.py`, `visualizer.py`) import `Value` via the package root (`from smolgrad import Value`). `engine` must stay the first import in `__init__.py` — reordering it fails with a circular-import `ImportError`.
- Runs are not reproducible: only the two-moons split is seeded. Dataset noise and MLP weight init are unseeded, so accuracy varies between runs — xor does not always reach 100%. A single non-converged run is not a bug.
- `Value` relies on identity hashing (`prev` is a `set`; `backward()` uses a `visited` set). Do not add value-based `__eq__` without also defining `__hash__`.
- pyproject.toml declares matplotlib, numpy, and torchvision, but the package and tests are pure stdlib — only `inspect_dataset.ipynb` uses any of them (numpy + matplotlib; torchvision is referenced nowhere).
- pyproject.toml sets `[tool.black]` with `line-length = 80`, but black is not installed (not in uv.lock). Match the existing ~80-column style by hand instead of reformatting.
