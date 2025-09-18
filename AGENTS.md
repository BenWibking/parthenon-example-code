# Repository Guidelines

## Project Structure & Module Organization
- `src/`: C++17 sources for the Euler sparse example (`main.cpp`, `euler_*`, `parthenon_app_inputs.cpp`, `CMakeLists.txt`).
- `extern/parthenon/`: Git submodule providing the Parthenon framework (library + infrastructure).
- `parthinput.euler_sparse`: Example runtime input file; pass via `-i` when running.
- Preferred integration: include this repo as a subdirectory of a CMake superproject alongside `extern/parthenon` and `src/`.

## Build, Test, and Development Commands
- Initialize submodules:
  - `git submodule update --init --recursive`
- Quick start (build Parthenon library locally):
  - `cmake -S extern/parthenon -B build`
  - `cmake --build build -j`
- Build this app in a superproject: add `src/CMakeLists.txt` and link `Parthenon::parthenon` to produce `euler_sparse-example`.
- Run (from your binary directory):
  - `./euler_sparse-example -i /path/to/parthinput.euler_sparse`
- Common CMake tips: set Parthenon options at configure time (e.g., `-DPARTHENON_ENABLE_MPI=ON`, `-DPARTHENON_ENABLE_OPENMP=ON`, HDF5/Kokkos backends as needed). Document flags in PRs.

## Coding Style & Naming Conventions
- Language: C++17. Headers `*.hpp`, sources `*.cpp`. Indent with 2 or 4 spaces; no tabs.
- Filenames: `snake_case` (e.g., `euler_package.cpp`). Types/classes: `PascalCase` (e.g., `EulerDriver`). Functions/vars: `lower_snake_case`.
- Namespace: `euler_sparse_example`.
- Includes: use Parthenon-style includes (e.g., `parthenon/package.hpp`). Prefer `auto` only when the type is obvious; avoid one-letter names. Keep changes minimal and focused.

## Testing Guidelines
- No in-repo unit tests. Validate by running with `parthinput.euler_sparse` and inspecting diagnostics/output.
- If adding tests, mirror Parthenon’s GTest pattern in a `tests/` dir and wire via CMake; then run with `ctest -V` from the build tree.

## Commit & Pull Request Guidelines
- Commits: concise, imperative mood, scoped changes (e.g., `driver: fix dt estimate for 2D`). Reference issues when applicable.
- PRs: include a clear description, rationale, logs or output snippets, linked issues, and exact build/run steps. Note any Parthenon/compile flags used (MPI/OpenMP/HDF5/Kokkos).

## Security & Configuration Tips
- Keep the Parthenon submodule pinned; update only after compatibility is verified.
- Avoid committing generated binaries, large outputs, or local build artifacts.
- Capture configuration in your PR (compiler, CMake flags, backend selections) for reproducibility.

