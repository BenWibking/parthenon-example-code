# Repository Guidelines

## Project Structure & Module Organization
The C++17 example lives in `src/` with `main.cpp`, `euler_*.*`, `parthenon_app_inputs.cpp`, and the local `CMakeLists.txt`. The Parthenon framework is tracked as the `extern/parthenon/` submodule; do not edit its sources directly. Runtime inputs such as `parthinput.euler_sparse` sit at the repo root. Typical superproject layouts add this repository and the `extern/parthenon` tree side by side so the top-level CMake can wire targets together.

## Build, Test, and Development Commands
Initialize dependencies via `git submodule update --init --recursive`. Build the Parthenon library: `cmake -S extern/parthenon -B build && cmake --build build -j`. Inside a superproject, add `src/CMakeLists.txt` and link `Parthenon::parthenon` to produce `euler_sparse-example`. Run local diagnostics from your binary directory with `./euler_sparse-example -i /path/to/parthinput.euler_sparse`. Tweak Parthenon flags at configure time (e.g., `-DPARTHENON_ENABLE_MPI=ON`, `-DPARTHENON_ENABLE_OPENMP=ON`) and record the exact options in reviews.

## Coding Style & Naming Conventions
Use C++17, 2–4 space indentation, and no tabs. Name files in `snake_case`, classes in `PascalCase`, and functions/variables in `lower_snake_case`. Keep everything inside the `euler_sparse_example` namespace. Prefer explicit types unless `auto` clarifies readability. Follow Parthenon include style (`parthenon/package.hpp`) and add concise comments only where logic is non-obvious.

## Testing Guidelines
There are no baked-in unit tests. Validate changes by running the solver with `parthinput.euler_sparse` and inspecting output fields and logs. When adding automated coverage, mirror Parthenon’s GTest setup in `tests/`, register suites in CMake, and execute with `ctest -V` from the build tree.

## Commit & Pull Request Guidelines
Write commits in imperative mood and scope them narrowly (example: `driver: fix dt estimate for 2D`). Reference issues when applicable. Pull requests should describe motivation, summarize changes, link relevant issues, and include build/run steps plus key output snippets or screenshots. Call out any required Parthenon options, MPI/OpenMP/HDF5/Kokkos backends, or compiler specifics.

## Security & Configuration Tips
Keep the Parthenon submodule pinned and update only after validating compatibility. Never commit generated binaries, large output dumps, or local build artifacts. Surface configuration and environment details in PRs so others can reproduce results quickly.
