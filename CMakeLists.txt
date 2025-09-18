cmake_minimum_required(VERSION 3.16)

# Standalone build for the Euler sparse example
project(euler_sparse_example LANGUAGES CXX)

# Require C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Build Parthenon from the bundled submodule, then this example.
# Parthenon defines the Parthenon::parthenon target and helper macros like lint_target.
add_subdirectory(extern/parthenon)
add_subdirectory(src)

