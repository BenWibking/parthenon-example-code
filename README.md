# Euler Sparse Example: How-To

This guide explains how this repository implements a minimal Parthenon-based hyperbolic PDE solver using sparse fields, typed SparsePacks, task-based time integration, and coalesced communication. It mirrors the current code under `src/` so you can map concepts directly to implementation.

## Build and Run

- Init submodules: `git submodule update --init --recursive`

- Standalone configure + build (top-level drives Parthenon + this example):
  - `cmake -S . -B build [-DPARTHENON_ENABLE_MPI=ON -DPARTHENON_ENABLE_OPENMP=ON ...]`
  - `cmake --build build -j`
- Pass Parthenon options at configure time (MPI/OpenMP/HDF5/Kokkos backend) as needed.
- Run from the build tree:
  - `build/src/euler_sparse-example -i parthinput.euler_sparse`

## Overview of Components

- Package: declares conserved variables as a sparse pool `U` and exposes runtime params and hooks. See `src/euler_package.cpp:Initialize`.
- Problem generator: allocates sparse IDs and initializes state via a typed SparsePack. See `src/parthenon_app_inputs.cpp:ProblemGenerator`.
- Fluxes: computes Rusanov fluxes in X1 and X2 using typed `MakePackDescriptor` with `WithFluxes`. See `src/euler_package.cpp:ComputeFluxes`.
- Driver: orchestrates receives, flux compute, divergence, update, and boundary exchanges across `MeshData` partitions. See `src/euler_driver.cpp:MakeTaskCollection`.
- Timestep: estimates `dt` from wavespeeds. See `src/euler_package.cpp:EstimateTimestepBlock`.

## Declaring Sparse Conserved Variables (Pool `U`)

We model density (`rho`), momentum (`mom[3]`), and total energy (`E`) as sparse components in a single pool `U` with fluxes and ghost fills enabled:

```c++
// src/euler_package.cpp:Initialize
Metadata m({Metadata::Cell, Metadata::Independent, Metadata::WithFluxes,
            Metadata::FillGhost, Metadata::Sparse});
SparsePool U("U", m);
U.Add(0, std::vector<int>{1}, std::vector<std::string>{"rho"});
U.Add(1, std::vector<int>{3}, Metadata::Vector,
      std::vector<std::string>{"mom_x", "mom_y", "mom_z"});
U.Add(2, std::vector<int>{1}, std::vector<std::string>{"E"});
pkg->AddSparsePool(U);
```

Typed tags for sparse pool access are defined in `src/euler_package.hpp` under `namespace U`. Important: sparse pool labels in Parthenon use the base name plus sparse ID, so `U_0`, `U_1`, `U_2` correspond to `rho`, `mom`, and `E` respectively. The tags encapsulate this naming.

Runtime parameters (gamma, cfl) are read from the input file and stored on the package.

## Allocating and Initializing State

The problem generator allocates the needed sparse IDs on each MeshBlock and initializes fields using a typed pack:

```c++
// src/parthenon_app_inputs.cpp:ProblemGenerator
pmb->AllocSparseID("U", 0);
pmb->AllocSparseID("U", 1);
pmb->AllocSparseID("U", 2);

using euler_sparse_example::U::rho;
using euler_sparse_example::U::mom;
using euler_sparse_example::U::E;
auto desc = parthenon::MakePackDescriptor<rho, mom, E>(data.get());
auto pack = desc.GetPack(data.get());

pmb->par_for(PARTHENON_AUTO_LABEL, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
             KOKKOS_LAMBDA(const int k, const int j, const int i) {
  // block index 0 for MeshBlockData packs
  pack(0, rho(), k, j, i) = ...;
  pack(0, E(),   k, j, i) = ...;
  if (pack.Contains(0, mom())) {
    pack(0, mom(0), k, j, i) = 0.0;
    pack(0, mom(1), k, j, i) = 0.0;
    pack(0, mom(2), k, j, i) = 0.0;
  }
});
```

## Flux Computation with Typed SparsePack

Fluxes are computed with a typed `MakePackDescriptor<rho, mom, E>(..., PDOpt::WithFluxes)` to ensure flux arrays are present. A simple Rusanov solver is implemented in X1 and, when 2D+, in X2:

```c++
// src/euler_package.cpp:ComputeFluxes (excerpt)
auto desc = parthenon::MakePackDescriptor<rho, mom, E>(
    rc.get(), std::vector<parthenon::MetadataFlag>{},
    std::set<parthenon::PDOpt>{parthenon::PDOpt::WithFluxes});
auto pack = desc.GetPack(rc.get());

pmb->par_for_outer(PARTHENON_AUTO_LABEL, scratch_size, scratch_level, kb.s, kb.e, jb.s, jb.e,
  KOKKOS_LAMBDA(parthenon::team_mbr_t member, const int k, const int j) {
    for (int i = ib.s; i <= ib.e + 1; ++i) {
      if (!(pack.Contains(0, rho()) && pack.Contains(0, mom()) && pack.Contains(0, E())))
        continue;
      // reconstruct L/R (here we use cell-centered values directly), compute smax
      // write fluxes via pack.flux(0, dir, tag[,comp], k, j, i)
    }
  });
```

Helper conversions (conserved → primitive and sound speed) live alongside the flux routine.

## Multi-Stage Driver and Task Graph

`src/euler_driver.cpp:MakeTaskCollection` follows this ordering per stage:

1) Start receives early on `MeshData` partitions
   - `StartReceiveFluxCorrections`, `StartReceiveBoundBufs<any>`
2) Per-block flux computation
   - Build `base` and stage views, call `ComputeFluxes`
3) Partitioned divergence, averaging, update, and boundary exchange
   - `AddFluxCorrectionTasks`, `FluxDivergence`, `AverageIndependentData`, `UpdateIndependentData`
   - `AddBoundaryExchangeTasks(update, ...)`
4) Tail tasks per block
   - `ApplyBoundaryConditions`, and on final stage `EstimateTimestep`

AMR tagging is not implemented in this example but can be added in the tail region when adaptive meshes are enabled.

## Timestep Estimation

`src/euler_package.cpp:EstimateTimestepBlock` computes a stable `dt` based on the maximum characteristic speed per direction using the package `gamma` and the mesh metrics. The driver scales this by `cfl` from the input file.

## Coalesced Buffer Communication

Coalesced comms aggregate neighbor messages into fewer, larger messages without API changes. Enable at runtime in the input file:

```
[parthenon/mesh]
do_coalesced_comms = true
```

With this set, boundary exchanges initiated via `StartReceiveBoundBufs`, `AddBoundaryExchangeTasks`, and flux corrections use the coalesced paths. Sparse variables are supported (including null messages when a component is not allocated).

## Input File

`parthinput.euler_sparse` configures the mesh, time integrator, and Euler app parameters. Relevant keys under `[euler]` include `gamma`, `cfl`, `rho0`, `p0`, `drho`, `dp`, and blob geometry for the initial condition. Output selects the sparse pool via `variables = U`.

## Physics/ICs

- Setup: a stationary pressure/density “blob” in a periodic box. Inside a radius `blob_radius` centered at `(blob_x0, blob_y0, 0)`, density and pressure are raised by `drho` and `dp` relative to `rho0` and `p0`.
- Momentum: initialized to zero everywhere (`mom = 0`).
- Energy: total energy is set to internal energy since velocity is zero, `E = p/(gamma-1)`. If you introduce velocity, remember to add kinetic energy: `E = p/(gamma-1) + 0.5*rho*(u^2+v^2+w^2)`.
- Dimensionality: example input uses 2D (`nx3=1`), periodic BCs on all faces, and coalesced comms enabled.
- Tweaking the blob: increase `drho` and/or `dp` to strengthen the perturbation; shift `(blob_x0, blob_y0)` to move the blob; adjust `tlim`/`dt` to control runtime.

## Where to Look

- Entry point: `src/main.cpp`
- Driver and tasks: `src/euler_driver.hpp`, `src/euler_driver.cpp`
- Package (vars, fluxes, dt): `src/euler_package.hpp`, `src/euler_package.cpp`
- Problem generator + package registration: `src/parthenon_app_inputs.cpp`
- Parthenon internals referenced by this example (from submodule):
  - Packs: `extern/parthenon/src/pack/`
  - Boundary comms (coalesced): `extern/parthenon/src/bvals/comms/`
  - Update utilities: `extern/parthenon/src/time_integration/` and `extern/parthenon/src/outputs/`

## Tips and Pitfalls

- Allocation: allocate sparse IDs on the host (`AllocSparseID`) and guard device work with `pack.Contains(...)`.
- Metadata: include `WithFluxes` for fields updated by flux divergence and `FillGhost` for halo exchange; use `Metadata::Vector` for multi-component momentum.
- Packs: prefer typed tags (`MakePackDescriptor<...>`) for safety and performance; request `PDOpt::WithFluxes` when you need flux arrays.
- Task ordering: start receives before compute to maximize overlap; operate on `MeshData` partitions for better communication/computation overlap.

With these pieces, this example provides a compact, consistent template for sparse hyperbolic solvers that benefit from Parthenon’s tasking, typed SparsePacks, and coalesced communication.
