# Euler Sparse Example: How-To

This guide explains how this repository implements a minimal Parthenon-based hyperbolic PDE solver using sparse fields, typed SparsePacks, task-based time integration, and coalesced communication. It mirrors the current code under `src/` so you can map concepts directly to implementation.

## Build and Run

- Init submodules: `git submodule update --init --recursive`

- Standalone configure + build (top-level drives Parthenon + this example):
  - `cmake -S . -B build [-DPARTHENON_ENABLE_MPI=ON -DPARTHENON_ENABLE_OPENMP=ON ...]`
  - `cmake --build build -j`
- Pass Parthenon options at configure time (MPI/OpenMP/HDF5/Kokkos backend) as needed.
- Requirement: this example uses PLM reconstruction and requires at least two ghost zones. Ensure `parthenon/mesh.nghost >= 2` (the Parthenon default is 2). You can set it in the input file under `[parthenon/mesh]` if needed.
- Run from the build tree:
  - `build/src/euler_sparse-example -i parthinput.euler_sparse`

## Overview of Components

- Package: declares conserved variables using one SparsePool per state variable: `rho`, `mom`, and `E`. The package also exposes runtime params and hooks. See `src/euler_package.cpp:Initialize`.
- Problem generator: allocates sparse IDs and initializes state via a typed SparsePack. See `src/parthenon_app_inputs.cpp:ProblemGenerator`.
- Fluxes: reconstructs PLM (MC limiter) interface states and computes Rusanov (HLL) fluxes in X1 and X2 using typed `MakePackDescriptor` with `WithFluxes`. The driver invokes `ComputeFluxesPLM_MC`. See `src/euler_package.cpp:ComputeFluxesPLM_MC`.
- Driver: orchestrates receives, flux compute, divergence, update, and boundary exchanges across `MeshData` partitions. See `src/euler_driver.cpp:MakeTaskCollection`.
- Timestep: estimates `dt` from wavespeeds. See `src/euler_package.cpp:EstimateTimestepBlock`.

## Declaring Sparse Conserved Variables (one pool per state var)

We model density (`rho`), momentum (`mom[3]`), and total energy (`E`) using separate SparsePools — one per state variable — with fluxes and ghost fills enabled. For a single material, we allocate sparse ID 0 in each pool, yielding variable labels `rho_0`, `mom_0`, and `E_0`:

```c++
// src/euler_package.cpp:Initialize
Metadata m({Metadata::Cell, Metadata::Independent, Metadata::WithFluxes,
            Metadata::FillGhost, Metadata::Sparse});

// Density pool
{
  SparsePool rho_pool("rho", m);
  rho_pool.Add(0, std::vector<int>{1}, std::vector<std::string>{"rho"});
  pkg->AddSparsePool(rho_pool);
}

// Momentum (vector<3>) pool
{
  SparsePool mom_pool("mom", m);
  mom_pool.Add(0, std::vector<int>{3}, Metadata::Vector,
               std::vector<std::string>{"mom_x", "mom_y", "mom_z"});
  pkg->AddSparsePool(mom_pool);
}

// Total energy pool
{
  SparsePool E_pool("E", m);
  E_pool.Add(0, std::vector<int>{1}, std::vector<std::string>{"E"});
  pkg->AddSparsePool(E_pool);
}
```

Typed tags for SparsePack access are defined in `src/euler_package.hpp` under `namespace U`. Important: sparse variable labels in Parthenon use the base name plus sparse ID, so `rho_0`, `mom_0`, and `E_0` correspond to the density, momentum, and energy variables in this example. The tags encapsulate this naming.

Runtime parameters (gamma, cfl) are read from the input file and stored on the package.

## Allocating and Initializing State

The problem generator allocates the needed sparse IDs (ID 0 in each pool) on each MeshBlock and initializes fields using a typed pack:

```c++
// src/parthenon_app_inputs.cpp:ProblemGenerator
pmb->AllocSparseID("rho", 0);
pmb->AllocSparseID("mom", 0);
pmb->AllocSparseID("E", 0);

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

Fluxes are computed over all MeshBlocks at once using a typed `MakePackDescriptor<rho, mom, E>(..., PDOpt::WithFluxes)`, then iterating with a block-aware outer loop and an inner i-loop. Piecewise linear (PLM) reconstruction with a monotonized central (MC) slope limiter forms left/right states; a Rusanov (HLL) solver is used in X1 and, when 2D+, in X2.

```c++
// src/euler_package.cpp:ComputeFluxesPLM_MC (excerpt)
auto desc = parthenon::MakePackDescriptor<rho, mom, E>(
    md, std::vector<parthenon::MetadataFlag>{},
    std::set<parthenon::PDOpt>{parthenon::PDOpt::WithFluxes});
auto pack = desc.GetPack(md);

const int nblocks = md->NumBlocks();

// X1 fluxes: outer loops over block b, k, j; inner over faces i in [ib.s, ib.e+1]
parthenon::par_for_outer(
    DEFAULT_OUTER_LOOP_PATTERN, PARTHENON_AUTO_LABEL, DevExecSpace(), scratch_size,
    scratch_level, 0, nblocks - 1, kb.s, kb.e, jb.s, jb.e,
    KOKKOS_LAMBDA(parthenon::team_mbr_t member, const int b, const int k, const int j) {
      if (!(pack.Contains(b, rho()) && pack.Contains(b, mom()) && pack.Contains(b, E())))
        return;

      // Take raw pointers to contiguous i-slices for speed
      Real *rho_bkj = &pack(b, rho(),   k, j, 0);
      Real *mx_bkj  = &pack(b, mom(0), k, j, 0);
      Real *my_bkj  = &pack(b, mom(1), k, j, 0);
      Real *mz_bkj  = &pack(b, mom(2), k, j, 0);
      Real *E_bkj   = &pack(b, E(),     k, j, 0);

      Real *Fx_rho = &pack.flux(b, 1, rho(),   k, j, 0);
      Real *Fx_mx  = &pack.flux(b, 1, mom(0),  k, j, 0);
      Real *Fx_my  = &pack.flux(b, 1, mom(1),  k, j, 0);
      Real *Fx_mz  = &pack.flux(b, 1, mom(2),  k, j, 0);
      Real *Fx_E   = &pack.flux(b, 1, E(),     k, j, 0);

      parthenon::par_for_inner(member, ib.s, ib.e + 1, [&](const int i_face) {
        const int iL = i_face - 1, iR = i_face;
        // form MC-limited PLM slopes in x for {rho,mom,E} at cell centers
        // reconstruct left/right interface states at i_face from (iL,iR)
        // compute HLL flux using reconstructed states and write Fx_*[i_face]
      });
    });

// X2 fluxes (if ndim >= 2): outer j over faces [jb.s, jb.e+1], inner i over [ib.s, ib.e]
if (pm->ndim >= 2) {
  parthenon::par_for_outer(
      DEFAULT_OUTER_LOOP_PATTERN, PARTHENON_AUTO_LABEL, DevExecSpace(), scratch_size,
      scratch_level, 0, nblocks - 1, kb.s, kb.e, jb.s, jb.e + 1,
      KOKKOS_LAMBDA(parthenon::team_mbr_t member, const int b, const int k, const int j_face) {
        if (!(pack.Contains(b, rho()) && pack.Contains(b, mom()) && pack.Contains(b, E())))
          return;

        const int jL = j_face - 1, jR = j_face;
        // form MC-limited PLM slopes in y and reconstruct at j_face
        parthenon::par_for_inner(member, ib.s, ib.e, [&](const int i) {
          // compute and store Fy_*[i]
        });
      });
}
```

Key features in the flux kernel:
- MeshData-based pack and block-aware Contains checks (`pack.Contains(b, ...)`).
- `par_for_outer` over `(b, k, j)` for X1 and `(b, k, j_face)` for X2.
- PLM reconstruction with MC limiter to form interface states before the HLL solve.
- Use pointer slices to contiguous i-lines and `par_for_inner` for face loops.

Helper conversions (conserved → primitive and sound speed) remain alongside the flux routine.

## Multi-Stage Driver and Task Graph

`src/euler_driver.cpp:MakeTaskCollection` follows this ordering per stage:

- Note: `ComputeFluxes` operates on a `MeshData` partition covering multiple blocks. The driver builds per-stage `MeshData` views and passes them into `ComputeFluxes`, which then iterates over blocks internally using the block-aware `par_for_outer` loops described above. This improves overlap with communication and keeps flux evaluation contiguous in memory along i-lines.

Call site in the driver:

```c++
// src/euler_driver.cpp: Flux computation task (MeshData partition)
auto &region_flux = tc.AddRegion(num_partitions);
for (int i = 0; i < num_partitions; i++) {
  auto &tl = region_flux[i];
  auto &mbase = pmesh->mesh_data.Add("base", partitions[i]);
  auto &mc0 = pmesh->mesh_data.Add(stage_name[stage - 1], mbase);
  // Reconstruct (PLM+MC) and compute hydro fluxes using a Riemann solver
  tl.AddTask(
      none,
      TF(static_cast<parthenon::TaskStatus (*)(parthenon::MeshData<Real> *)>(
          euler_sparse_example::ComputeFluxesPLM_MC)),
      mc0.get());
}
```

Exact call site: `src/euler_driver.cpp:50` (subject to drift with edits).

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

Task graph: The figure below illustrates the per-stage task dependencies constructed in `MakeTaskCollection`. Each node is a task and arrows denote dependencies and execution order across `MeshData` partitions. The image was generated from `task_graph.dot`.

![Task graph for a single RK stage](task_graph.png)

## Timestep Estimation

`src/euler_package.cpp:EstimateTimestepBlock` computes a stable `dt` based on the maximum characteristic speed per direction using the package `gamma` and the mesh metrics. The driver scales this by `cfl` from the input file.

Note: PLM+MC reconstruction requires at least two ghost zones; the package enforces `parthenon/mesh.nghost >= 2` at runtime.

## Coalesced Buffer Communication

Coalesced comms aggregate neighbor messages into fewer, larger messages without API changes. Enable at runtime in the input file:

```
[parthenon/mesh]
do_coalesced_comms = true
```

With this set, boundary exchanges initiated via `StartReceiveBoundBufs`, `AddBoundaryExchangeTasks`, and flux corrections use the coalesced paths. Sparse variables are supported (including null messages when a component is not allocated).

## Input File

`parthinput.euler_sparse` configures the mesh, time integrator, and Euler app parameters. Relevant keys under `[euler]` include `gamma`, `cfl`, `rho0`, `p0`, `drho`, `dp`, and blob geometry for the initial condition. For outputs, select variables explicitly, e.g. `variables = rho, mom, E` (or the specific sparse labels like `rho_0`, `mom_0`, `E_0`).

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

## Parthenon Objects Used (API Links)

Core runtime
- [ParthenonManager](https://github.com/parthenon-hpc-lab/parthenon/blob/df0cceb9b5fceb643a5869d231afa44b1613db70/src/parthenon_manager.hpp#L40)
- [ParthenonStatus](https://github.com/parthenon-hpc-lab/parthenon/blob/df0cceb9b5fceb643a5869d231afa44b1613db70/src/parthenon_manager.hpp#L38)
- [DriverStatus](https://github.com/parthenon-hpc-lab/parthenon/blob/df0cceb9b5fceb643a5869d231afa44b1613db70/src/driver/driver.hpp#L36)
- [Real](https://github.com/parthenon-hpc-lab/parthenon/blob/df0cceb9b5fceb643a5869d231afa44b1613db70/src/basic_types.hpp#L36)

Inputs, mesh, and data
- [ParameterInput](https://github.com/parthenon-hpc-lab/parthenon/blob/df0cceb9b5fceb643a5869d231afa44b1613db70/src/parameter_input.hpp#L191)
- [ApplicationInput](https://github.com/parthenon-hpc-lab/parthenon/blob/df0cceb9b5fceb643a5869d231afa44b1613db70/src/application_input.hpp#L37)
- [Mesh](https://github.com/parthenon-hpc-lab/parthenon/blob/df0cceb9b5fceb643a5869d231afa44b1613db70/src/mesh/mesh.hpp#L79)
- [MeshBlock](https://github.com/parthenon-hpc-lab/parthenon/blob/df0cceb9b5fceb643a5869d231afa44b1613db70/src/mesh/meshblock.hpp#L72)
- [MeshData<Real>](https://github.com/parthenon-hpc-lab/parthenon/blob/df0cceb9b5fceb643a5869d231afa44b1613db70/src/interface/mesh_data.hpp#L193)
- [MeshBlockData<Real>](https://github.com/parthenon-hpc-lab/parthenon/blob/df0cceb9b5fceb643a5869d231afa44b1613db70/src/interface/meshblock_data.hpp#L56)
- [Packages_t](https://github.com/parthenon-hpc-lab/parthenon/blob/df0cceb9b5fceb643a5869d231afa44b1613db70/src/interface/packages.hpp#L25)
- [IndexRange](https://github.com/parthenon-hpc-lab/parthenon/blob/df0cceb9b5fceb643a5869d231afa44b1613db70/src/basic_types.hpp#L42)
- [IndexDomain::interior](https://github.com/parthenon-hpc-lab/parthenon/blob/df0cceb9b5fceb643a5869d231afa44b1613db70/src/mesh/domain.hpp#L69)
- [X1DIR, X2DIR, X3DIR](https://github.com/parthenon-hpc-lab/parthenon/blob/df0cceb9b5fceb643a5869d231afa44b1613db70/src/basic_types.hpp#L60)

Driver and tasking
- [driver::prelude::MultiStageDriver](https://github.com/parthenon-hpc-lab/parthenon/blob/df0cceb9b5fceb643a5869d231afa44b1613db70/src/driver/multistage.hpp#L58)
- [driver::prelude::TaskCollection](https://github.com/parthenon-hpc-lab/parthenon/blob/df0cceb9b5fceb643a5869d231afa44b1613db70/src/tasks/tasks.hpp#L488)
- [driver::prelude::TaskRegion](https://github.com/parthenon-hpc-lab/parthenon/blob/df0cceb9b5fceb643a5869d231afa44b1613db70/src/tasks/tasks.hpp#L457)
- [driver::prelude::TaskID](https://github.com/parthenon-hpc-lab/parthenon/blob/df0cceb9b5fceb643a5869d231afa44b1613db70/src/tasks/tasks.hpp#L75)
- [driver::prelude::BlockList_t](https://github.com/parthenon-hpc-lab/parthenon/blob/df0cceb9b5fceb643a5869d231afa44b1613db70/src/mesh/meshblock.hpp#L476)

Update and communication tasks
- [Update::FluxDivergence](https://github.com/parthenon-hpc-lab/parthenon/blob/df0cceb9b5fceb643a5869d231afa44b1613db70/src/interface/update.hpp)
- [Update::AverageIndependentData](https://github.com/parthenon-hpc-lab/parthenon/blob/df0cceb9b5fceb643a5869d231afa44b1613db70/src/interface/update.hpp)
- [Update::UpdateIndependentData](https://github.com/parthenon-hpc-lab/parthenon/blob/df0cceb9b5fceb643a5869d231afa44b1613db70/src/interface/update.hpp)
- [Update::EstimateTimestep](https://github.com/parthenon-hpc-lab/parthenon/blob/df0cceb9b5fceb643a5869d231afa44b1613db70/src/interface/update.hpp)
- [AddFluxCorrectionTasks](https://github.com/parthenon-hpc-lab/parthenon/blob/df0cceb9b5fceb643a5869d231afa44b1613db70/src/bvals/comms/bvals_in_one.hpp)
- [AddBoundaryExchangeTasks](https://github.com/parthenon-hpc-lab/parthenon/blob/df0cceb9b5fceb643a5869d231afa44b1613db70/src/bvals/comms/bvals_in_one.hpp)
- [ApplyBoundaryConditions](https://github.com/parthenon-hpc-lab/parthenon/blob/df0cceb9b5fceb643a5869d231afa44b1613db70/src/bvals/boundary_conditions.hpp)
- [StartReceiveFluxCorrections](https://github.com/parthenon-hpc-lab/parthenon/blob/df0cceb9b5fceb643a5869d231afa44b1613db70/src/bvals/comms/bvals_in_one.hpp)
- [StartReceiveBoundBufs](https://github.com/parthenon-hpc-lab/parthenon/blob/df0cceb9b5fceb643a5869d231afa44b1613db70/src/bvals/comms/bvals_in_one.hpp)
- [BoundaryType::any](https://github.com/parthenon-hpc-lab/parthenon/blob/df0cceb9b5fceb643a5869d231afa44b1613db70/src/basic_types.hpp)
- [TaskStatus](https://github.com/parthenon-hpc-lab/parthenon/blob/df0cceb9b5fceb643a5869d231afa44b1613db70/src/basic_types.hpp)

Package system and metadata
- [StateDescriptor](https://github.com/parthenon-hpc-lab/parthenon/blob/df0cceb9b5fceb643a5869d231afa44b1613db70/src/interface/state_descriptor.hpp)
- [Packages_t](https://github.com/parthenon-hpc-lab/parthenon/blob/df0cceb9b5fceb643a5869d231afa44b1613db70/src/interface/packages.hpp)
- [SparsePool](https://github.com/parthenon-hpc-lab/parthenon/blob/df0cceb9b5fceb643a5869d231afa44b1613db70/src/interface/sparse_pool.hpp)
- [Metadata](https://github.com/parthenon-hpc-lab/parthenon/blob/df0cceb9b5fceb643a5869d231afa44b1613db70/src/interface/metadata.hpp) (flags like Cell, Independent, WithFluxes, FillGhost, Sparse, Vector)
- [PDOpt::WithFluxes](https://github.com/parthenon-hpc-lab/parthenon/blob/df0cceb9b5fceb643a5869d231afa44b1613db70/src/pack/pack_descriptor.hpp)

Variable tagging and packs
- [variable_names::base_t<...>](https://github.com/parthenon-hpc-lab/parthenon/blob/df0cceb9b5fceb643a5869d231afa44b1613db70/src/pack/pack_utils.hpp)
- [MakePackDescriptor](https://github.com/parthenon-hpc-lab/parthenon/blob/df0cceb9b5fceb643a5869d231afa44b1613db70/src/pack/make_pack_descriptor.hpp)

Execution and coordinates
- [par_for_outer / par_for_inner](https://github.com/parthenon-hpc-lab/parthenon/blob/df0cceb9b5fceb643a5869d231afa44b1613db70/src/kokkos_abstraction.hpp)
- [team_mbr_t, DevExecSpace](https://github.com/parthenon-hpc-lab/parthenon/blob/df0cceb9b5fceb643a5869d231afa44b1613db70/src/kokkos_types.hpp)
- [Coordinates_t](https://github.com/parthenon-hpc-lab/parthenon/blob/df0cceb9b5fceb643a5869d231afa44b1613db70/src/coordinates/coordinates.hpp)
- [coords.Xc<...>, coords.Dxc<...>](https://github.com/parthenon-hpc-lab/parthenon/blob/df0cceb9b5fceb643a5869d231afa44b1613db70/src/coordinates/uniform_coordinates.hpp)

## Tips and Pitfalls

- Allocation: allocate sparse IDs on the host (`AllocSparseID`) and guard device work with `pack.Contains(...)`. Use one SparsePool per state variable; add additional sparse IDs per pool if modeling multiple materials.
- Metadata: include `WithFluxes` for fields updated by flux divergence and `FillGhost` for halo exchange; use `Metadata::Vector` for multi-component momentum.
- Packs: prefer typed tags (`MakePackDescriptor<...>`) for safety and performance; request `PDOpt::WithFluxes` when you need flux arrays.
- Task ordering: start receives before compute to maximize overlap; operate on `MeshData` partitions for better communication/computation overlap.

With these pieces, this example provides a compact, consistent template for sparse hyperbolic solvers that benefit from Parthenon’s tasking, typed SparsePacks, and coalesced communication.

## TaskRegion Sync: Rules of Thumb

- Regions run sequentially: Parthenon executes TaskRegions in order; Region N completes before Region N+1 starts.
- Not an implicit device fence: finishing a TaskRegion does not call `Kokkos::fence()`. Fence explicitly before host access or MPI on device-computed data.
- Not an implicit MPI barrier: there is no automatic MPI barrier between regions. Use `global_sync` tasks (and MPI as needed) for cross-rank coordination.
- Qualifiers wire dependencies, not fences: `local_sync`/`global_sync`/`once_per_region` create cross-list/rank dependencies within a Region. If a task requires device-host sync, call `Kokkos::fence()` inside that task.
- Practical pattern: add a small sync task at the end of a Region to fence and/or coordinate MPI reductions, and depend subsequent Regions on it.
