//========================================================================================
#include <memory>

#include "parthenon/driver.hpp"
#include "euler_driver.hpp"
#include "euler_package.hpp"

namespace euler_sparse_example {

parthenon::driver::prelude::TaskCollection EulerDriver::MakeTaskCollection(
    parthenon::driver::prelude::BlockList_t &blocks, const int stage) {
  parthenon::driver::prelude::TaskCollection tc;
  parthenon::driver::prelude::TaskID none(0);

  const parthenon::Real beta = integrator->beta[stage - 1];
  const parthenon::Real dt = integrator->dt;
  const auto &stage_name = integrator->stage_name;

  // NOTE: TF(...) tags the task with a readable name and passes the function pointer.

  // Post MPI receives early in order to maximize overlap between compute and communication
  //  * Motivation: posting Irecv early lets the NIC/MPI start transferring while ComputeFluxes runs,
  //    hiding network latency behind kernel work and shortening the critical path.
  auto partitions = pmesh->GetDefaultBlockPartitions();
  const int num_partitions = partitions.size();
  parthenon::driver::prelude::TaskRegion &start_region = tc.AddRegion(num_partitions);
  for (int i = 0; i < num_partitions; i++) {
    auto &tl = start_region[i];
    auto &mbase = pmesh->mesh_data.Add("base", partitions[i]);
    auto &mc0 = pmesh->mesh_data.Add(stage_name[stage - 1], mbase);
    const auto any = parthenon::BoundaryType::any;

    // Posts nonblocking receives for AMR flux-correction (“refluxing”) buffers on the given MeshData partition.
    tl.AddTask(none, TF(parthenon::StartReceiveFluxCorrections), mc0);

    // Prepares the receive side of boundary exchanges for a given boundary category and posts nonblocking receives (when not using coalesced comms).
    //  - Ensures the receive buffer cache for this MeshData partition and boundary type exists/initialized via InitializeBufferCache with ReceiveKey.
    //  - If coalesced comms are OFF, iterates all buffers and calls TryStartReceive() to post receives immediately.
    //  - If coalesced comms are ON, it only initializes caches; actual receive posting is coordinated later by the coalesced comms manager during ReceiveBoundBufs.
    tl.AddTask(none, TF(parthenon::StartReceiveBoundBufs<any>), mc0);
  }

  // Flux computation on MeshData partitions
  auto &region_flux = tc.AddRegion(num_partitions);
  for (int i = 0; i < num_partitions; i++) {
    auto &tl = region_flux[i];
    auto &mbase = pmesh->mesh_data.Add("base", partitions[i]);
    auto &mc0 = pmesh->mesh_data.Add(stage_name[stage - 1], mbase);
    // Compute the hydro flux using a Riemann solver
    tl.AddTask(
        none,
        TF(static_cast<parthenon::TaskStatus (*)(parthenon::MeshData<parthenon::Real> *)>(
            euler_sparse_example::ComputeFluxes)),
        mc0.get());
  }

  // MeshData-partition tasks: divergence, update, exchange
  parthenon::driver::prelude::TaskRegion &region_update = tc.AddRegion(num_partitions);
  for (int i = 0; i < num_partitions; i++) {
    // All operations act only on variables flagged Metadata::Independent (for Average/Update)
    // and Metadata::WithFluxes (for FluxDivergence).

    auto &tl = region_update[i];
    auto &mbase = pmesh->mesh_data.Add("base", partitions[i]);
    auto &mc0 = pmesh->mesh_data.Add(stage_name[stage - 1], mbase);
    auto &mc1 = pmesh->mesh_data.Add(stage_name[stage], mbase);
    auto &mdudt = pmesh->mesh_data.Add("dUdt", mbase);

    // set_flx: Adds flux-correction communication tasks for this partition.
    //    Starts immediately (no dependency). Uses AMR/multilevel info via pmesh->multilevel.
    auto set_flx = parthenon::AddFluxCorrectionTasks(none, tl, mc0, pmesh->multilevel);

    // flux_div: After flux corrections are scheduled, computes dudt = −∇·F for all independent cc vars with fluxes:
    //    calls parthenon::Update::FluxDivergence, reading mc0, writing mdudt.
    auto flux_div = tl.AddTask(
        set_flx,
        TF(parthenon::Update::FluxDivergence<parthenon::MeshData<parthenon::Real>>),
        mc0.get(), mdudt.get());
    
    // avg_data: Forms the convex combination of the current stage input and the base state, in place on mc0:
    //    mc0 = beta*mc0 + (1-beta)*mbase
    //    via Update::AverageIndependentData. This is the SSP/RK stage “averaging” step on independent vars.
    auto avg_data = tl.AddTask(
        flux_div,
        TF(parthenon::Update::AverageIndependentData<parthenon::MeshData<parthenon::Real>>),
        mc0.get(), mbase.get(), beta);
      
    // update: Advances to the next stage into mc1 using the RHS: mc1 = mc0 + (beta*dt)*mdudt
    //    via Update::UpdateIndependentData.
    // Resulting stage formula:
    //  mc1 = (1−beta)·U^n + beta·(U^(s−1) + dt·F(U^(s−1))),
    //  which matches the low-storage SSP RK stage update when decomposed into an average plus an RHS update.
    auto update = tl.AddTask(
        avg_data,
        TF(parthenon::Update::UpdateIndependentData<parthenon::MeshData<parthenon::Real>>),
        mc0.get(), mdudt.get(), beta * dt, mc1.get());

    // Adds the full boundary-halo exchange sequence to a TaskList and returns the last TaskID for chaining.
    parthenon::AddBoundaryExchangeTasks(update, tl, mc1, pmesh->multilevel);
  }

  // Tail: BCs and dt on MeshBlockData (per-block)
  parthenon::driver::prelude::TaskRegion &region_tail = tc.AddRegion(blocks.size());
  for (int i = 0; i < blocks.size(); i++) {
    auto &pmb = blocks[i];
    auto &tl = region_tail[i];
    auto &sc1 = pmb->meshblock_data.Get(stage_name[stage]);

    // Applies physical boundary conditions to ghost zones on a MeshBlock’s cell-centered fields
    auto set_bc = tl.AddTask(none, TF(parthenon::ApplyBoundaryConditions), sc1);

    // If this is the final RK stage, compute dt for the next cycle
    if (stage == integrator->nstages) {
      tl.AddTask(set_bc,
                 TF(parthenon::Update::EstimateTimestep<
                     parthenon::MeshBlockData<parthenon::Real>>),
                 sc1.get());
    }
  }
  return tc;
}

} // namespace euler_sparse_example
