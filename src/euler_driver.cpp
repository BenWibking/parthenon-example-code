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

  // Start receives early to ensure ghosts are valid for fluxes
  auto partitions = pmesh->GetDefaultBlockPartitions();
  const int num_partitions = partitions.size();
  parthenon::driver::prelude::TaskRegion &start_region = tc.AddRegion(num_partitions);
  for (int i = 0; i < num_partitions; i++) {
    auto &tl = start_region[i];
    auto &mbase = pmesh->mesh_data.Add("base", partitions[i]);
    auto &mc0 = pmesh->mesh_data.Add(stage_name[stage - 1], mbase);
    const auto any = parthenon::BoundaryType::any;
    tl.AddTask(none, TF(parthenon::StartReceiveFluxCorrections), mc0);
    tl.AddTask(none, TF(parthenon::StartReceiveBoundBufs<any>), mc0);
  }

  // Flux computation on MeshData partitions (explicit b index)
  auto &region_flux = tc.AddRegion(num_partitions);
  for (int i = 0; i < num_partitions; i++) {
    auto &tl = region_flux[i];
    auto &mbase = pmesh->mesh_data.Add("base", partitions[i]);
    auto &mc0 = pmesh->mesh_data.Add(stage_name[stage - 1], mbase);
    tl.AddTask(
        none,
        TF(static_cast<parthenon::TaskStatus (*)(parthenon::MeshData<parthenon::Real> *)>(
            euler_sparse_example::ComputeFluxes)),
        mc0.get());
  }

  // MeshData-partition tasks: divergence, update, exchange
  parthenon::driver::prelude::TaskRegion &region_update = tc.AddRegion(num_partitions);
  for (int i = 0; i < num_partitions; i++) {
    auto &tl = region_update[i];
    auto &mbase = pmesh->mesh_data.Add("base", partitions[i]);
    auto &mc0 = pmesh->mesh_data.Add(stage_name[stage - 1], mbase);
    auto &mc1 = pmesh->mesh_data.Add(stage_name[stage], mbase);
    auto &mdudt = pmesh->mesh_data.Add("dUdt", mbase);

    auto set_flx = parthenon::AddFluxCorrectionTasks(none, tl, mc0, pmesh->multilevel);
    auto flux_div = tl.AddTask(
        set_flx,
        TF(parthenon::Update::FluxDivergence<parthenon::MeshData<parthenon::Real>>),
        mc0.get(), mdudt.get());
    auto avg_data = tl.AddTask(
        flux_div,
        TF(parthenon::Update::AverageIndependentData<parthenon::MeshData<parthenon::Real>>),
        mc0.get(), mbase.get(), beta);
    auto update = tl.AddTask(
        avg_data,
        TF(parthenon::Update::UpdateIndependentData<parthenon::MeshData<parthenon::Real>>),
        mc0.get(), mdudt.get(), beta * dt, mc1.get());

    parthenon::AddBoundaryExchangeTasks(update, tl, mc1, pmesh->multilevel);
  }

  // Tail: BCs and dt on MeshBlockData (per-block)
  parthenon::driver::prelude::TaskRegion &region_tail = tc.AddRegion(blocks.size());
  for (int i = 0; i < blocks.size(); i++) {
    auto &pmb = blocks[i];
    auto &tl = region_tail[i];
    auto &sc1 = pmb->meshblock_data.Get(stage_name[stage]);
    auto set_bc = tl.AddTask(none, TF(parthenon::ApplyBoundaryConditions), sc1);
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
