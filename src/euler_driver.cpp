//========================================================================================
#include <memory>

#include "parthenon/driver.hpp"
#include "euler_driver.hpp"
#include "euler_package.hpp"

using namespace parthenon::driver::prelude;

namespace euler_sparse_example {

TaskCollection EulerDriver::MakeTaskCollection(BlockList_t &blocks, const int stage) {
  using namespace parthenon::Update;
  TaskCollection tc;
  TaskID none(0);

  const Real beta = integrator->beta[stage - 1];
  const Real dt = integrator->dt;
  const auto &stage_name = integrator->stage_name;

  // Start receives early to ensure ghosts are valid for fluxes
  auto partitions = pmesh->GetDefaultBlockPartitions();
  const int num_partitions = partitions.size();
  TaskRegion &start_region = tc.AddRegion(num_partitions);
  for (int i = 0; i < num_partitions; i++) {
    auto &tl = start_region[i];
    auto &mbase = pmesh->mesh_data.Add("base", partitions[i]);
    auto &mc0 = pmesh->mesh_data.Add(stage_name[stage - 1], mbase);
    const auto any = parthenon::BoundaryType::any;
    tl.AddTask(none, TF(parthenon::StartReceiveFluxCorrections), mc0);
    tl.AddTask(none, TF(parthenon::StartReceiveBoundBufs<any>), mc0);
  }

  // Per-block flux computation
  auto &region_flux = tc.AddRegion(blocks.size());
  for (int i = 0; i < blocks.size(); i++) {
    auto &pmb = blocks[i];
    auto &tl = region_flux[i];
    auto &base = pmb->meshblock_data.Add("base", pmb);
    auto &sc0 = pmb->meshblock_data.Add(stage_name[stage - 1], base);
    tl.AddTask(none, TF(euler_sparse_example::ComputeFluxes), sc0);
  }

  // MeshData-partition tasks: divergence, update, exchange
  TaskRegion &region_update = tc.AddRegion(num_partitions);
  for (int i = 0; i < num_partitions; i++) {
    auto &tl = region_update[i];
    auto &mbase = pmesh->mesh_data.Add("base", partitions[i]);
    auto &mc0 = pmesh->mesh_data.Add(stage_name[stage - 1], mbase);
    auto &mc1 = pmesh->mesh_data.Add(stage_name[stage], mbase);
    auto &mdudt = pmesh->mesh_data.Add("dUdt", mbase);

    auto set_flx = parthenon::AddFluxCorrectionTasks(none, tl, mc0, pmesh->multilevel);
    auto flux_div = tl.AddTask(set_flx, TF(FluxDivergence<MeshData<Real>>), mc0.get(),
                               mdudt.get());
    auto avg_data = tl.AddTask(flux_div, TF(AverageIndependentData<MeshData<Real>>),
                               mc0.get(), mbase.get(), beta);
    auto update = tl.AddTask(avg_data, TF(UpdateIndependentData<MeshData<Real>>),
                             mc0.get(), mdudt.get(), beta * dt, mc1.get());

    parthenon::AddBoundaryExchangeTasks(update, tl, mc1, pmesh->multilevel);
  }

  // Tail: BCs and dt
  auto &region_tail = tc.AddRegion(blocks.size());
  for (int i = 0; i < blocks.size(); i++) {
    auto &pmb = blocks[i];
    auto &tl = region_tail[i];
    auto &sc1 = pmb->meshblock_data.Get(stage_name[stage]);
    auto set_bc = tl.AddTask(none, TF(parthenon::ApplyBoundaryConditions), sc1);
    if (stage == integrator->nstages) {
      tl.AddTask(set_bc, TF(EstimateTimestep<MeshBlockData<Real>>), sc1.get());
    }
  }
  return tc;
}

} // namespace euler_sparse_example

