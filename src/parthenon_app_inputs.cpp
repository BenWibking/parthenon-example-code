//========================================================================================
#include <memory>

#include <parthenon/package.hpp>

#include "euler_driver.hpp"
#include "euler_package.hpp"
#include "pack/make_pack_descriptor.hpp"

namespace euler_sparse_example {

void ProblemGenerator(parthenon::MeshBlock *pmb, parthenon::ParameterInput *pin) {
  auto cellbounds = pmb->cellbounds;
  parthenon::IndexRange ib = cellbounds.GetBoundsI(parthenon::IndexDomain::interior);
  parthenon::IndexRange jb = cellbounds.GetBoundsJ(parthenon::IndexDomain::interior);
  parthenon::IndexRange kb = cellbounds.GetBoundsK(parthenon::IndexDomain::interior);

  auto coords = pmb->coords;
  auto &data = pmb->meshblock_data.Get();
  auto pkg = pmb->packages.Get("euler_sparse");

  const parthenon::Real gamma = pkg->Param<parthenon::Real>("gamma");
  const parthenon::Real rho0 = pin->GetOrAddReal("euler", "rho0", 1.0);
  const parthenon::Real p0 = pin->GetOrAddReal("euler", "p0", 1.0);
  const parthenon::Real drho = pin->GetOrAddReal("euler", "drho", 0.0);
  const parthenon::Real dp = pin->GetOrAddReal("euler", "dp", 0.0);
  const parthenon::Real r0 = pin->GetOrAddReal("euler", "blob_radius", 0.25);
  const parthenon::Real x0 = pin->GetOrAddReal("euler", "blob_x0", 0.0);
  const parthenon::Real y0 = pin->GetOrAddReal("euler", "blob_y0", 0.0);

  // Allocate sparse ID 0 in each separate pool (rho, mom, E)
  pmb->AllocSparseID("rho", 0);
  pmb->AllocSparseID("mom", 0);
  pmb->AllocSparseID("E", 0);

  // Initialize using typed SparsePack access for clarity and safety
  using euler_sparse_example::U::rho;
  using euler_sparse_example::U::mom;
  using euler_sparse_example::U::E;
  auto desc = parthenon::MakePackDescriptor<rho, mom, E>(data.get());
  auto pack = desc.GetPack(data.get());

  pmb->par_for(
      PARTHENON_AUTO_LABEL, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        const parthenon::Real x = coords.Xc<1>(i) - x0;
        const parthenon::Real y = coords.Xc<2>(j) - y0;
        const parthenon::Real z = coords.Xc<3>(k);
        const parthenon::Real r2 = x * x + y * y + z * z;
        const bool in_blob = (r2 < r0 * r0);
        const parthenon::Real rho_ic = rho0 + (in_blob ? drho : 0.0);
        const parthenon::Real p_ic = p0 + (in_blob ? dp : 0.0);
        const parthenon::Real eint = p_ic / (gamma - 1.0);

        // block index 0 for MeshBlockData packs
        pack(0, rho(), k, j, i) = rho_ic;
        pack(0, E(),   k, j, i) = eint;
        if (pack.Contains(0, mom())) {
          pack(0, mom(0), k, j, i) = 0.0;
          pack(0, mom(1), k, j, i) = 0.0;
          pack(0, mom(2), k, j, i) = 0.0;
        }
      });
}

parthenon::Packages_t ProcessPackages(std::unique_ptr<parthenon::ParameterInput> &pin) {
  parthenon::Packages_t packages;
  auto pkg = euler_sparse_example::Initialize(pin.get());
  packages.Add(pkg);
  auto app = std::make_shared<parthenon::StateDescriptor>("euler_app");
  packages.Add(app);
  return packages;
}

} // namespace euler_sparse_example
