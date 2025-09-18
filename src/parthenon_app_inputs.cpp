//========================================================================================
#include <cmath>
#include <memory>
#include <vector>

#include <parthenon/package.hpp>

#include "defs.hpp"
#include "globals.hpp"
#include "interface/variable.hpp"
#include "euler_driver.hpp"
#include "euler_package.hpp"

using namespace parthenon::package::prelude;
using namespace parthenon;

namespace euler_sparse_example {

void ProblemGenerator(MeshBlock *pmb, ParameterInput *pin) {
  auto cellbounds = pmb->cellbounds;
  IndexRange ib = cellbounds.GetBoundsI(IndexDomain::interior);
  IndexRange jb = cellbounds.GetBoundsJ(IndexDomain::interior);
  IndexRange kb = cellbounds.GetBoundsK(IndexDomain::interior);

  auto coords = pmb->coords;
  auto &data = pmb->meshblock_data.Get();
  auto pkg = pmb->packages.Get("euler_sparse");

  const Real gamma = pkg->Param<Real>("gamma");
  const Real rho0 = pin->GetOrAddReal("euler", "rho0", 1.0);
  const Real p0 = pin->GetOrAddReal("euler", "p0", 1.0);
  const Real drho = pin->GetOrAddReal("euler", "drho", 0.0);
  const Real dp = pin->GetOrAddReal("euler", "dp", 0.0);
  const Real r0 = pin->GetOrAddReal("euler", "blob_radius", 0.25);
  const Real x0 = pin->GetOrAddReal("euler", "blob_x0", 0.0);
  const Real y0 = pin->GetOrAddReal("euler", "blob_y0", 0.0);

  pmb->AllocSparseID("U", 0);
  pmb->AllocSparseID("U", 1);
  pmb->AllocSparseID("U", 2);

  auto v = data->PackVariables(std::vector<std::string>{"U"});

  pmb->par_for(
      PARTHENON_AUTO_LABEL, 0, v.GetDim(4) - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int n, const int k, const int j, const int i) {
        const Real x = coords.Xc<1>(i) - x0;
        const Real y = coords.Xc<2>(j) - y0;
        const Real z = coords.Xc<3>(k);
        const Real r2 = x * x + y * y + z * z;
        const bool in_blob = (r2 < r0 * r0);
        const Real rho = rho0 + (in_blob ? drho : 0.0);
        const Real p = p0 + (in_blob ? dp : 0.0);
        if (v(n).label() == "U::rho") {
          v(n, k, j, i) = rho;
        } else if (v(n).label() == "U::mom_x" || v(n).label() == "U::mom_y" ||
                   v(n).label() == "U::mom_z") {
          v(n, k, j, i) = 0.0;
        } else if (v(n).label() == "U::E") {
          const Real eint = p / (gamma - 1.0);
          v(n, k, j, i) = eint;
        }
      });
}

Packages_t ProcessPackages(std::unique_ptr<ParameterInput> &pin) {
  Packages_t packages;
  auto pkg = euler_sparse_example::Initialize(pin.get());
  packages.Add(pkg);
  auto app = std::make_shared<StateDescriptor>("euler_app");
  packages.Add(app);
  return packages;
}

} // namespace euler_sparse_example
