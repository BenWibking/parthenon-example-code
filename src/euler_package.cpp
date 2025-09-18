//========================================================================================
// Minimal Euler (hyperbolic) package using sparse fields and SparsePack
//========================================================================================

#include <cmath>
#include <memory>
#include <string>
#include <vector>

#include <parthenon/package.hpp>

#include "interface/metadata.hpp"
#include "interface/sparse_pool.hpp"
#include "mesh/meshblock.hpp"
#include "pack/make_pack_descriptor.hpp"
#include "pack/pack_utils.hpp"

#include "euler_package.hpp"

using namespace parthenon::package::prelude;

namespace euler_sparse_example {

static inline KOKKOS_INLINE_FUNCTION Real max3(Real a, Real b, Real c) {
  return std::max(a, std::max(b, c));
}

std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin) {
  auto pkg = std::make_shared<StateDescriptor>("euler_sparse");

  const Real gamma = pin->GetOrAddReal("euler", "gamma", 1.4);
  const Real cfl = pin->GetOrAddReal("euler", "cfl", 0.45);
  pkg->AddParam("gamma", gamma);
  pkg->AddParam("cfl", cfl);

  Metadata m({Metadata::Cell, Metadata::Independent, Metadata::WithFluxes,
              Metadata::FillGhost, Metadata::Sparse});
  SparsePool U("U", m);
  U.Add(0, std::vector<int>{1}, std::vector<std::string>{"rho"});
  U.Add(1, std::vector<int>{3}, Metadata::Vector,
        std::vector<std::string>{"mom_x", "mom_y", "mom_z"});
  U.Add(2, std::vector<int>{1}, std::vector<std::string>{"E"});
  pkg->AddSparsePool(U);

  pkg->EstimateTimestepBlock = EstimateTimestepBlock;
  return pkg;
}

static KOKKOS_INLINE_FUNCTION void cons_to_prim(
    const Real gamma, const Real rho, const Real mN, const Real mT1, const Real mT2,
    const Real E, Real *u, Real *v, Real *w, Real *p, Real *a) {
  const Real inv_rho = 1.0 / rho;
  *u = mN * inv_rho;
  *v = mT1 * inv_rho;
  *w = mT2 * inv_rho;
  const Real ke = 0.5 * rho * ((*u) * (*u) + (*v) * (*v) + (*w) * (*w));
  *p = (gamma - 1.0) * (E - ke);
  *a = std::sqrt(std::max(Real(0.0), gamma * (*p) * inv_rho));
}

template <int DIR>
KOKKOS_INLINE_FUNCTION void hll_flux_dir(
    const Real gamma,
    const Real rhoL, const Real mxL, const Real myL, const Real mzL, const Real EL,
    const Real rhoR, const Real mxR, const Real myR, const Real mzR, const Real ER,
    Real *F_rho, Real *F_mx, Real *F_my, Real *F_mz, Real *F_E) {
  // Map momenta so mN is normal, mT1/mT2 tangential to the face normal
  Real mN_L, mT1_L, mT2_L;
  Real mN_R, mT1_R, mT2_R;
  if constexpr (DIR == X1DIR) {
    mN_L = mxL; mT1_L = myL; mT2_L = mzL;
    mN_R = mxR; mT1_R = myR; mT2_R = mzR;
  } else if constexpr (DIR == X2DIR) {
    mN_L = myL; mT1_L = mxL; mT2_L = mzL;
    mN_R = myR; mT1_R = mxR; mT2_R = mzR;
  } else { // X3DIR (not used here, but keep generic)
    mN_L = mzL; mT1_L = mxL; mT2_L = myL;
    mN_R = mzR; mT1_R = mxR; mT2_R = myR;
  }

  Real uL, vL, wL, pL, aL; // u = normal velocity
  Real uR, vR, wR, pR, aR;
  cons_to_prim(gamma, rhoL, mN_L, mT1_L, mT2_L, EL, &uL, &vL, &wL, &pL, &aL);
  cons_to_prim(gamma, rhoR, mN_R, mT1_R, mT2_R, ER, &uR, &vR, &wR, &pR, &aR);

  const Real smax = std::max(std::abs(uL) + aL, std::abs(uR) + aR);

  // Physical fluxes (left/right) in the normal direction
  const Real Fr_rho_L = rhoL * uL;
  const Real Fr_mN_L = mN_L * uL + pL;
  const Real Fr_mT1_L = mT1_L * uL;
  const Real Fr_mT2_L = mT2_L * uL;
  const Real Fr_E_L = (EL + pL) * uL;

  const Real Fr_rho_R = rhoR * uR;
  const Real Fr_mN_R = mN_R * uR + pR;
  const Real Fr_mT1_R = mT1_R * uR;
  const Real Fr_mT2_R = mT2_R * uR;
  const Real Fr_E_R = (ER + pR) * uR;

  // HLL (Rusanov) flux for each conservative component
  const Real F_rho_dir = 0.5 * (Fr_rho_L + Fr_rho_R) - 0.5 * smax * (rhoR - rhoL);
  const Real F_mN_dir  = 0.5 * (Fr_mN_L + Fr_mN_R)   - 0.5 * smax * (mN_R - mN_L);
  const Real F_mT1_dir = 0.5 * (Fr_mT1_L + Fr_mT1_R) - 0.5 * smax * (mT1_R - mT1_L);
  const Real F_mT2_dir = 0.5 * (Fr_mT2_L + Fr_mT2_R) - 0.5 * smax * (mT2_R - mT2_L);
  const Real F_E_dir   = 0.5 * (Fr_E_L + Fr_E_R)     - 0.5 * smax * (ER - EL);

  // Map back to x,y,z ordering
  if constexpr (DIR == X1DIR) {
    *F_rho = F_rho_dir;
    *F_mx  = F_mN_dir;
    *F_my  = F_mT1_dir;
    *F_mz  = F_mT2_dir;
    *F_E   = F_E_dir;
  } else if constexpr (DIR == X2DIR) {
    *F_rho = F_rho_dir;
    *F_mx  = F_mT1_dir;
    *F_my  = F_mN_dir;
    *F_mz  = F_mT2_dir;
    *F_E   = F_E_dir;
  } else { // X3DIR
    *F_rho = F_rho_dir;
    *F_mx  = F_mT1_dir;
    *F_my  = F_mT2_dir;
    *F_mz  = F_mN_dir;
    *F_E   = F_E_dir;
  }
}

parthenon::TaskStatus ComputeFluxes(std::shared_ptr<MeshBlockData<Real>> &rc) {
  auto pmb = rc->GetBlockPointer();
  auto pkg = pmb->packages.Get("euler_sparse");
  const Real gamma = pkg->Param<Real>("gamma");

  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);

  // Build a typed pack that includes flux arrays for the requested variables.
  // Using typed tags avoids manual index arithmetic and std::unordered_map lookups.
  using euler_sparse_example::U::rho;
  using euler_sparse_example::U::mom;
  using euler_sparse_example::U::E;
  auto desc = parthenon::MakePackDescriptor<rho, mom, E>(
      rc.get(), std::vector<parthenon::MetadataFlag>{},
      std::set<parthenon::PDOpt>{parthenon::PDOpt::WithFluxes});
  auto pack = desc.GetPack(rc.get());

  // X1 fluxes
  const int scratch_level = 0;
  const size_t scratch_size = 0;
  pmb->par_for_outer(
      PARTHENON_AUTO_LABEL, scratch_size, scratch_level, kb.s, kb.e, jb.s, jb.e,
      KOKKOS_LAMBDA(parthenon::team_mbr_t member, const int k, const int j) {
        for (int i = ib.s; i <= ib.e + 1; ++i) {
          if (!(pack.Contains(0, rho()) && pack.Contains(0, mom()) &&
                pack.Contains(0, E())))
            continue;

          const int iL = i - 1;
          const int iR = i;

          const Real rhoL = pack(0, rho(), k, j, iL);
          const Real mxL  = pack(0, mom(0), k, j, iL);
          const Real myL  = pack(0, mom(1), k, j, iL);
          const Real mzL  = pack(0, mom(2), k, j, iL);
          const Real EL   = pack(0, E(), k, j, iL);

          const Real rhoR = pack(0, rho(), k, j, iR);
          const Real mxR  = pack(0, mom(0), k, j, iR);
          const Real myR  = pack(0, mom(1), k, j, iR);
          const Real mzR  = pack(0, mom(2), k, j, iR);
          const Real ER   = pack(0, E(), k, j, iR);
          Real F_rho, F_mx, F_my, F_mz, F_E;
          hll_flux_dir<X1DIR>(gamma,
                              rhoL, mxL, myL, mzL, EL,
                              rhoR, mxR, myR, mzR, ER,
                              &F_rho, &F_mx, &F_my, &F_mz, &F_E);

          pack.flux(0, 1, rho(),   k, j, i) = F_rho;
          pack.flux(0, 1, mom(0),  k, j, i) = F_mx;
          pack.flux(0, 1, mom(1),  k, j, i) = F_my;
          pack.flux(0, 1, mom(2),  k, j, i) = F_mz;
          pack.flux(0, 1, E(),     k, j, i) = F_E;
        }
      });

  // X2 fluxes (if 2D or 3D)
  if (pmb->pmy_mesh->ndim >= 2) {
    pmb->par_for_outer(
        PARTHENON_AUTO_LABEL, scratch_size, scratch_level, kb.s, kb.e, jb.s, jb.e + 1,
        KOKKOS_LAMBDA(parthenon::team_mbr_t member, const int k, const int j) {
          for (int i = ib.s; i <= ib.e; ++i) {
            if (!(pack.Contains(0, rho()) && pack.Contains(0, mom()) &&
                  pack.Contains(0, E())))
              continue;

            const int jL = j - 1;
            const int jR = j;

            const Real rhoL = pack(0, rho(), k, jL, i);
            const Real mxL  = pack(0, mom(0), k, jL, i);
            const Real myL  = pack(0, mom(1), k, jL, i);
            const Real mzL  = pack(0, mom(2), k, jL, i);
            const Real EL   = pack(0, E(), k, jL, i);

            const Real rhoR = pack(0, rho(), k, jR, i);
            const Real mxR  = pack(0, mom(0), k, jR, i);
            const Real myR  = pack(0, mom(1), k, jR, i);
            const Real mzR  = pack(0, mom(2), k, jR, i);
            const Real ER   = pack(0, E(), k, jR, i);

            Real F_rho, F_mx, F_my, F_mz, F_E;
            hll_flux_dir<X2DIR>(gamma,
                                rhoL, mxL, myL, mzL, EL,
                                rhoR, mxR, myR, mzR, ER,
                                &F_rho, &F_mx, &F_my, &F_mz, &F_E);

            pack.flux(0, 2, rho(),   k, j, i) = F_rho;
            pack.flux(0, 2, mom(0),  k, j, i) = F_mx;
            pack.flux(0, 2, mom(1),  k, j, i) = F_my;
            pack.flux(0, 2, mom(2),  k, j, i) = F_mz;
            pack.flux(0, 2, E(),     k, j, i) = F_E;
          }
        });
  }

  return TaskStatus::complete;
}

Real EstimateTimestepBlock(MeshBlockData<Real> *rc) {
  auto pmb = rc->GetBlockPointer();
  auto pkg = pmb->packages.Get("euler_sparse");
  const Real gamma = pkg->Param<Real>("gamma");

  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);

  using euler_sparse_example::U::rho;
  using euler_sparse_example::U::mom;
  using euler_sparse_example::U::E;
  auto desc = parthenon::MakePackDescriptor<rho, mom, E>(rc);
  auto pack = desc.GetPack(rc);

  auto &coords = pmb->coords;

  Real min_dt;
  pmb->par_reduce(
      PARTHENON_AUTO_LABEL, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i, Real &lmin_dt) {
        if (!(pack.Contains(0, rho()) && pack.Contains(0, mom()) && pack.Contains(0, E())))
          return;
        const Real rho_v = pack(0, rho(), k, j, i);
        const Real mx   = pack(0, mom(0), k, j, i);
        const Real my   = pack(0, mom(1), k, j, i);
        const Real mz   = pack(0, mom(2), k, j, i);
        const Real E_v  = pack(0, E(), k, j, i);
        Real u, v, w, p, a;
        cons_to_prim(gamma, rho_v, mx, my, mz, E_v, &u, &v, &w, &p, &a);
        Real inv_dt = 0.0;
        inv_dt = std::max(inv_dt, (std::abs(u) + a) / coords.Dxc<X1DIR>(k, j, i));
        if (pmb->pmy_mesh->ndim >= 2)
          inv_dt = std::max(inv_dt, (std::abs(v) + a) / coords.Dxc<X2DIR>(k, j, i));
        if (pmb->pmy_mesh->ndim >= 3)
          inv_dt = std::max(inv_dt, (std::abs(w) + a) / coords.Dxc<X3DIR>(k, j, i));
        lmin_dt = std::min(lmin_dt, 1.0 / inv_dt);
      },
      Kokkos::Min<Real>(min_dt));

  const Real cfl = pkg->Param<Real>("cfl");
  return cfl * min_dt;
}

} // namespace euler_sparse_example
