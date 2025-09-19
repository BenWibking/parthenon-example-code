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

#include "euler_package.hpp"

using namespace parthenon::package::prelude;

namespace euler_sparse_example {

static inline Real max3(Real a, Real b, Real c) {
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

  // Separate sparse pools for each state variable. For a single material,
  // we allocate sparse ID 0 in each pool, yielding labels: rho_0, mom_0, E_0.
  // Additional materials would use further sparse IDs per pool (e.g., *_1, *_2, ...).

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

// MeshData variant: compute fluxes for all blocks (explicit b index)
parthenon::TaskStatus ComputeFluxes(MeshData<Real> *md) {
  auto pm = md->GetMeshPointer();
  auto pkg = pm->packages.Get("euler_sparse");
  const Real gamma = pkg->Param<Real>("gamma");

  IndexRange ib = md->GetBoundsI(IndexDomain::interior);
  IndexRange jb = md->GetBoundsJ(IndexDomain::interior);
  IndexRange kb = md->GetBoundsK(IndexDomain::interior);

  using euler_sparse_example::U::rho;
  using euler_sparse_example::U::mom;
  using euler_sparse_example::U::E;
  auto desc = parthenon::MakePackDescriptor<rho, mom, E>(
      md, std::vector<parthenon::MetadataFlag>{},
      std::set<parthenon::PDOpt>{parthenon::PDOpt::WithFluxes});
  auto pack = desc.GetPack(md);

  const int nblocks = md->NumBlocks();
  const int scratch_level = 0;
  const size_t scratch_size = 0;

  // X1 fluxes across all blocks
  parthenon::par_for_outer(
      DEFAULT_OUTER_LOOP_PATTERN, PARTHENON_AUTO_LABEL, DevExecSpace(), scratch_size,
      scratch_level, 0, nblocks - 1, kb.s, kb.e, jb.s, jb.e,
      KOKKOS_LAMBDA(parthenon::team_mbr_t member, const int b, const int k, const int j) {
        if (!(pack.Contains(b, rho()) && pack.Contains(b, mom()) && pack.Contains(b, E())))
          return;

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

        parthenon::par_for_inner(member, ib.s, ib.e + 1, [&](const int i) {
          const int iL = i - 1;
          const int iR = i;

          const Real rhoL = rho_bkj[iL];
          const Real mxL  = mx_bkj[iL];
          const Real myL  = my_bkj[iL];
          const Real mzL  = mz_bkj[iL];
          const Real EL   = E_bkj[iL];

          const Real rhoR = rho_bkj[iR];
          const Real mxR  = mx_bkj[iR];
          const Real myR  = my_bkj[iR];
          const Real mzR  = mz_bkj[iR];
          const Real ER   = E_bkj[iR];

          Real F_rho, F_mx, F_my, F_mz, F_E;
          hll_flux_dir<X1DIR>(gamma,
                              rhoL, mxL, myL, mzL, EL,
                              rhoR, mxR, myR, mzR, ER,
                              &F_rho, &F_mx, &F_my, &F_mz, &F_E);

          Fx_rho[i] = F_rho;
          Fx_mx[i]  = F_mx;
          Fx_my[i]  = F_my;
          Fx_mz[i]  = F_mz;
          Fx_E[i]   = F_E;
        });
      });

  // X2 fluxes (if 2D or 3D)
  if (pm->ndim >= 2) {
    parthenon::par_for_outer(
        DEFAULT_OUTER_LOOP_PATTERN, PARTHENON_AUTO_LABEL, DevExecSpace(), scratch_size,
        scratch_level, 0, nblocks - 1, kb.s, kb.e, jb.s, jb.e + 1,
        KOKKOS_LAMBDA(parthenon::team_mbr_t member, const int b, const int k, const int j) {
          if (!(pack.Contains(b, rho()) && pack.Contains(b, mom()) && pack.Contains(b, E())))
            return;

          const int jL = j - 1;
          const int jR = j;

          Real *rho_jL = &pack(b, rho(),   k, jL, 0);
          Real *mx_jL  = &pack(b, mom(0), k, jL, 0);
          Real *my_jL  = &pack(b, mom(1), k, jL, 0);
          Real *mz_jL  = &pack(b, mom(2), k, jL, 0);
          Real *E_jL   = &pack(b, E(),     k, jL, 0);

          Real *rho_jR = &pack(b, rho(),   k, jR, 0);
          Real *mx_jR  = &pack(b, mom(0), k, jR, 0);
          Real *my_jR  = &pack(b, mom(1), k, jR, 0);
          Real *mz_jR  = &pack(b, mom(2), k, jR, 0);
          Real *E_jR   = &pack(b, E(),     k, jR, 0);

          Real *Fy_rho = &pack.flux(b, 2, rho(),   k, j, 0);
          Real *Fy_mx  = &pack.flux(b, 2, mom(0),  k, j, 0);
          Real *Fy_my  = &pack.flux(b, 2, mom(1),  k, j, 0);
          Real *Fy_mz  = &pack.flux(b, 2, mom(2),  k, j, 0);
          Real *Fy_E   = &pack.flux(b, 2, E(),     k, j, 0);

          parthenon::par_for_inner(member, ib.s, ib.e, [&](const int i) {
            const Real rhoL = rho_jL[i];
            const Real mxL  = mx_jL[i];
            const Real myL  = my_jL[i];
            const Real mzL  = mz_jL[i];
            const Real EL   = E_jL[i];

            const Real rhoR = rho_jR[i];
            const Real mxR  = mx_jR[i];
            const Real myR  = my_jR[i];
            const Real mzR  = mz_jR[i];
            const Real ER   = E_jR[i];

            Real F_rho, F_mx, F_my, F_mz, F_E;
            hll_flux_dir<X2DIR>(gamma,
                                rhoL, mxL, myL, mzL, EL,
                                rhoR, mxR, myR, mzR, ER,
                                &F_rho, &F_mx, &F_my, &F_mz, &F_E);

            Fy_rho[i] = F_rho;
            Fy_mx[i]  = F_mx;
            Fy_my[i]  = F_my;
            Fy_mz[i]  = F_mz;
            Fy_E[i]   = F_E;
          });
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

        const Real rho_v = pack(0, rho(),   k, j, i);
        const Real mx    = pack(0, mom(0), k, j, i);
        const Real my    = pack(0, mom(1), k, j, i);
        const Real mz    = pack(0, mom(2), k, j, i);
        const Real E_v   = pack(0, E(),     k, j, i);

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
