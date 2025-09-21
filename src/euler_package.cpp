//========================================================================================
// Minimal Euler (hyperbolic) package using sparse fields and SparsePack
//========================================================================================

#include <cmath>
#include <memory>
#include <string>
#include <vector>

#include <parthenon/package.hpp>
#include "globals.hpp"

#include "interface/metadata.hpp"
#include "interface/sparse_pool.hpp"
#include "mesh/meshblock.hpp"
#include "pack/make_pack_descriptor.hpp"

#include "euler_package.hpp"

namespace euler_sparse_example {

static inline parthenon::Real max3(parthenon::Real a, parthenon::Real b, parthenon::Real c) {
  return std::max(a, std::max(b, c));
}

KOKKOS_INLINE_FUNCTION static parthenon::Real minmod3(parthenon::Real a, parthenon::Real b,
                                                      parthenon::Real c) {
  const parthenon::Real sa = (a > 0) - (a < 0);
  const parthenon::Real sb = (b > 0) - (b < 0);
  const parthenon::Real sc = (c > 0) - (c < 0);
  if (sa == sb && sb == sc) {
    const parthenon::Real ab = fabs(a);
    const parthenon::Real bb = fabs(b);
    const parthenon::Real cb = fabs(c);
    const parthenon::Real m = ab < bb ? (ab < cb ? ab : cb) : (bb < cb ? bb : cb);
    return sa * m;
  }
  return parthenon::Real(0.0);
}

std::shared_ptr<parthenon::StateDescriptor> Initialize(parthenon::ParameterInput *pin) {
  auto pkg = std::make_shared<parthenon::StateDescriptor>("euler_sparse");

  const parthenon::Real gamma = pin->GetOrAddReal("euler", "gamma", 1.4);
  const parthenon::Real cfl = pin->GetOrAddReal("euler", "cfl", 0.45);
  pkg->AddParam("gamma", gamma);
  pkg->AddParam("cfl", cfl);

  parthenon::Metadata m({parthenon::Metadata::Cell, parthenon::Metadata::Independent,
                         parthenon::Metadata::WithFluxes, parthenon::Metadata::FillGhost,
                         parthenon::Metadata::Sparse});

  // Separate sparse pools for each state variable. For a single material,
  // we allocate sparse ID 0 in each pool, yielding labels: rho_0, mom_0, E_0.
  // Additional materials would use further sparse IDs per pool (e.g., *_1, *_2, ...).

  // Density pool
  {
    parthenon::SparsePool rho_pool("rho", m);
    rho_pool.Add(0, std::vector<int>{1}, std::vector<std::string>{"rho"});
    pkg->AddSparsePool(rho_pool);
  }

  // Momentum (vector<3>) pool
  {
    parthenon::SparsePool mom_pool("mom", m);
    mom_pool.Add(0, std::vector<int>{3}, parthenon::Metadata::Vector,
                 std::vector<std::string>{"mom_x", "mom_y", "mom_z"});
    pkg->AddSparsePool(mom_pool);
  }

  // Total energy pool
  {
    parthenon::SparsePool E_pool("E", m);
    E_pool.Add(0, std::vector<int>{1}, std::vector<std::string>{"E"});
    pkg->AddSparsePool(E_pool);
  }

  // Require at least two ghost cells for PLM+MC reconstruction
  PARTHENON_REQUIRE_THROWS(parthenon::Globals::nghost >= 2,
                           "euler_sparse requires parthenon/mesh.nghost >= 2");

  pkg->EstimateTimestepBlock = EstimateTimestepBlock;
  return pkg;
}

static KOKKOS_INLINE_FUNCTION void cons_to_prim(
    const parthenon::Real gamma, const parthenon::Real rho, const parthenon::Real mN,
    const parthenon::Real mT1, const parthenon::Real mT2, const parthenon::Real E,
    parthenon::Real &u, parthenon::Real &v, parthenon::Real &w, parthenon::Real &p,
    parthenon::Real &a) {
  const parthenon::Real inv_rho = 1.0 / rho;
  u = mN * inv_rho;
  v = mT1 * inv_rho;
  w = mT2 * inv_rho;
  const parthenon::Real ke = 0.5 * rho * (u * u + v * v + w * w);
  p = (gamma - 1.0) * (E - ke);
  a = std::sqrt(std::max(parthenon::Real(0.0), gamma * p * inv_rho));
}

template <int DIR>
KOKKOS_INLINE_FUNCTION void hll_flux_dir(
    const parthenon::Real gamma, const parthenon::Real rhoL, const parthenon::Real mxL,
    const parthenon::Real myL, const parthenon::Real mzL, const parthenon::Real EL,
    const parthenon::Real rhoR, const parthenon::Real mxR, const parthenon::Real myR,
    const parthenon::Real mzR, const parthenon::Real ER, parthenon::Real &F_rho,
    parthenon::Real &F_mx, parthenon::Real &F_my, parthenon::Real &F_mz,
    parthenon::Real &F_E) {
  // Map momenta so mN is normal, mT1/mT2 tangential to the face normal
  parthenon::Real mN_L, mT1_L, mT2_L;
  parthenon::Real mN_R, mT1_R, mT2_R;
  if constexpr (DIR == parthenon::X1DIR) {
    mN_L = mxL; mT1_L = myL; mT2_L = mzL;
    mN_R = mxR; mT1_R = myR; mT2_R = mzR;
  } else if constexpr (DIR == parthenon::X2DIR) {
    mN_L = myL; mT1_L = mxL; mT2_L = mzL;
    mN_R = myR; mT1_R = mxR; mT2_R = mzR;
  } else { // X3DIR (not used here, but keep generic)
    mN_L = mzL; mT1_L = mxL; mT2_L = myL;
    mN_R = mzR; mT1_R = mxR; mT2_R = myR;
  }

  parthenon::Real uL, vL, wL, pL, aL; // u = normal velocity
  parthenon::Real uR, vR, wR, pR, aR;
  cons_to_prim(gamma, rhoL, mN_L, mT1_L, mT2_L, EL, uL, vL, wL, pL, aL);
  cons_to_prim(gamma, rhoR, mN_R, mT1_R, mT2_R, ER, uR, vR, wR, pR, aR);

  const parthenon::Real smax = std::max(std::abs(uL) + aL, std::abs(uR) + aR);

  // Physical fluxes (left/right) in the normal direction
  const parthenon::Real Fr_rho_L = rhoL * uL;
  const parthenon::Real Fr_mN_L = mN_L * uL + pL;
  const parthenon::Real Fr_mT1_L = mT1_L * uL;
  const parthenon::Real Fr_mT2_L = mT2_L * uL;
  const parthenon::Real Fr_E_L = (EL + pL) * uL;

  const parthenon::Real Fr_rho_R = rhoR * uR;
  const parthenon::Real Fr_mN_R = mN_R * uR + pR;
  const parthenon::Real Fr_mT1_R = mT1_R * uR;
  const parthenon::Real Fr_mT2_R = mT2_R * uR;
  const parthenon::Real Fr_E_R = (ER + pR) * uR;

  // HLL (Rusanov) flux for each conservative component
  const parthenon::Real F_rho_dir = 0.5 * (Fr_rho_L + Fr_rho_R) - 0.5 * smax * (rhoR - rhoL);
  const parthenon::Real F_mN_dir  = 0.5 * (Fr_mN_L + Fr_mN_R)   - 0.5 * smax * (mN_R - mN_L);
  const parthenon::Real F_mT1_dir = 0.5 * (Fr_mT1_L + Fr_mT1_R) - 0.5 * smax * (mT1_R - mT1_L);
  const parthenon::Real F_mT2_dir = 0.5 * (Fr_mT2_L + Fr_mT2_R) - 0.5 * smax * (mT2_R - mT2_L);
  const parthenon::Real F_E_dir   = 0.5 * (Fr_E_L + Fr_E_R)     - 0.5 * smax * (ER - EL);

  // Map back to x,y,z ordering
  if constexpr (DIR == parthenon::X1DIR) {
    F_rho = F_rho_dir;
    F_mx  = F_mN_dir;
    F_my  = F_mT1_dir;
    F_mz  = F_mT2_dir;
    F_E   = F_E_dir;
  } else if constexpr (DIR == parthenon::X2DIR) {
    F_rho = F_rho_dir;
    F_mx  = F_mT1_dir;
    F_my  = F_mN_dir;
    F_mz  = F_mT2_dir;
    F_E   = F_E_dir;
  } else { // X3DIR
    F_rho = F_rho_dir;
    F_mx  = F_mT1_dir;
    F_my  = F_mT2_dir;
    F_mz  = F_mN_dir;
    F_E   = F_E_dir;
  }
}

// MeshData variant: compute fluxes for all blocks (explicit b index)
parthenon::TaskStatus ComputeFluxes(parthenon::MeshData<parthenon::Real> *md) {
  auto pm = md->GetMeshPointer();
  auto pkg = pm->packages.Get("euler_sparse");
  const parthenon::Real gamma = pkg->Param<parthenon::Real>("gamma");

  parthenon::IndexRange ib = md->GetBoundsI(parthenon::IndexDomain::interior);
  parthenon::IndexRange jb = md->GetBoundsJ(parthenon::IndexDomain::interior);
  parthenon::IndexRange kb = md->GetBoundsK(parthenon::IndexDomain::interior);

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
      DEFAULT_OUTER_LOOP_PATTERN, PARTHENON_AUTO_LABEL, parthenon::DevExecSpace(), scratch_size,
      scratch_level, 0, nblocks - 1, kb.s, kb.e, jb.s, jb.e,
      KOKKOS_LAMBDA(parthenon::team_mbr_t member, const int b, const int k, const int j) {
        if (!(pack.Contains(b, rho()) && pack.Contains(b, mom()) && pack.Contains(b, E())))
          return;

        parthenon::Real *rho_bkj = &pack(b, rho(),   k, j, 0);
        parthenon::Real *mx_bkj  = &pack(b, mom(0), k, j, 0);
        parthenon::Real *my_bkj  = &pack(b, mom(1), k, j, 0);
        parthenon::Real *mz_bkj  = &pack(b, mom(2), k, j, 0);
        parthenon::Real *E_bkj   = &pack(b, E(),     k, j, 0);

        parthenon::Real *Fx_rho = &pack.flux(b, 1, rho(),   k, j, 0);
        parthenon::Real *Fx_mx  = &pack.flux(b, 1, mom(0),  k, j, 0);
        parthenon::Real *Fx_my  = &pack.flux(b, 1, mom(1),  k, j, 0);
        parthenon::Real *Fx_mz  = &pack.flux(b, 1, mom(2),  k, j, 0);
        parthenon::Real *Fx_E   = &pack.flux(b, 1, E(),     k, j, 0);

        parthenon::par_for_inner(member, ib.s, ib.e + 1, [&](const int i) {
          const int iL = i - 1;
          const int iR = i;

          const parthenon::Real rhoL = rho_bkj[iL];
          const parthenon::Real mxL  = mx_bkj[iL];
          const parthenon::Real myL  = my_bkj[iL];
          const parthenon::Real mzL  = mz_bkj[iL];
          const parthenon::Real EL   = E_bkj[iL];

          const parthenon::Real rhoR = rho_bkj[iR];
          const parthenon::Real mxR  = mx_bkj[iR];
          const parthenon::Real myR  = my_bkj[iR];
          const parthenon::Real mzR  = mz_bkj[iR];
          const parthenon::Real ER   = E_bkj[iR];

          parthenon::Real F_rho, F_mx, F_my, F_mz, F_E;
          hll_flux_dir<parthenon::X1DIR>(gamma,
                              rhoL, mxL, myL, mzL, EL,
                              rhoR, mxR, myR, mzR, ER,
                              F_rho, F_mx, F_my, F_mz, F_E);

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
        DEFAULT_OUTER_LOOP_PATTERN, PARTHENON_AUTO_LABEL, parthenon::DevExecSpace(), scratch_size,
        scratch_level, 0, nblocks - 1, kb.s, kb.e, jb.s, jb.e + 1,
        KOKKOS_LAMBDA(parthenon::team_mbr_t member, const int b, const int k, const int j) {
          if (!(pack.Contains(b, rho()) && pack.Contains(b, mom()) && pack.Contains(b, E())))
            return;

          const int jL = j - 1;
          const int jR = j;

          parthenon::Real *rho_jL = &pack(b, rho(),   k, jL, 0);
          parthenon::Real *mx_jL  = &pack(b, mom(0), k, jL, 0);
          parthenon::Real *my_jL  = &pack(b, mom(1), k, jL, 0);
          parthenon::Real *mz_jL  = &pack(b, mom(2), k, jL, 0);
          parthenon::Real *E_jL   = &pack(b, E(),     k, jL, 0);

          parthenon::Real *rho_jR = &pack(b, rho(),   k, jR, 0);
          parthenon::Real *mx_jR  = &pack(b, mom(0), k, jR, 0);
          parthenon::Real *my_jR  = &pack(b, mom(1), k, jR, 0);
          parthenon::Real *mz_jR  = &pack(b, mom(2), k, jR, 0);
          parthenon::Real *E_jR   = &pack(b, E(),     k, jR, 0);

          parthenon::Real *Fy_rho = &pack.flux(b, 2, rho(),   k, j, 0);
          parthenon::Real *Fy_mx  = &pack.flux(b, 2, mom(0),  k, j, 0);
          parthenon::Real *Fy_my  = &pack.flux(b, 2, mom(1),  k, j, 0);
          parthenon::Real *Fy_mz  = &pack.flux(b, 2, mom(2),  k, j, 0);
          parthenon::Real *Fy_E   = &pack.flux(b, 2, E(),     k, j, 0);

          parthenon::par_for_inner(member, ib.s, ib.e, [&](const int i) {
            const parthenon::Real rhoL = rho_jL[i];
            const parthenon::Real mxL  = mx_jL[i];
            const parthenon::Real myL  = my_jL[i];
            const parthenon::Real mzL  = mz_jL[i];
            const parthenon::Real EL   = E_jL[i];

            const parthenon::Real rhoR = rho_jR[i];
            const parthenon::Real mxR  = mx_jR[i];
            const parthenon::Real myR  = my_jR[i];
            const parthenon::Real mzR  = mz_jR[i];
            const parthenon::Real ER   = E_jR[i];

            parthenon::Real F_rho, F_mx, F_my, F_mz, F_E;
            hll_flux_dir<parthenon::X2DIR>(gamma,
                                rhoL, mxL, myL, mzL, EL,
                                rhoR, mxR, myR, mzR, ER,
                                F_rho, F_mx, F_my, F_mz, F_E);

            Fy_rho[i] = F_rho;
            Fy_mx[i]  = F_mx;
            Fy_my[i]  = F_my;
            Fy_mz[i]  = F_mz;
            Fy_E[i]   = F_E;
          });
        });
  }

  return parthenon::TaskStatus::complete;
}

// MeshData variant: compute fluxes using PLM reconstruction with MC limiter
parthenon::TaskStatus ComputeFluxesPLM_MC(parthenon::MeshData<parthenon::Real> *md) {
  auto pm = md->GetMeshPointer();
  auto pkg = pm->packages.Get("euler_sparse");
  const parthenon::Real gamma = pkg->Param<parthenon::Real>("gamma");

  parthenon::IndexRange ib = md->GetBoundsI(parthenon::IndexDomain::interior);
  parthenon::IndexRange jb = md->GetBoundsJ(parthenon::IndexDomain::interior);
  parthenon::IndexRange kb = md->GetBoundsK(parthenon::IndexDomain::interior);

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
      DEFAULT_OUTER_LOOP_PATTERN, PARTHENON_AUTO_LABEL, parthenon::DevExecSpace(), scratch_size,
      scratch_level, 0, nblocks - 1, kb.s, kb.e, jb.s, jb.e,
      KOKKOS_LAMBDA(parthenon::team_mbr_t member, const int b, const int k, const int j) {
        if (!(pack.Contains(b, rho()) && pack.Contains(b, mom()) && pack.Contains(b, E())))
          return;

        parthenon::Real *rho_bkj = &pack(b, rho(),   k, j, 0);
        parthenon::Real *mx_bkj  = &pack(b, mom(0), k, j, 0);
        parthenon::Real *my_bkj  = &pack(b, mom(1), k, j, 0);
        parthenon::Real *mz_bkj  = &pack(b, mom(2), k, j, 0);
        parthenon::Real *E_bkj   = &pack(b, E(),     k, j, 0);

        parthenon::Real *Fx_rho = &pack.flux(b, 1, rho(),   k, j, 0);
        parthenon::Real *Fx_mx  = &pack.flux(b, 1, mom(0),  k, j, 0);
        parthenon::Real *Fx_my  = &pack.flux(b, 1, mom(1),  k, j, 0);
        parthenon::Real *Fx_mz  = &pack.flux(b, 1, mom(2),  k, j, 0);
        parthenon::Real *Fx_E   = &pack.flux(b, 1, E(),     k, j, 0);

        parthenon::par_for_inner(member, ib.s, ib.e + 1, [&](const int i_face) {
          const int iL = i_face - 1;
          const int iR = i_face;

          auto mc_slope = [&](parthenon::Real vm1, parthenon::Real v0, parthenon::Real vp1) {
            const parthenon::Real dl = v0 - vm1;
            const parthenon::Real dr = vp1 - v0;
            const parthenon::Real dc = 0.5 * (vp1 - vm1);
            return minmod3(2.0 * dl, 2.0 * dr, dc);
          };

          parthenon::Real rhoL = rho_bkj[iL] + 0.5 * mc_slope(rho_bkj[iL - 1], rho_bkj[iL], rho_bkj[iL + 1]);
          parthenon::Real rhoR = rho_bkj[iR] - 0.5 * mc_slope(rho_bkj[iR - 1], rho_bkj[iR], rho_bkj[iR + 1]);

          parthenon::Real mxL  = mx_bkj[iL]  + 0.5 * mc_slope(mx_bkj[iL - 1],  mx_bkj[iL],  mx_bkj[iL + 1]);
          parthenon::Real mxR  = mx_bkj[iR]  - 0.5 * mc_slope(mx_bkj[iR - 1],  mx_bkj[iR],  mx_bkj[iR + 1]);

          parthenon::Real myL  = my_bkj[iL]  + 0.5 * mc_slope(my_bkj[iL - 1],  my_bkj[iL],  my_bkj[iL + 1]);
          parthenon::Real myR  = my_bkj[iR]  - 0.5 * mc_slope(my_bkj[iR - 1],  my_bkj[iR],  my_bkj[iR + 1]);

          parthenon::Real mzL  = mz_bkj[iL]  + 0.5 * mc_slope(mz_bkj[iL - 1],  mz_bkj[iL],  mz_bkj[iL + 1]);
          parthenon::Real mzR  = mz_bkj[iR]  - 0.5 * mc_slope(mz_bkj[iR - 1],  mz_bkj[iR],  mz_bkj[iR + 1]);

          parthenon::Real EL   = E_bkj[iL]   + 0.5 * mc_slope(E_bkj[iL - 1],   E_bkj[iL],   E_bkj[iL + 1]);
          parthenon::Real ER   = E_bkj[iR]   - 0.5 * mc_slope(E_bkj[iR - 1],   E_bkj[iR],   E_bkj[iR + 1]);

          parthenon::Real F_rho, F_mx, F_my, F_mz, F_E;
          hll_flux_dir<parthenon::X1DIR>(gamma,
                              rhoL, mxL, myL, mzL, EL,
                              rhoR, mxR, myR, mzR, ER,
                              F_rho, F_mx, F_my, F_mz, F_E);

          Fx_rho[i_face] = F_rho;
          Fx_mx[i_face]  = F_mx;
          Fx_my[i_face]  = F_my;
          Fx_mz[i_face]  = F_mz;
          Fx_E[i_face]   = F_E;
        });
      });

  // X2 fluxes (if 2D or 3D)
  if (pm->ndim >= 2) {
    parthenon::par_for_outer(
        DEFAULT_OUTER_LOOP_PATTERN, PARTHENON_AUTO_LABEL, parthenon::DevExecSpace(), scratch_size,
        scratch_level, 0, nblocks - 1, kb.s, kb.e, jb.s, jb.e + 1,
        KOKKOS_LAMBDA(parthenon::team_mbr_t member, const int b, const int k, const int j_face) {
          if (!(pack.Contains(b, rho()) && pack.Contains(b, mom()) && pack.Contains(b, E())))
            return;

          const int jL = j_face - 1;
          const int jR = j_face;

          // We'll access values via pack(b, var, k, j, i) directly

          parthenon::Real *Fy_rho = &pack.flux(b, 2, rho(),   k, j_face, 0);
          parthenon::Real *Fy_mx  = &pack.flux(b, 2, mom(0),  k, j_face, 0);
          parthenon::Real *Fy_my  = &pack.flux(b, 2, mom(1),  k, j_face, 0);
          parthenon::Real *Fy_mz  = &pack.flux(b, 2, mom(2),  k, j_face, 0);
          parthenon::Real *Fy_E   = &pack.flux(b, 2, E(),     k, j_face, 0);

          parthenon::par_for_inner(member, ib.s, ib.e, [&](const int i) {
            auto mc_slope_y = [&](auto var_tag, int jcell) {
              const parthenon::Real vm1 = pack(b, var_tag, k, jcell - 1, i);
              const parthenon::Real v0  = pack(b, var_tag, k, jcell, i);
              const parthenon::Real vp1 = pack(b, var_tag, k, jcell + 1, i);
              const parthenon::Real dl = v0 - vm1;
              const parthenon::Real dr = vp1 - v0;
              const parthenon::Real dc = 0.5 * (vp1 - vm1);
              return minmod3(2.0 * dl, 2.0 * dr, dc);
            };

            parthenon::Real rhoL = pack(b, rho(),   k, jL, i) + 0.5 * mc_slope_y(rho(), jL);
            parthenon::Real rhoR = pack(b, rho(),   k, jR, i) - 0.5 * mc_slope_y(rho(), jR);

            parthenon::Real mxL  = pack(b, mom(0), k, jL, i) + 0.5 * mc_slope_y(mom(0), jL);
            parthenon::Real mxR  = pack(b, mom(0), k, jR, i) - 0.5 * mc_slope_y(mom(0), jR);

            parthenon::Real myL  = pack(b, mom(1), k, jL, i) + 0.5 * mc_slope_y(mom(1), jL);
            parthenon::Real myR  = pack(b, mom(1), k, jR, i) - 0.5 * mc_slope_y(mom(1), jR);

            parthenon::Real mzL  = pack(b, mom(2), k, jL, i) + 0.5 * mc_slope_y(mom(2), jL);
            parthenon::Real mzR  = pack(b, mom(2), k, jR, i) - 0.5 * mc_slope_y(mom(2), jR);

            parthenon::Real EL   = pack(b, E(),     k, jL, i) + 0.5 * mc_slope_y(E(), jL);
            parthenon::Real ER   = pack(b, E(),     k, jR, i) - 0.5 * mc_slope_y(E(), jR);

            parthenon::Real F_rho, F_mx, F_my, F_mz, F_E;
            hll_flux_dir<parthenon::X2DIR>(gamma,
                                rhoL, mxL, myL, mzL, EL,
                                rhoR, mxR, myR, mzR, ER,
                                F_rho, F_mx, F_my, F_mz, F_E);

            Fy_rho[i] = F_rho;
            Fy_mx[i]  = F_mx;
            Fy_my[i]  = F_my;
            Fy_mz[i]  = F_mz;
            Fy_E[i]   = F_E;
          });
        });
  }

  return parthenon::TaskStatus::complete;
}

parthenon::Real EstimateTimestepBlock(parthenon::MeshBlockData<parthenon::Real> *rc) {
  auto pmb = rc->GetBlockPointer();
  auto pkg = pmb->packages.Get("euler_sparse");
  const parthenon::Real gamma = pkg->Param<parthenon::Real>("gamma");

  parthenon::IndexRange ib = pmb->cellbounds.GetBoundsI(parthenon::IndexDomain::interior);
  parthenon::IndexRange jb = pmb->cellbounds.GetBoundsJ(parthenon::IndexDomain::interior);
  parthenon::IndexRange kb = pmb->cellbounds.GetBoundsK(parthenon::IndexDomain::interior);

  using euler_sparse_example::U::rho;
  using euler_sparse_example::U::mom;
  using euler_sparse_example::U::E;
  auto desc = parthenon::MakePackDescriptor<rho, mom, E>(rc);
  auto pack = desc.GetPack(rc);

  auto &coords = pmb->coords;

  parthenon::Real min_dt;
  pmb->par_reduce(
      PARTHENON_AUTO_LABEL, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i, parthenon::Real &lmin_dt) {
        if (!(pack.Contains(0, rho()) && pack.Contains(0, mom()) && pack.Contains(0, E())))
          return;

        const parthenon::Real rho_v = pack(0, rho(),   k, j, i);
        const parthenon::Real mx    = pack(0, mom(0), k, j, i);
        const parthenon::Real my    = pack(0, mom(1), k, j, i);
        const parthenon::Real mz    = pack(0, mom(2), k, j, i);
        const parthenon::Real E_v   = pack(0, E(),     k, j, i);

        parthenon::Real u, v, w, p, a;
        cons_to_prim(gamma, rho_v, mx, my, mz, E_v, u, v, w, p, a);

        parthenon::Real inv_dt = 0.0;
        inv_dt = std::max(inv_dt, (std::abs(u) + a) / coords.Dxc<parthenon::X1DIR>(k, j, i));
        if (pmb->pmy_mesh->ndim >= 2)
          inv_dt = std::max(inv_dt, (std::abs(v) + a) / coords.Dxc<parthenon::X2DIR>(k, j, i));
        if (pmb->pmy_mesh->ndim >= 3)
          inv_dt = std::max(inv_dt, (std::abs(w) + a) / coords.Dxc<parthenon::X3DIR>(k, j, i));
        lmin_dt = std::min(lmin_dt, 1.0 / inv_dt);
      },
      Kokkos::Min<parthenon::Real>(min_dt));

  const parthenon::Real cfl = pkg->Param<parthenon::Real>("cfl");
  return cfl * min_dt;
}

} // namespace euler_sparse_example
