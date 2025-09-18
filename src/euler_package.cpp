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

static KOKKOS_INLINE_FUNCTION void cons_to_prim_x(
    const Real gamma, const Real rho, const Real mx, const Real my, const Real mz,
    const Real E, Real *u, Real *v, Real *w, Real *p, Real *a) {
  const Real inv_rho = 1.0 / rho;
  *u = mx * inv_rho;
  *v = my * inv_rho;
  *w = mz * inv_rho;
  const Real ke = 0.5 * rho * ((*u) * (*u) + (*v) * (*v) + (*w) * (*w));
  *p = (gamma - 1.0) * (E - ke);
  *a = std::sqrt(std::max(Real(0.0), gamma * (*p) * inv_rho));
}

parthenon::TaskStatus ComputeFluxes(std::shared_ptr<MeshBlockData<Real>> &rc) {
  auto pmb = rc->GetBlockPointer();
  auto pkg = pmb->packages.Get("euler_sparse");
  const Real gamma = pkg->Param<Real>("gamma");

  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);

  auto desc = parthenon::MakePackDescriptor(
      rc.get(), std::vector<std::string>{"U/.*"},
      std::vector<parthenon::MetadataFlag>{parthenon::Metadata::WithFluxes});
  auto U = desc.GetPack(rc.get());
  auto map = desc.GetMap();
  parthenon::PackIdx i_rho(map.at("U::rho"));
  parthenon::PackIdx i_mx(map.at("U::mom_x"));
  parthenon::PackIdx i_my(map.at("U::mom_y"));
  parthenon::PackIdx i_mz(map.at("U::mom_z"));
  parthenon::PackIdx i_E(map.at("U::E"));

  // X1 fluxes
  const int scratch_level = 0;
  const size_t scratch_size = 0;
  pmb->par_for_outer(
      PARTHENON_AUTO_LABEL, scratch_size, scratch_level, kb.s, kb.e, jb.s, jb.e,
      KOKKOS_LAMBDA(parthenon::team_mbr_t member, const int k, const int j) {
        for (int i = ib.s; i <= ib.e + 1; ++i) {
          if (!(U.Contains(0, i_rho) && U.Contains(0, i_mx) && U.Contains(0, i_my) &&
                U.Contains(0, i_E)))
            continue;

          const int iL = i - 1;
          const int iR = i;

          const Real rhoL = U(0, i_rho, k, j, iL);
          const Real mxL = U(0, i_mx, k, j, iL);
          const Real myL = U(0, i_my, k, j, iL);
          const Real mzL = U.Contains(0, i_mz) ? U(0, i_mz, k, j, iL) : 0.0;
          const Real EL = U(0, i_E, k, j, iL);

          const Real rhoR = U(0, i_rho, k, j, iR);
          const Real mxR = U(0, i_mx, k, j, iR);
          const Real myR = U(0, i_my, k, j, iR);
          const Real mzR = U.Contains(0, i_mz) ? U(0, i_mz, k, j, iR) : 0.0;
          const Real ER = U(0, i_E, k, j, iR);

          Real uL, vL, wL, pL, aL; // normal is x
          Real uR, vR, wR, pR, aR;
          cons_to_prim_x(gamma, rhoL, mxL, myL, mzL, EL, &uL, &vL, &wL, &pL, &aL);
          cons_to_prim_x(gamma, rhoR, mxR, myR, mzR, ER, &uR, &vR, &wR, &pR, &aR);
          const Real smax = std::max(std::abs(uL) + aL, std::abs(uR) + aR);

          const Real Fr_rho_L = rhoL * uL;
          const Real Fr_mx_L = mxL * uL + pL;
          const Real Fr_my_L = myL * uL;
          const Real Fr_E_L = (EL + pL) * uL;

          const Real Fr_rho_R = rhoR * uR;
          const Real Fr_mx_R = mxR * uR + pR;
          const Real Fr_my_R = myR * uR;
          const Real Fr_E_R = (ER + pR) * uR;

          U.flux(0, 1, i_rho, k, j, i) = 0.5 * (Fr_rho_L + Fr_rho_R) - 0.5 * smax * (rhoR - rhoL);
          U.flux(0, 1, i_mx, k, j, i) = 0.5 * (Fr_mx_L + Fr_mx_R) - 0.5 * smax * (mxR - mxL);
          U.flux(0, 1, i_my, k, j, i) = 0.5 * (Fr_my_L + Fr_my_R) - 0.5 * smax * (myR - myL);
          if (U.Contains(0, i_mz)) {
            const Real Fr_mz_L = mzL * uL;
            const Real Fr_mz_R = mzR * uR;
            U.flux(0, 1, i_mz, k, j, i) =
                0.5 * (Fr_mz_L + Fr_mz_R) - 0.5 * smax * (mzR - mzL);
          }
          U.flux(0, 1, i_E, k, j, i) = 0.5 * (Fr_E_L + Fr_E_R) - 0.5 * smax * (ER - EL);
        }
      });

  // X2 fluxes (if 2D or 3D)
  if (pmb->pmy_mesh->ndim >= 2) {
    pmb->par_for_outer(
        PARTHENON_AUTO_LABEL, scratch_size, scratch_level, kb.s, kb.e, jb.s, jb.e + 1,
        KOKKOS_LAMBDA(parthenon::team_mbr_t member, const int k, const int j) {
          for (int i = ib.s; i <= ib.e; ++i) {
            if (!(U.Contains(0, i_rho) && U.Contains(0, i_mx) && U.Contains(0, i_my) &&
                  U.Contains(0, i_E)))
              continue;

            const int jL = j - 1;
            const int jR = j;

            const Real rhoL = U(0, i_rho, k, jL, i);
            const Real mxL = U(0, i_mx, k, jL, i);
            const Real myL = U(0, i_my, k, jL, i);
            const Real mzL = U.Contains(0, i_mz) ? U(0, i_mz, k, jL, i) : 0.0;
            const Real EL = U(0, i_E, k, jL, i);

            const Real rhoR = U(0, i_rho, k, jR, i);
            const Real mxR = U(0, i_mx, k, jR, i);
            const Real myR = U(0, i_my, k, jR, i);
            const Real mzR = U.Contains(0, i_mz) ? U(0, i_mz, k, jR, i) : 0.0;
            const Real ER = U(0, i_E, k, jR, i);

            Real uL, vL, wL, pL, aL; // normal is y
            Real uR, vR, wR, pR, aR;
            cons_to_prim_x(gamma, rhoL, mxL, myL, mzL, EL, &uL, &vL, &wL, &pL, &aL);
            cons_to_prim_x(gamma, rhoR, mxR, myR, mzR, ER, &uR, &vR, &wR, &pR, &aR);
            const Real smax = std::max(std::abs(vL) + aL, std::abs(vR) + aR);

            const Real Fr_rho_L = rhoL * vL;
            const Real Fr_mx_L = mxL * vL;
            const Real Fr_my_L = myL * vL + pL;
            const Real Fr_E_L = (EL + pL) * vL;

            const Real Fr_rho_R = rhoR * vR;
            const Real Fr_mx_R = mxR * vR;
            const Real Fr_my_R = myR * vR + pR;
            const Real Fr_E_R = (ER + pR) * vR;

            U.flux(0, 2, i_rho, k, j, i) = 0.5 * (Fr_rho_L + Fr_rho_R) - 0.5 * smax * (rhoR - rhoL);
            U.flux(0, 2, i_mx, k, j, i) = 0.5 * (Fr_mx_L + Fr_mx_R) - 0.5 * smax * (mxR - mxL);
            U.flux(0, 2, i_my, k, j, i) = 0.5 * (Fr_my_L + Fr_my_R) - 0.5 * smax * (myR - myL);
            if (U.Contains(0, i_mz)) {
              const Real Fr_mz_L = mzL * vL;
              const Real Fr_mz_R = mzR * vR;
              U.flux(0, 2, i_mz, k, j, i) =
                  0.5 * (Fr_mz_L + Fr_mz_R) - 0.5 * smax * (mzR - mzL);
            }
            U.flux(0, 2, i_E, k, j, i) = 0.5 * (Fr_E_L + Fr_E_R) - 0.5 * smax * (ER - EL);
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

  auto desc = parthenon::MakePackDescriptor(
      rc, std::vector<std::string>{"U/.*"});
  auto U = desc.GetPack(rc);
  auto map = desc.GetMap();
  parthenon::PackIdx i_rho(map.at("U::rho"));
  parthenon::PackIdx i_mx(map.at("U::mom_x"));
  parthenon::PackIdx i_my(map.at("U::mom_y"));
  parthenon::PackIdx i_mz(map.at("U::mom_z"));
  parthenon::PackIdx i_E(map.at("U::E"));

  auto &coords = pmb->coords;

  Real min_dt;
  pmb->par_reduce(
      PARTHENON_AUTO_LABEL, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i, Real &lmin_dt) {
        if (!(U.Contains(0, i_rho) && U.Contains(0, i_mx) && U.Contains(0, i_my) &&
              U.Contains(0, i_E)))
          return;
        const Real rho = U(0, i_rho, k, j, i);
        const Real mx = U(0, i_mx, k, j, i);
        const Real my = U(0, i_my, k, j, i);
        const Real mz = U.Contains(0, i_mz) ? U(0, i_mz, k, j, i) : 0.0;
        const Real E = U(0, i_E, k, j, i);
        Real u, v, w, p, a;
        cons_to_prim_x(gamma, rho, mx, my, mz, E, &u, &v, &w, &p, &a);
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
