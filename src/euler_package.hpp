//========================================================================================
// Minimal Euler (hyperbolic) package using sparse fields
//========================================================================================
#ifndef EXAMPLE_EULER_SPARSE_EULER_PACKAGE_HPP_
#define EXAMPLE_EULER_SPARSE_EULER_PACKAGE_HPP_

#include <memory>

#include <parthenon/package.hpp>

namespace euler_sparse_example {

// Typed variable tags for SparsePack access.
// IMPORTANT: For sparse pools, Parthenon variable labels are "<base>_<sparse_id>".
// We now maintain separate sparse pools for each state variable: bases are
// "rho", "mom", and "E". For a single material, we use sparse ID 0, thus
// variable labels are "rho_0", "mom_0", and "E_0".
namespace U {
struct rho : public parthenon::variable_names::base_t<false> {
  template <class... Ts>
  KOKKOS_INLINE_FUNCTION rho(Ts &&...args)
      : parthenon::variable_names::base_t<false>(std::forward<Ts>(args)...) {}
  static std::string name() { return "rho_0"; }
};
// Momentum is a 3-vector under sparse id 1
struct mom : public parthenon::variable_names::base_t<false, 3> {
  template <class... Ts>
  KOKKOS_INLINE_FUNCTION mom(Ts &&...args)
      : parthenon::variable_names::base_t<false, 3>(std::forward<Ts>(args)...) {}
  static std::string name() { return "mom_0"; }
};
struct E : public parthenon::variable_names::base_t<false> {
  template <class... Ts>
  KOKKOS_INLINE_FUNCTION E(Ts &&...args)
      : parthenon::variable_names::base_t<false>(std::forward<Ts>(args)...) {}
  static std::string name() { return "E_0"; }
};
} // namespace U

std::shared_ptr<parthenon::StateDescriptor> Initialize(parthenon::ParameterInput *pin);

parthenon::TaskStatus ComputeFluxes(
    std::shared_ptr<parthenon::MeshBlockData<parthenon::Real>> &rc);

// MeshData variant with explicit block index in loops
parthenon::TaskStatus ComputeFluxes(parthenon::MeshData<parthenon::Real> *md);

// MeshData variant computing fluxes using PLM reconstruction with MC limiter
parthenon::TaskStatus ComputeFluxesPLM_MC(parthenon::MeshData<parthenon::Real> *md);

// MeshBlock variant for dt estimation (per-block)
parthenon::Real EstimateTimestepBlock(
    parthenon::MeshBlockData<parthenon::Real> *rc);

} // namespace euler_sparse_example

#endif // EXAMPLE_EULER_SPARSE_EULER_PACKAGE_HPP_
