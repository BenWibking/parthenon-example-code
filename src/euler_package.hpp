//========================================================================================
// Minimal Euler (hyperbolic) package using sparse fields
//========================================================================================
#ifndef EXAMPLE_EULER_SPARSE_EULER_PACKAGE_HPP_
#define EXAMPLE_EULER_SPARSE_EULER_PACKAGE_HPP_

#include <memory>

#include <parthenon/package.hpp>

namespace euler_sparse_example {

using namespace parthenon::package::prelude;

// Typed variable tags for SparsePack access.
// IMPORTANT: For sparse pools, Parthenon variable labels are "<base>_<sparse_id>".
// Our pool base name is "U" with IDs: 0 -> rho (scalar), 1 -> mom (vector<3>),
// 2 -> E (scalar). The typed names must therefore return "U_0", "U_1", and "U_2".
namespace U {
struct rho : public parthenon::variable_names::base_t<false> {
  template <class... Ts>
  KOKKOS_INLINE_FUNCTION rho(Ts &&...args)
      : parthenon::variable_names::base_t<false>(std::forward<Ts>(args)...) {}
  static std::string name() { return "U_0"; }
};
// Momentum is a 3-vector under sparse id 1
struct mom : public parthenon::variable_names::base_t<false, 3> {
  template <class... Ts>
  KOKKOS_INLINE_FUNCTION mom(Ts &&...args)
      : parthenon::variable_names::base_t<false, 3>(std::forward<Ts>(args)...) {}
  static std::string name() { return "U_1"; }
};
struct E : public parthenon::variable_names::base_t<false> {
  template <class... Ts>
  KOKKOS_INLINE_FUNCTION E(Ts &&...args)
      : parthenon::variable_names::base_t<false>(std::forward<Ts>(args)...) {}
  static std::string name() { return "U_2"; }
};
} // namespace U

std::shared_ptr<parthenon::StateDescriptor> Initialize(parthenon::ParameterInput *pin);

parthenon::TaskStatus ComputeFluxes(std::shared_ptr<parthenon::MeshBlockData<Real>> &rc);

parthenon::Real EstimateTimestepBlock(parthenon::MeshBlockData<Real> *rc);

} // namespace euler_sparse_example

#endif // EXAMPLE_EULER_SPARSE_EULER_PACKAGE_HPP_
