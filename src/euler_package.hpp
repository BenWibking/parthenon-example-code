//========================================================================================
// Minimal Euler (hyperbolic) package using sparse fields
//========================================================================================
#ifndef EXAMPLE_EULER_SPARSE_EULER_PACKAGE_HPP_
#define EXAMPLE_EULER_SPARSE_EULER_PACKAGE_HPP_

#include <memory>

#include <parthenon/package.hpp>

namespace euler_sparse_example {

using namespace parthenon::package::prelude;

std::shared_ptr<parthenon::StateDescriptor> Initialize(parthenon::ParameterInput *pin);

parthenon::TaskStatus ComputeFluxes(std::shared_ptr<parthenon::MeshBlockData<Real>> &rc);

parthenon::Real EstimateTimestepBlock(parthenon::MeshBlockData<Real> *rc);

} // namespace euler_sparse_example

#endif // EXAMPLE_EULER_SPARSE_EULER_PACKAGE_HPP_
