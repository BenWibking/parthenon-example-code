//========================================================================================
#ifndef EXAMPLE_EULER_SPARSE_EULER_DRIVER_HPP_
#define EXAMPLE_EULER_SPARSE_EULER_DRIVER_HPP_

#include <parthenon/driver.hpp>
#include <memory>

namespace euler_sparse_example {
using namespace parthenon::driver::prelude;

class EulerDriver : public MultiStageDriver {
 public:
  EulerDriver(parthenon::ParameterInput *pin, parthenon::ApplicationInput *app_in,
              parthenon::Mesh *pm)
      : MultiStageDriver(pin, app_in, pm) {}
  TaskCollection MakeTaskCollection(BlockList_t &blocks, const int stage) override;
};

void ProblemGenerator(parthenon::MeshBlock *pmb, parthenon::ParameterInput *pin);

parthenon::Packages_t ProcessPackages(
    std::unique_ptr<parthenon::ParameterInput> &pin);

} // namespace euler_sparse_example

#endif // EXAMPLE_EULER_SPARSE_EULER_DRIVER_HPP_
