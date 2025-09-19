//========================================================================================
#ifndef EXAMPLE_EULER_SPARSE_EULER_DRIVER_HPP_
#define EXAMPLE_EULER_SPARSE_EULER_DRIVER_HPP_

#include <parthenon/driver.hpp>
#include <memory>

namespace euler_sparse_example {
class EulerDriver : public parthenon::driver::prelude::MultiStageDriver {
 public:
  EulerDriver(parthenon::ParameterInput *pin, parthenon::ApplicationInput *app_in,
              parthenon::Mesh *pm)
      : parthenon::driver::prelude::MultiStageDriver(pin, app_in, pm) {}
  parthenon::driver::prelude::TaskCollection MakeTaskCollection(
      parthenon::driver::prelude::BlockList_t &blocks, const int stage) override;
};

void ProblemGenerator(parthenon::MeshBlock *pmb, parthenon::ParameterInput *pin);

parthenon::Packages_t ProcessPackages(
    std::unique_ptr<parthenon::ParameterInput> &pin);

} // namespace euler_sparse_example

#endif // EXAMPLE_EULER_SPARSE_EULER_DRIVER_HPP_
