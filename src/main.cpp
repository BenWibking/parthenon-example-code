//========================================================================================
#include "driver/driver.hpp"
#include "euler_driver.hpp"

int main(int argc, char *argv[]) {
  using parthenon::DriverStatus;
  using parthenon::ParthenonManager;
  using parthenon::ParthenonStatus;

  ParthenonManager pman;
  auto status = pman.ParthenonInitEnv(argc, argv);
  if (status != ParthenonStatus::ok) return static_cast<int>(status);
  { // scope to destruct driver before Finalize
    pman.ParthenonInitPackagesAndMesh();
    pman.app_input->ProblemGenerator = euler_sparse_example::ProblemGenerator;
    euler_sparse_example::EulerDriver driver(pman.pinput.get(), pman.app_input.get(),
                                             pman.pmesh.get());
    auto driver_status = driver.Execute();
    if (driver_status != DriverStatus::complete)
      return static_cast<int>(driver_status);
  }
  pman.ParthenonFinalize();
  return 0;
}
