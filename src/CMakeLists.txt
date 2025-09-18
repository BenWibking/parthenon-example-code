#=========================================================================================
# (C) 2025. Triad National Security, LLC. All rights reserved.
#=========================================================================================

get_property(DRIVER_LIST GLOBAL PROPERTY DRIVERS_USED_IN_TESTS)
if( "euler_sparse-example" IN_LIST DRIVER_LIST OR NOT PARTHENON_DISABLE_EXAMPLES)
  add_executable(euler_sparse-example
    euler_driver.cpp
    euler_driver.hpp
    euler_package.cpp
    euler_package.hpp
    main.cpp
    parthenon_app_inputs.cpp
  )
  target_link_libraries(euler_sparse-example PRIVATE Parthenon::parthenon)
  lint_target(euler_sparse-example)
endif()

