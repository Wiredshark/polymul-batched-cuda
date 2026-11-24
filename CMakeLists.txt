# Based on generated scaffold; extended to embed CUDA source and load NVRTC.
cmake_minimum_required(VERSION 3.20)
project(fherma_solution CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Release unless told otherwise: an answer measured at -O0 is not a measurement
# of the answer.
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

add_executable(solution main.cpp solve.cpp)
find_path(CUDA_HEADER_DIR cuda.h HINTS ${CUDA_INCLUDE_DIR}
  /usr/local/cuda/include /usr/local/cuda/targets/x86_64-linux/include REQUIRED)
file(READ "${CMAKE_CURRENT_SOURCE_DIR}/kernels.cu" KERNEL_SOURCE)
file(READ "${CMAKE_CURRENT_SOURCE_DIR}/explicit_crt.cuh" EXPLICIT_CRT_SOURCE)
string(APPEND KERNEL_SOURCE "\n#ifndef BATCHED_ONLY\n${EXPLICIT_CRT_SOURCE}\n#endif\n")
foreach(WEIGHTED_HEADER weighted_prepare.cuh weighted_ntt.cuh weighted_crt.cuh column_pipeline.cuh batched_prepare.cuh batched_crt.cuh batched_ntt.cuh batched_outer.cuh batched_prepfwd.cuh batched_prepfwd_tc.cuh batched_passorder.cuh batched_p1x.cuh batched_crt_lean.cuh)
  file(READ "${CMAKE_CURRENT_SOURCE_DIR}/${WEIGHTED_HEADER}" WEIGHTED_SOURCE)
  # Legacy single-product kernels are skipped by NVRTC when BATCHED_ONLY is defined (shorter init).
  if(WEIGHTED_HEADER MATCHES "^(weighted_ntt|weighted_crt|column_pipeline)\\.cuh$")
    string(APPEND KERNEL_SOURCE "\n#ifndef BATCHED_ONLY\n${WEIGHTED_SOURCE}\n#endif\n")
  else()
    string(APPEND KERNEL_SOURCE "\n${WEIGHTED_SOURCE}")
  endif()
endforeach()
configure_file(kernel_source.h.in kernel_source.h @ONLY)
# Exit-time sweep only (diag_sweep.h): CRT diet / P3 evict_last variant kernels appended to the sweep copy of the source, never to the production one.
file(READ "${CMAKE_CURRENT_SOURCE_DIR}/crtd_sweep.cuh" CRTD_SWEEP_SOURCE)
configure_file(crtd_sweep_src.h.in crtd_sweep_src.h @ONLY)
# Round 21 (clean): P3 / CRT / P1 variant kernels appended to the combo sweep copy of the source only.
file(READ "${CMAKE_CURRENT_SOURCE_DIR}/clean_sweep.cuh" CL_SWEEP_SOURCE)
configure_file(clean_sweep_src.h.in clean_sweep_src.h @ONLY)
target_include_directories(solution PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_BINARY_DIR} ${CUDA_HEADER_DIR})
target_link_libraries(solution PRIVATE ${CMAKE_DL_LIBS})
target_compile_options(solution PRIVATE -Wall -Wextra)

# Local-only throughput bench (not part of the official build target).
if(FHERMA_LOCAL_BENCH)
  add_executable(bench_batched ${FHERMA_LOCAL_BENCH} solve.cpp)
  target_include_directories(bench_batched PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_BINARY_DIR} ${CUDA_HEADER_DIR})
  target_link_libraries(bench_batched PRIVATE ${CMAKE_DL_LIBS})
endif()
