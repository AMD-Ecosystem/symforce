/* ----------------------------------------------------------------------------
 * SymForce - Copyright 2025, Skydio, Inc.
 * Copyright (c) 2026 Advanced Micro Devices, Inc.
 * This source code is under the Apache 2.0 license found in the LICENSE file.
 *
 * Author: Jeff Daily <jeff.daily@amd.com>
 * ---------------------------------------------------------------------------- */

#pragma once

// CUDA-to-HIP compatibility header for ROCm/HIP builds.
// Provides the CUDA API spellings the project uses via aliases to HIP equivalents.

#if defined(USE_HIP) || defined(__HIP_PLATFORM_AMD__)

#include <hip/hip_cooperative_groups.h>
#include <hip/hip_runtime.h>
#include <hipcub/hipcub.hpp>

// CUDA runtime API -> HIP runtime API
#define cudaMalloc hipMalloc
#define cudaFree hipFree
#define cudaMemset hipMemset
#define cudaMemcpy hipMemcpy
#define cudaMemcpyDeviceToDevice hipMemcpyDeviceToDevice
#define cudaMemcpyDeviceToHost hipMemcpyDeviceToHost
#define cudaMemcpyHostToDevice hipMemcpyHostToDevice
#define cudaError_t hipError_t
#define cudaSuccess hipSuccess
#define cudaStream_t hipStream_t
#define cudaDeviceSynchronize hipDeviceSynchronize
#define cudaGetLastError hipGetLastError
#define cudaGetErrorString hipGetErrorString
#define cudaSetDevice hipSetDevice
#define cudaGetDevice hipGetDevice
#define cudaPointerGetAttributes hipPointerGetAttributes
#define cudaPointerAttributes hipPointerAttribute_t
#define cudaGetDeviceCount hipGetDeviceCount
#define cudaFuncAttributeMaxDynamicSharedMemorySize hipFuncAttributeMaxDynamicSharedMemorySize

// hipFuncSetAttribute takes the kernel as const void*; CUDA also accepts the kernel itself.
template <typename T>
inline hipError_t cudaFuncSetAttribute(T* func, hipFuncAttribute attr, int value) {
  return hipFuncSetAttribute(reinterpret_cast<const void*>(func), attr, value);
}

// HIP shared-memory atomics are block-scoped by definition (no inter-block visibility),
// so atomicAdd_block is equivalent to atomicAdd on shared memory.
#define atomicAdd_block atomicAdd

// CUB -> hipCUB namespace
namespace cub = hipcub;

#else  // CUDA path

#include <cuda_runtime.h>

// Host-only translation units (e.g. code using the generated headers) get just the runtime API.
#if defined(__CUDACC__)
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cub/cub.cuh>

namespace cg = cooperative_groups;
#endif

#endif
