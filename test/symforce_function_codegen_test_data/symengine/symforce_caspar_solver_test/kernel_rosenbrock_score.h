#pragma once

#include "cuda_to_hip.h"
#include "shared_indices.h"

namespace caspar {

void RosenbrockScore(double* x, unsigned int x_num_alloc, SharedIndex* x_indices,
                     double* const out_rTr, size_t problem_size);

}  // namespace caspar