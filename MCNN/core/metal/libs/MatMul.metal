//
//  MatMul.metal
//  MCNN iOS
//
//  Created by BerthCloud Chou on 2022/4/28.
//

#include <metal_stdlib>
#include <simd/simd.h>

// Including header shared between this Metal shader code and Swift/C code executing Metal API commands
#import "BridgeTypes.h"

using namespace metal;

kernel void matmul(device float *output [[ buffer(0) ]],
                   const device float *mat1 [[ buffer(1) ]],
                   const device float *mat2 [[ buffer(2) ]],
                   const device float *bias [[ buffer(3) ]],
                   const device MatMulParams *params [[ buffer(4) ]],
                   uint2 thread_group_id [[ threadgroup_position_in_grid ]],
                   uint2 threads_per_group [[ threads_per_threadgroup ]],
                   uint2 thread_id [[ thread_position_in_threadgroup ]]) {
    
    uint col = thread_group_id.x * threads_per_group.x + thread_id.x;
    uint row = thread_group_id.y * threads_per_group.y + thread_id.y;
    
    if (row >= params->mat1_height || col >= params->mat2_width) {
        return;
    }
    
    float sum = 0;
    for (uint i = 0; i < params->mat1_width; i++) {
        sum += mat1[row * params->mat1_width + i] * mat2[i * params->mat2_width + col];
    }
    sum += ((params->mat1_bias) ? bias[row] : ((params->mat2_bias) ? bias[col] : 0));
    output[params->output_offset + row * params->mat2_width + col] = sum;
}
