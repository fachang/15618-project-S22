//
//  MatMul.metal
//  MCNN iOS
//
//  Created by BerthCloud Chou on 2022/3/24.
//

#include <metal_stdlib>
#include <simd/simd.h>

// Including header shared between this Metal shader code and Swift/C code executing Metal API commands
#import "ShaderTypes.h"

using namespace metal;

kernel void linear_forward(device float *output [[ buffer(0) ]],
                           const device float *input [[ buffer(1) ]],
                           const device float *weight [[ buffer(2) ]],
                           const device float *bias [[ buffer(3) ]],
                           const device LinearLayerParams *params [[ buffer(4) ]],
                           uint2 thread_group_id [[ threadgroup_position_in_grid ]],
                           uint2 threads_per_group [[ threads_per_threadgroup ]],
                           uint2 thread_id [[ thread_position_in_threadgroup ]]) {
    
    uint row = thread_group_id.y * threads_per_group.y + thread_id.y;
    uint col = thread_group_id.x * threads_per_group.x + thread_id.x;
    
    if (row >= params->batch_size || col >= params->n_output_channel) {
        return;
    }
    
    float sum = 0;
    for (uint i = 0; i < params->n_input_channel; i++) {
        sum += input[row * params->n_input_channel + i] * weight[i * params->n_output_channel + col];
    }
    sum += ((params->bias) ? bias[col] : 0);
    output[row * params->n_output_channel + col] = sum;
}

kernel void reluForward(const device float *input [[ buffer(0) ]],
                        device float *output [[ buffer(1) ]],
                        uint idx [[ thread_position_in_grid ]]) {
    if(input[idx] <= 0) {
        output[idx] = 0;
    } else {
        output[idx] = input[idx];
    }
}
