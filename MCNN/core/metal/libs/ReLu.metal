//
//  MatMul.metal
//  MCNN iOS
//
//  Created by BerthCloud Chou on 2022/3/24.
//

#include <metal_stdlib>
#include <simd/simd.h>

using namespace metal;

kernel void relu_forward(device float *output [[ buffer(0) ]],
                         const device float *input [[ buffer(1) ]],
                         uint idx [[ thread_position_in_grid ]]) {
    if(input[idx] <= 0) {
        output[idx] = 0;
    } else {
        output[idx] = input[idx];
    }
}
