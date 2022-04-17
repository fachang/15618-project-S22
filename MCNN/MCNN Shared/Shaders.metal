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

kernel void matmul() {
    return;
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
