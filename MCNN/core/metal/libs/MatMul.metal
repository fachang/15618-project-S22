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

kernel void matmul_tiling(device float *output [[ buffer(0) ]],
                          const device float *mat1 [[ buffer(1) ]],
                          const device float *mat2 [[ buffer(2) ]],
                          const device float *bias [[ buffer(3) ]],
                          const device MatMulParams *params [[ buffer(4) ]],
                          uint2 groups_per_grid [[ threadgroups_per_grid ]],
                          uint2 thread_group_id [[ threadgroup_position_in_grid ]],
                          uint2 threads_per_group [[ threads_per_threadgroup ]],
                          uint2 thread_id [[ thread_position_in_threadgroup ]]) {

    assert(threads_per_group.x == MM_TILE_W && thread_group_id.y == MM_TILE_W);

    threadgroup float mat1_tile[MM_TILE_W * MM_TILE_W];
    threadgroup float mat2_tile[MM_TILE_W * MM_TILE_W];
    
    uint col = thread_group_id.x * threads_per_group.x + thread_id.x;
    uint row = thread_group_id.y * threads_per_group.y + thread_id.y;

    if (row >= MM_TILE_W * groups_per_grid.y || col >= MM_TILE_W * groups_per_grid.x) {
        return;
    }
    
    float sum = 0;
    uint mat1_width_n_tiles = (params->mat1_width + MM_TILE_W - 1) / MM_TILE_W;
    for (uint t = 0; t < mat1_width_n_tiles; t++) {
        uint tmp_col = t * MM_TILE_W + thread_id.x;
        mat1_tile[thread_id.y * MM_TILE_W + thread_id.x] = (
            (row < params->mat1_height && tmp_col < params->mat1_width) ?
            mat1[row * params->mat1_width + tmp_col] : 0
        );
        uint tmp_row = t * MM_TILE_W + thread_id.y;
        mat2_tile[thread_id.y * MM_TILE_W + thread_id.x] = (
            (tmp_row < params->mat1_width && col < params->mat2_width) ?
            mat2[tmp_row * params->mat2_width + col] : 0
        );
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        for (uint i = 0; i < MM_TILE_W; i++) {
            sum += (mat1_tile[thread_id.y * MM_TILE_W + i] * mat2_tile[i * MM_TILE_W + thread_id.x]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (row < params->mat1_height && col < params->mat2_width) {
        sum += ((params->mat1_bias) ? bias[row] : ((params->mat2_bias) ? bias[col] : 0));
        output[params->output_offset + row * params->mat2_width + col] = sum;
    }
}
