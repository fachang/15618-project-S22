//
//  Conv2DLayer.metal
//  MCNN iOS
//
//  Created by BerthCloud Chou on 2022/4/18.
//

#include <metal_stdlib>

#import "BridgeTypes.h"

using namespace metal;

kernel void conv2d_forward(device float *output [[ buffer(0) ]],
                           const device float *input [[ buffer(1) ]],
                           const device float *weight [[ buffer(2) ]],
                           const device float *bias [[ buffer(3) ]],
                           const device Conv2DLayerParams *params [[ buffer(4) ]],
                           uint3 thread_group_id [[ threadgroup_position_in_grid ]],
                           uint3 threads_per_group [[ threads_per_threadgroup ]],
                           uint3 thread_id [[ thread_position_in_threadgroup ]]) {

    uint threadgroups_per_grid_dim4 = params->threadgroups_per_grid_dim4;
    
    // TODO: confirm which dimension order performs better
    uint output_col = (thread_group_id.z % threadgroups_per_grid_dim4) * threads_per_group.x + thread_id.x;
    uint output_row = (thread_group_id.z / threadgroups_per_grid_dim4) * threads_per_group.y + thread_id.y;
    uint output_channel_idx = thread_group_id.y;
    uint batch_idx = thread_group_id.x;

    if (batch_idx >= params->batch_size || output_channel_idx >= params->n_output_channels ||
            output_row >= params->output_height || output_col >= params->output_width) {
        return;
    }

    uint input_col_start = output_col * params->stride_width - params->padding;
    uint input_row_start = output_row * params->stride_height - params->padding;

    float sum = 0;
    for (uint c = 0; c < params->n_input_channels; c++) {
        for (uint i = 0; i < params->kernel_size; i++) {
            for (uint j = 0; j < params->kernel_size; j++) {
                if (input_row_start + i >= 0 && input_row_start + i < params->input_height &&
                        input_col_start + j >= 0 && input_col_start + j < params->input_width) {
                    sum += (input[
                                batch_idx * (params->n_input_channels * params->input_height *
                                             params->input_width) +
                                c * (params->input_height * params->input_width) +
                                (input_row_start + i) * (params->input_width) +
                                (input_col_start + j)
                            ] * weight[
                                output_channel_idx * (params->n_input_channels * params->kernel_size *                                   params->kernel_size) +
                                c * (params->kernel_size * params->kernel_size) +
                                (i) * (params->kernel_size) +
                                (j)
                            ]);
                }
            }
        }
    }
    
    sum += ((params->bias) ? bias[output_channel_idx] : 0);

    output[
        batch_idx * (params->n_output_channels * params->output_height * params->output_width) +
        output_channel_idx * (params->output_height * params->output_width) +
        output_row * (params->output_width) +
        output_col] = sum;
}

kernel void conv2d_img2col(device float *output [[ buffer(0) ]],
                           const device float *input [[ buffer(1) ]],
                           const device Conv2DImg2ColParams *params [[ buffer(4) ]],
                           uint thread_id [[ thread_position_in_grid ]]) {

    if (thread_id >= params->n_input_channels * params->output_width * params->output_height) {
        return;
    }
    
    uint input_channel_idx = thread_id / (params->output_height * params->output_width);
    uint output_flatten_idx = thread_id % (params->output_height * params->output_width);
    uint output_col = output_flatten_idx % params->output_width;
    uint output_row = output_flatten_idx / params->output_width;
    uint input_col_start = output_col * params->stride_width - params->padding;
    uint input_row_start = output_row * params->stride_height - params->padding;
    uint batch_idx = params->batch_id;

    for (uint i = 0; i < params->kernel_size; i++) {
        for (uint j = 0; j < params->kernel_size; j++) {
            float val = 0.0f;
            if (input_row_start + i >= 0 && input_row_start + i < params->input_height &&
                    input_col_start + j >= 0 && input_col_start + j < params->input_width) {
                val = input[
                    batch_idx * (params->n_input_channels * params->input_height *
                                 params->input_width) +
                    input_channel_idx * (params->input_height * params->input_width) +
                    (input_row_start + i) * (params->input_width) +
                    (input_col_start + j)
                ];
            }
            output[thread_id] = val;
        }
    }
}
