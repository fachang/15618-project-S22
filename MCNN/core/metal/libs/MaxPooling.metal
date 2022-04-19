//
//  MaxPooling.metal
//  MCNN iOS
//
//  Created by Fong-An Chang on 2022/4/18.
//


#include <metal_stdlib>
#include <simd/simd.h>
#import "BridgeTypes.h"
using namespace metal;

kernel void max_pooling(
                        device float *output [[ buffer(0) ]],
                        const device float *input [[ buffer(1) ]],
                        const device PoolingLayerParams *poolingParams [[ buffer(2) ]],
                        uint3 thread_group_id [[ threadgroup_position_in_grid ]],
                        uint3 threads_per_group [[ threads_per_threadgroup ]],
                        uint3 thread_id [[ thread_position_in_threadgroup ]]) {
    int n, m, h, w, p, q;
    int w_in = poolingParams->w_in;
    int h_in = poolingParams->h_in;
    int w_out = poolingParams->w_out;
    int h_out = poolingParams->h_out;
    int stride = poolingParams->stride;
    int pool_size = poolingParams->pool_size;
    int channel_idx = poolingParams->channel_size;
    


    int W_grid = ((w_out+threads_per_group.x-1)/threads_per_group.x);
    n = thread_group_id.x;
    m = thread_group_id.y;
    h = (thread_group_id.z / W_grid)*threads_per_group.y + thread_id.y;
    w = (thread_group_id.z % W_grid)*threads_per_group.x + thread_id.x;

    float acc = FLT_MIN;

    int base_idx_h = h * stride;
    int base_idx_w = w * stride;
    for (p = 0; p < pool_size; p++) {
        for (q = 0; q < pool_size; q++)
            if(h < h_out && w < w_out)
                acc = max(acc, input[n*(channel_idx*h_in*w_in)+
                                     m*(h_in*w_in)+
                                     (base_idx_h + p)*(w_in)+
                                     (base_idx_w + q)]);
    }
    
    if(h < h_out && w < w_out) {
        output[n*(channel_idx*h_out*w_out)+
               m*(h_out*w_out)+
               h*(w_out)+
               w] = acc;
    }
}


