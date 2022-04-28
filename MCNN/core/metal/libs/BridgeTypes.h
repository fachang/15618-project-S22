//
//  BridgeTypes.h
//  MCNN Shared
//
//  Created by BerthCloud Chou on 2022/3/22.
//

//
//  Header containing types and enum constants shared between Metal shaders and Swift/ObjC source
//
#ifndef BridgeTypes_h
#define BridgeTypes_h

#ifdef __METAL_VERSION__
#define NS_ENUM(_type, _name) enum _name : _type _name; enum _name : _type
#define NSInteger metal::int32_t
#else
#import <Foundation/Foundation.h>
#endif

struct LinearLayerParams {
    uint batch_size;
    uint n_input_channel;
    uint n_output_channel;
    bool bias;
};

struct PoolingLayerParams {
    uint h_in;
    uint w_in;
    uint channel_size;
    uint pool_size;
    uint h_out;
    uint w_out;
    uint stride;
    uint padding;
    uint batchSize;
};

struct Conv2DLayerParams {
    uint batch_size;
    uint n_input_channels;
    uint n_output_channels;
    uint output_height;
    uint output_width;
    uint input_height;
    uint input_width;
    uint kernel_size;
    uint stride_height;
    uint stride_width;
    uint padding;
    bool bias;
    uint threadgroups_per_grid_dim4;
};

#endif /* BridgeTypes_h */

