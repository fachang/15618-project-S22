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

#endif /* BridgeTypes_h */

