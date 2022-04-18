//
//  MTLUtils.swift
//  MCNN iOS
//
//  Created by BerthCloud Chou on 2022/4/18.
//

import Foundation
import Metal

internal class MTLUtils {
    internal static func copyToGPU(dataPtr: UnsafeRawPointer, size: Int) -> MTLBuffer? {
        return MTLCommons.mtlDevice.makeBuffer(
            bytes: dataPtr, length: size,
            options: MTLResourceOptions.cpuCacheModeWriteCombined)
    }
    
    internal static func addComputePipeline(cmdEncoder: MTLComputeCommandEncoder,
                                          kernelLibrary: MTLLibrary, kernelFuncName: String) -> Bool {
        let kernelFunc = kernelLibrary.makeFunction(name: kernelFuncName)
        if (kernelFunc == nil) {
            return false;
        }

        var computePipelineState: MTLComputePipelineState!
        do {
            computePipelineState = try MTLCommons.mtlDevice.makeComputePipelineState(function: kernelFunc!)
        } catch {
            return false;
        }
        
        cmdEncoder.setComputePipelineState(computePipelineState)
        return true
    }
}
