//
//  LinearLayer.swift
//  MCNN iOS
//
//  Created by BerthCloud Chou on 2022/3/24.
//

import Foundation
import Metal

public class LinearLayer: ForwardableProtocol {
    private let mtlBundle: MTLBundle
    private let nInputFeatures: Int
    private let nOutputFeatures: Int
    
    public init(mtlBundle: MTLBundle, nInputFeatures: Int, nOutputFeatures: Int) {
        self.mtlBundle = mtlBundle
        self.nInputFeatures = nInputFeatures
        self.nOutputFeatures = nOutputFeatures
    }
    
    public func forward(input: Tensor) -> Tensor {
        let cmdBuffer = mtlBundle.mtlCommandQueue.makeCommandBuffer()
        let cmdEncoder = cmdBuffer?.makeComputeCommandEncoder()
        
        let mtlFunc = mtlBundle.mtlLibrary.makeFunction(name: "matmul")
        
        var computePipelineState: MTLComputePipelineState!
        do {
            computePipelineState = try mtlBundle.mtlDevice.makeComputePipelineState(function: mtlFunc!)
        } catch {
            print("Fail to create Pipeline.")
        }
        
        cmdEncoder?.setComputePipelineState(computePipelineState)
        
        let nthreadsPerBlock = MTLSize(width: 1, height: 1, depth: 1)
        let nblocks = MTLSize(width: 1, height: 1, depth: 1)
        cmdEncoder?.dispatchThreadgroups(nblocks, threadsPerThreadgroup: nthreadsPerBlock)
        cmdEncoder?.endEncoding()

        cmdBuffer?.commit()
        cmdBuffer?.waitUntilCompleted()
        return Tensor(nDim: nOutputFeatures)
    }
}
