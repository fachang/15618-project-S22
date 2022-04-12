//
//  Conv2DLayer.swift
//  MCNN iOS
//
//  Created by BerthCloud Chou on 2022/4/11.
//

import Foundation
import Metal

public class Conv2DLayer: NetworkModuleProtocol {
    private let mtlBundle: MTLBundle
    private let nInputFeatures: Int
    private let nOutputFeatures: Int
    private let gpu: Bool
    private var params: Tensor<DataType>
    
    public init(mtlBundle: MTLBundle, nInputFeatures: Int, nOutputFeatures: Int, gpu: Bool) {
        self.mtlBundle = mtlBundle
        self.nInputFeatures = nInputFeatures
        self.nOutputFeatures = nOutputFeatures
        self.gpu = gpu
        
        self.params = Tensor<DataType>(shape: [nInputFeatures, nOutputFeatures], initValue: 1)
    }
    
    public func forward(input: Tensor<DataType>) -> Tensor<DataType> {
        if (gpu) {
            return gpuForward(input: input);
        } else {
            return cpuForward(input: input);
        }
    }
    
    private func cpuForward(input: Tensor<DataType>) -> Tensor<DataType> {
        return TensorUtilsCPU.matMul(t1: input, t2: params)
    }
    
    private func gpuForward(input: Tensor<DataType>) -> Tensor<DataType> {
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
        return Tensor<DataType>(
            shape: [input.getShape()[0], nOutputFeatures], initValue: DataType.zero)
    }
}
