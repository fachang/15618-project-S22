//
//  LinearLayer.swift
//  MCNN iOS
//
//  Created by BerthCloud Chou on 2022/3/24.
//

import Foundation
import Metal

public class LinearLayer: NetworkModuleProtocol {
    private let mtlBundle: MTLBundle
    private let nInputFeatures: Int
    private let nOutputFeatures: Int
    private let gpu: Bool
    private var params: Tensor<DataType>
    private var bias: Tensor<DataType>?
    
    public init(mtlBundle: MTLBundle, nInputFeatures: Int, nOutputFeatures: Int, bias: Bool,
                gpu: Bool) {
        self.mtlBundle = mtlBundle
        self.nInputFeatures = nInputFeatures
        self.nOutputFeatures = nOutputFeatures
        self.gpu = gpu
        
        self.params = Tensor<DataType>(shape: [nInputFeatures, nOutputFeatures], initValue: 1)
        self.bias = (bias) ? Tensor<DataType>(shape: [1, nOutputFeatures], initValue: 1) : nil
    }
    
    public func forward(input: Tensor<DataType>) -> Tensor<DataType> {
        if (gpu) {
            return gpuForward(input: input);
        } else {
            return cpuForward(input: input);
        }
    }
    
    private func cpuForward(input: Tensor<DataType>) -> Tensor<DataType> {
        let result: Tensor<DataType> = TensorUtilsCPU.matMul(t1: input, t2: params)
        if bias != nil {
            let batchSize: Int = result.getShape()[0]
            for i in 0..<batchSize {
                for j in 0..<nOutputFeatures {
                    result.setData(
                        idx: [i, j],
                        value: result.getData(idx: [i, j]) + bias!.getData(idx: [0, j]))
                }
            }
        }
        return result
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
