//
//  ReLu.swift
//  MCNN iOS
//
//  Created by BerthCloud Chou on 2022/4/12.
//

import Foundation
import Metal

public class ReLu: NetworkModuleProtocol {

    private static let GROUP_SIZE: Int = 32 * 32

    private let gpu: Bool
    
    public init(gpu: Bool) {
        self.gpu = gpu
    }
    
    public func forward(input: Tensor<DataType>) -> Tensor<DataType> {
        if (gpu) {
            return gpuForward(input: input)
        } else {
            return cpuForward(input: input)
        }
    }
    
    private func cpuForward(input: Tensor<DataType>) -> Tensor<DataType> {
        for i in 0..<input.data.count {
            if (input.data[i] < 0) {
                input.data[i] = 0
            }
        }
        return input
    }
    
    private func gpuForward(input: Tensor<DataType>) -> Tensor<DataType> {
        assert(input.dataGPU != nil)
        
        let cmdBuffer = MTLCommons.mtlCommandQueue.makeCommandBuffer()!
        let cmdEncoder = cmdBuffer.makeComputeCommandEncoder()!
        assert(MTLUtils.addComputePipeline(cmdEncoder: cmdEncoder,
                                           kernelLibrary: MTLCommons.defaultLib,
                                           kernelFuncName: "relu_forward") == true)

        let result = Tensor<DataType>(shape: input.getShape(), initValue: 0)
        result.copyToGPU()

        cmdEncoder.setBuffer(result.dataGPU, offset: 0, index: 0)
        cmdEncoder.setBuffer(input.dataGPU, offset: 0, index: 1)

        let nthreadsPerBlock = MTLSize(width: ReLu.GROUP_SIZE, height: 1, depth: 1)
        let nblocks = MTLSize(
            width: (result.data.count + ReLu.GROUP_SIZE - 1) / ReLu.GROUP_SIZE,
            height: 1, depth: 1)
        cmdEncoder.dispatchThreadgroups(nblocks, threadsPerThreadgroup: nthreadsPerBlock)
        
        cmdEncoder.endEncoding()

        cmdBuffer.commit()
        cmdBuffer.waitUntilCompleted()

        return result
    }
}
