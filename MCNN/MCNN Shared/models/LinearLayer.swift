//
//  LinearLayer.swift
//  MCNN iOS
//
//  Created by BerthCloud Chou on 2022/3/24.
//

import Foundation
import Metal

public class LinearLayer: NetworkModuleProtocol {
    private static let GROUP_W: Int = 32

    private let nInputFeatures: Int
    private let nOutputFeatures: Int
    private let gpu: Bool
    private var params: Tensor<DataType>
    private var bias: Tensor<DataType>?
    
    public init(nInputFeatures: Int, nOutputFeatures: Int, bias: Bool, gpu: Bool) {
        self.nInputFeatures = nInputFeatures
        self.nOutputFeatures = nOutputFeatures
        self.gpu = gpu
        
        self.params = Tensor<DataType>(shape: [nInputFeatures, nOutputFeatures], initValue: 1)
        self.params.copyToGPU()

        if (bias) {
            self.bias = Tensor<DataType>(shape: [1, nOutputFeatures], initValue: 1)
            self.bias!.copyToGPU()
        }
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
        assert(input.dataGPU != nil)
        
        let cmdBuffer = MTLCommons.mtlCommandQueue.makeCommandBuffer()!
        let cmdEncoder = cmdBuffer.makeComputeCommandEncoder()!

        assert(MTLUtils.addComputePipeline(cmdEncoder: cmdEncoder,
                                            kernelLibrary: MTLCommons.defaultLib,
                                            kernelFuncName: "linear_forward") == true)

        let batchSize = input.getShape()[0]
        let result = Tensor<DataType>(shape: [batchSize, nOutputFeatures], initValue: 0)
        result.copyToGPU()
        var layerParamsCPU = LinearLayerParams(
            batch_size: UInt32(batchSize), n_input_channel: UInt32(nInputFeatures),
            n_output_channel: UInt32(nOutputFeatures), bias: (bias != nil))
        let layerParamsDevice = MTLUtils.copyToGPU(
            dataPtr: &layerParamsCPU, size: MemoryLayout<LinearLayerParams>.stride)

        cmdEncoder.setBuffer(result.dataGPU, offset: 0, index: 0)
        cmdEncoder.setBuffer(input.dataGPU, offset: 0, index: 1)
        cmdEncoder.setBuffer(params.dataGPU, offset: 0, index: 2)
        cmdEncoder.setBuffer((bias == nil) ? nil : bias!.dataGPU, offset: 0, index: 3)
        cmdEncoder.setBuffer(layerParamsDevice, offset: 0, index: 4)
        
        let nthreadsPerBlock = MTLSize(
            width: LinearLayer.GROUP_W, height: LinearLayer.GROUP_W, depth: 1)
        let nblocks = MTLSize(
            width: (nOutputFeatures + LinearLayer.GROUP_W - 1) / LinearLayer.GROUP_W,
            height: batchSize, depth: 1)
        cmdEncoder.dispatchThreadgroups(nblocks, threadsPerThreadgroup: nthreadsPerBlock)
        
        cmdEncoder.endEncoding()

        cmdBuffer.commit()
        cmdBuffer.waitUntilCompleted()

        return result
    }
}
