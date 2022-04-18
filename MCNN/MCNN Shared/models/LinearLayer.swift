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
        
        let mtlFunc = mtlBundle.mtlLibrary.makeFunction(name: "linear_forward")
        
        var computePipelineState: MTLComputePipelineState!
        do {
            computePipelineState = try mtlBundle.mtlDevice.makeComputePipelineState(function: mtlFunc!)
        } catch {
            print("Fail to create Pipeline.")
        }
        
        cmdEncoder?.setComputePipelineState(computePipelineState)
        
        let batchSize = input.getShape()[0]
        
        let metalDevice = MTLCreateSystemDefaultDevice();
        let outputDataCPU = Tensor<DataType>(shape: [batchSize, nOutputFeatures], initValue: 0);
        let outputDataSize = MemoryLayout<DataType>.stride * outputDataCPU.data.count;
        let outputDataDevice = metalDevice!.makeBuffer(
            bytes: outputDataCPU.data, length: outputDataSize,
            options: MTLResourceOptions.cpuCacheModeWriteCombined)
        let inputDataDevice = metalDevice!.makeBuffer(
            bytes: input.data, length: MemoryLayout<DataType>.stride * input.data.count,
            options: MTLResourceOptions.cpuCacheModeWriteCombined)
        let paramsDataDevice = metalDevice!.makeBuffer(
            bytes: params.data, length: MemoryLayout<DataType>.stride * params.data.count,
            options: MTLResourceOptions.cpuCacheModeWriteCombined)
        let biasDataDevice = (bias == nil) ? nil : metalDevice!.makeBuffer(
            bytes: bias!.data, length: MemoryLayout<DataType>.stride * bias!.data.count,
            options: MTLResourceOptions.cpuCacheModeWriteCombined)
        var layerParamsCPU = LinearLayerParams(
            batch_size: UInt32(batchSize), n_input_channel: UInt32(nInputFeatures),
            n_output_channel: UInt32(nOutputFeatures), bias: (bias != nil));
        let layerParamsDevice = metalDevice!.makeBuffer(
            bytes: &layerParamsCPU, length: MemoryLayout<LinearLayerParams>.stride,
            options: MTLResourceOptions.cpuCacheModeWriteCombined)
        
        cmdEncoder?.setBuffer(outputDataDevice, offset: 0, index: 0)
        cmdEncoder?.setBuffer(inputDataDevice, offset: 0, index: 1)
        cmdEncoder?.setBuffer(paramsDataDevice, offset: 0, index: 2)
        cmdEncoder?.setBuffer(biasDataDevice, offset: 0, index: 3)
        cmdEncoder?.setBuffer(layerParamsDevice, offset: 0, index: 4)
        
        let nthreadsPerBlock = MTLSize(width: 32, height: 32, depth: 1)
        let nblocks = MTLSize(width: (nOutputFeatures + 31) / 32, height: batchSize, depth: 1)
        cmdEncoder?.dispatchThreadgroups(nblocks, threadsPerThreadgroup: nthreadsPerBlock)
        
        cmdEncoder?.endEncoding()

        cmdBuffer?.commit()
        cmdBuffer?.waitUntilCompleted()
        
        let outputDataDeviceBuf = NSData(bytesNoCopy: (outputDataDevice!.contents()),
                                         length: outputDataSize, freeWhenDone: false)
        outputDataDeviceBuf.getBytes(&outputDataCPU.data, length: outputDataSize)
        
        return outputDataCPU
    }
}
