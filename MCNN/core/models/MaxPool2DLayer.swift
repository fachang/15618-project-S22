//
//  MaxPool2DLayer.swift
//  MCNN iOS
//
//  Created by BerthCloud Chou on 2022/4/12.
//

import Foundation
import Metal

public class MaxPool2DLayer: NetworkModuleProtocol {
    private static let GROUP_W: Int = 32
    private let gpu: Bool
    
    private let kernelSize: Int
    private let strideHeight: Int
    private let strideWidth: Int
    private let padding: Int
    
    public init(kernelSize: Int, strideHeight: Int, strideWidth: Int,
                padding: Int, gpu: Bool) {
        self.gpu = gpu
        
        self.kernelSize = kernelSize
        self.strideHeight = strideHeight
        self.strideWidth = strideWidth
        self.padding = padding
    }
    
    public func forward(input: Tensor<DataType>) -> Tensor<DataType> {
        if (gpu) {
            return gpuForward(input: input)
        } else {
            return cpuForward(input: input)
        }
    }
    
    private func cpuForward(input: Tensor<DataType>) -> Tensor<DataType> {
        let inputShape: [Int] = input.getShape()
        let batchSize: Int = inputShape[0]
        let nInputChannels: Int = inputShape[1]
        let inputHeight: Int = inputShape[2]
        let inputWidth: Int = inputShape[3]
        let outputHeight: Int = Int(floor(Double(inputHeight + 2 * padding - kernelSize) / Double(strideHeight))) + 1
        let outputWidth: Int = Int(floor(Double(inputWidth + 2 * padding - kernelSize) / Double(strideWidth))) + 1
        let result: Tensor<DataType> = Tensor<DataType>(
            shape: [batchSize, nInputChannels, outputHeight, outputWidth],
            initValue: DataType.zero);

        for batchIdx in 0..<batchSize {
            for channelIdx in 0..<nInputChannels {
                for i in 0..<outputHeight {
                    for j in 0..<outputWidth {
                        let inputHeightStart: Int = i * strideHeight - padding
                        let inputWidthStart: Int = j * strideWidth - padding
                        var tmpMax: DataType? = nil
                        for m in 0..<kernelSize {
                            for n in 0..<kernelSize {
                                let inputM = inputHeightStart + m
                                let inputN = inputWidthStart + n
                                let tmpInput: DataType;
                                if (inputM >= 0 && inputM < inputHeight &&
                                        inputN >= 0 && inputN < inputWidth) {
                                    tmpInput = input.getData(
                                        idx: [batchIdx, channelIdx, inputM, inputN])
                                    if (tmpMax == nil || tmpMax! < tmpInput) {
                                        tmpMax = tmpInput
                                    }
                                }
                            }
                        }
                        result.setData(idx: [batchIdx, channelIdx, i, j], value: tmpMax!)
                    }
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
                                            kernelFuncName: "max_pooling") == true)
        let inputShape: [Int] = input.getShape()
        let batchSize: Int = inputShape[0]
        let nInputChannels: Int = inputShape[1]
        let inputHeight: Int = inputShape[2]
        let inputWidth: Int = inputShape[3]
        let outputHeight: Int = Int(floor(Double(inputHeight + 2 * padding - kernelSize) / Double(strideHeight))) + 1
        let outputWidth: Int = Int(floor(Double(inputWidth + 2 * padding - kernelSize) / Double(strideWidth))) + 1
        
        let result: Tensor<DataType> = Tensor<DataType>(
            shape: [batchSize, nInputChannels, outputHeight, outputWidth],
            initValue: DataType.zero);

        result.copyToGPU()

        var layerParamsGPU = PoolingLayerParams(h_in: UInt32(inputHeight),
                                                w_in: UInt32(inputWidth),
                                                channel_size: UInt32(nInputChannels),
                                                pool_size: UInt32(kernelSize),
                                                h_out: UInt32(outputHeight),
                                                w_out: UInt32(outputWidth),
                                                stride: UInt32(strideWidth),
                                                padding: UInt32(padding),
                                                batchSize: UInt32(batchSize));
        let layerParamsDevice = MTLUtils.copyToGPU(
            dataPtr: &layerParamsGPU, size: MemoryLayout<PoolingLayerParams>.stride)

        cmdEncoder.setBuffer(result.dataGPU, offset: 0, index: 0)
        cmdEncoder.setBuffer(input.dataGPU, offset: 0, index: 1)
        cmdEncoder.setBuffer(layerParamsDevice, offset: 0, index: 2)
        
        let nthreadsPerBlock = MTLSize(
            width: MaxPool2DLayer.GROUP_W, height: MaxPool2DLayer.GROUP_W, depth: 1)
        let bz = Int(((outputWidth+MaxPool2DLayer.GROUP_W-1)/MaxPool2DLayer.GROUP_W)) * Int(((outputHeight+MaxPool2DLayer.GROUP_W-1)/MaxPool2DLayer.GROUP_W));

        let nblocks = MTLSize(
            width: batchSize,
            height: nInputChannels, depth: bz)
        cmdEncoder.dispatchThreadgroups(nblocks, threadsPerThreadgroup: nthreadsPerBlock)
        
        cmdEncoder.endEncoding()

        cmdBuffer.commit()
        cmdBuffer.waitUntilCompleted()

        return result
    }
}
