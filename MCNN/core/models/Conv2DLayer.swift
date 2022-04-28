//
//  Conv2DLayer.swift
//  MCNN iOS
//
//  Created by BerthCloud Chou on 2022/4/11.
//

import Foundation
import Metal

public class Conv2DLayerNaive: NetworkModuleProtocol {
    private static let GROUP_W: Int = 32
    
    private let nInputChannels: Int
    private let nOutputChannels: Int
    private let gpu: Bool
    
    private let kernelSize: Int
    private let strideHeight: Int
    private let strideWidth: Int
    private let padding: Int
    private let paddingMode: PaddingMode
    
    private let kernels: Tensor<DataType>
    private let bias: Tensor<DataType>?
    
    public init(nInputChannels: Int, nOutputChannels: Int, bias: Bool,
                kernelSize: Int, strideHeight: Int, strideWidth: Int,
                padding: Int, paddingMode: PaddingMode, gpu: Bool,
                initKernels: [DataType]? = nil, initBias: [DataType]? = nil) {
        self.nInputChannels = nInputChannels
        self.nOutputChannels = nOutputChannels
        self.gpu = gpu

        self.kernelSize = kernelSize
        self.strideHeight = strideHeight
        self.strideWidth = strideWidth
        self.padding = padding
        self.paddingMode = paddingMode

        let kernelShape = [nOutputChannels, nInputChannels, kernelSize, kernelSize]
        if (initKernels == nil) {
            self.kernels = Tensor<DataType>(
                shape: kernelShape, initValue: 1)
        } else {
            self.kernels = Tensor<DataType>(
                shape: kernelShape, data: initKernels!)
        }
        if (gpu) {
            self.kernels.copyToGPU()
        }
        
        if (!bias) {
            self.bias = nil;
        } else {
            let biasShape = [1, nOutputChannels];
            if (initBias == nil) {
                self.bias = Tensor<DataType>(
                    shape: biasShape, initValue: 1)
            } else {
                self.bias = Tensor<DataType>(
                    shape: biasShape, data: initBias!)
            }
            if (gpu) {
                self.bias!.copyToGPU()
            }
        }
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
        let inputHeight: Int = inputShape[2]
        let inputWidth: Int = inputShape[3]
        let outputHeight: Int = Int(floor(Double(inputHeight + 2 * padding - kernelSize) / Double(strideHeight))) + 1
        let outputWidth: Int = Int(floor(Double(inputWidth + 2 * padding - kernelSize) / Double(strideWidth))) + 1
        let result: Tensor<DataType> = Tensor<DataType>(
            shape: [batchSize, nOutputChannels, outputHeight, outputWidth],
            initValue: DataType.zero);

        for batchIdx in 0..<batchSize {
            for outChannelIdx in 0..<nOutputChannels {
                for i in 0..<outputHeight {
                    for j in 0..<outputWidth {
                        let inputHeightStart: Int = i * strideHeight - padding
                        let inputWidthStart: Int = j * strideWidth - padding
                        var tmpConv: DataType = DataType.zero
                        for inChannelIdx in 0..<nInputChannels {
                            for m in 0..<kernelSize {
                                for n in 0..<kernelSize {
                                    let inputM = inputHeightStart + m
                                    let inputN = inputWidthStart + n
                                    let tmpInput: DataType;
                                    if (inputM >= 0 && inputM < inputHeight &&
                                            inputN >= 0 && inputN < inputWidth) {
                                        tmpInput = input.getData(
                                            idx: [batchIdx, inChannelIdx, inputM, inputN])
                                    } else {
                                        // paddingMode == PaddingMode.zeros
                                        tmpInput = DataType.zero
                                    }
                                    tmpConv += (
                                        tmpInput * kernels.getData(idx: [outChannelIdx, inChannelIdx, m, n])
                                    )
                                }
                            }
                        }
                        if (bias != nil) {
                            tmpConv += bias!.getData(idx: [0, outChannelIdx])
                        }
                        result.setData(idx: [batchIdx, outChannelIdx, i, j], value: tmpConv)
                    }
                }
            }
        }
        
        return result
    }
    
    private func gpuForward(input: Tensor<DataType>) -> Tensor<DataType> {
        assert(input.dataGPU != nil)

        let inputShape: [Int] = input.getShape()
        let batchSize: Int = inputShape[0]
        let inputHeight: Int = inputShape[2]
        let inputWidth: Int = inputShape[3]
        let outputHeight: Int = Int(floor(Double(inputHeight + 2 * padding - kernelSize) / Double(strideHeight))) + 1
        let outputWidth: Int = Int(floor(Double(inputWidth + 2 * padding - kernelSize) / Double(strideWidth))) + 1
        let result: Tensor<DataType> = Tensor<DataType>(
            shape: [batchSize, nOutputChannels, outputHeight, outputWidth],
            initValue: DataType.zero);
        result.copyToGPU()

        let threadgroupsPerGridW = (outputWidth + Conv2DLayerNaive.GROUP_W - 1) / Conv2DLayerNaive.GROUP_W
        let threadgroupsPerGridH = (outputHeight + Conv2DLayerNaive.GROUP_W - 1) / Conv2DLayerNaive.GROUP_W
        var threadgroupsPerGrid = threadgroupsPerGridW * threadgroupsPerGridH
        threadgroupsPerGrid = (threadgroupsPerGrid == 0) ? 1 : threadgroupsPerGrid

        var convParamsCPU = Conv2DLayerParams(
            batch_size: uint(batchSize),
            n_input_channels: uint(nInputChannels),
            n_output_channels: uint(nOutputChannels),
            output_height: uint(outputHeight),
            output_width: uint(outputWidth),
            input_height: uint(inputHeight),
            input_width: uint(inputWidth),
            kernel_size: uint(kernelSize),
            stride_height: uint(strideHeight),
            stride_width: uint(strideWidth),
            padding: uint(padding),
            bias: (bias != nil),
            threadgroups_per_grid_dim4: uint(threadgroupsPerGridW)
        )
        let convParamsGPU = MTLUtils.copyToGPU(
            dataPtr: &convParamsCPU, size: MemoryLayout<Conv2DLayerParams>.stride)

        let cmdBuffer = MTLCommons.mtlCommandQueue.makeCommandBuffer()!
        let cmdEncoder = cmdBuffer.makeComputeCommandEncoder()!
        assert(MTLUtils.addComputePipeline(cmdEncoder: cmdEncoder,
                                           kernelLibrary: MTLCommons.defaultLib,
                                           kernelFuncName: "conv2d_forward") == true)
        
        cmdEncoder.setBuffer(result.dataGPU, offset: 0, index: 0)
        cmdEncoder.setBuffer(input.dataGPU, offset: 0, index: 1)
        cmdEncoder.setBuffer(kernels.dataGPU, offset: 0, index: 2)
        cmdEncoder.setBuffer((bias == nil) ? nil : bias!.dataGPU, offset: 0, index: 3)
        cmdEncoder.setBuffer(convParamsGPU, offset: 0, index: 4)

        let nthreadsPerBlock = MTLSize(
            width: Conv2DLayerNaive.GROUP_W, height: Conv2DLayerNaive.GROUP_W, depth: 1)
        let nblocks = MTLSize(
            width: batchSize, height: nOutputChannels, depth: threadgroupsPerGrid)
        cmdEncoder.dispatchThreadgroups(nblocks, threadsPerThreadgroup: nthreadsPerBlock)
        
        cmdEncoder.endEncoding()

        cmdBuffer.commit()
        cmdBuffer.waitUntilCompleted()

        return result
    }
}

public class Conv2DLayerImg2col: NetworkModuleProtocol {
    private static let GROUP_W: Int = 32
    
    private let nInputChannels: Int
    private let nOutputChannels: Int
    private let gpu: Bool
    
    private let kernelSize: Int
    private let strideHeight: Int
    private let strideWidth: Int
    private let padding: Int
    private let paddingMode: PaddingMode
    
    private let kernels: Tensor<DataType>
    private let bias: Tensor<DataType>?
    
    public init(nInputChannels: Int, nOutputChannels: Int, bias: Bool,
                kernelSize: Int, strideHeight: Int, strideWidth: Int,
                padding: Int, paddingMode: PaddingMode, gpu: Bool,
                initKernels: [DataType]? = nil, initBias: [DataType]? = nil) {
        self.nInputChannels = nInputChannels
        self.nOutputChannels = nOutputChannels
        self.gpu = gpu

        self.kernelSize = kernelSize
        self.strideHeight = strideHeight
        self.strideWidth = strideWidth
        self.padding = padding
        self.paddingMode = paddingMode

        let kernelShape = [nOutputChannels, nInputChannels, kernelSize, kernelSize]
        if (initKernels == nil) {
            self.kernels = Tensor<DataType>(
                shape: kernelShape, initValue: 1)
        } else {
            self.kernels = Tensor<DataType>(
                shape: kernelShape, data: initKernels!)
        }
        self.kernels.reshape(shape: [nOutputChannels, nInputChannels * kernelSize * kernelSize])
        if (gpu) {
            self.kernels.copyToGPU()
        }
        
        if (!bias) {
            self.bias = nil;
        } else {
            let biasShape = [1, nOutputChannels];
            if (initBias == nil) {
                self.bias = Tensor<DataType>(
                    shape: biasShape, initValue: 1)
            } else {
                self.bias = Tensor<DataType>(
                    shape: biasShape, data: initBias!)
            }
            if (gpu) {
                self.bias!.copyToGPU()
            }
        }
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
        let inputHeight: Int = inputShape[2]
        let inputWidth: Int = inputShape[3]
        let outputHeight: Int = Int(floor(Double(inputHeight + 2 * padding - kernelSize) / Double(strideHeight))) + 1
        let outputWidth: Int = Int(floor(Double(inputWidth + 2 * padding - kernelSize) / Double(strideWidth))) + 1
        let result: Tensor<DataType> = Tensor<DataType>(
            shape: [batchSize, nOutputChannels, outputHeight, outputWidth],
            initValue: DataType.zero);
        return result;
    }
    
    private func gpuForward(input: Tensor<DataType>) -> Tensor<DataType> {
        assert(input.dataGPU != nil)

        let inputShape: [Int] = input.getShape()
        let batchSize: Int = inputShape[0]
        let inputHeight: Int = inputShape[2]
        let inputWidth: Int = inputShape[3]
        let outputHeight: Int = Int(floor(Double(inputHeight + 2 * padding - kernelSize) / Double(strideHeight))) + 1
        let outputWidth: Int = Int(floor(Double(inputWidth + 2 * padding - kernelSize) / Double(strideWidth))) + 1
        let img2colBuffer: Tensor<DataType> = Tensor<DataType>(
            shape: [nInputChannels * kernelSize * kernelSize * outputHeight * outputWidth],
            initValue: DataType.zero)
        img2colBuffer.copyToGPU()
        /*
        let result: Tensor<DataType> = Tensor<DataType>(
            shape: [batchSize, nOutputChannels, outputHeight, outputWidth],
            initValue: DataType.zero);
        result.copyToGPU()
        */

        var convParamsCPU = Conv2DLayerParams(
            batch_size: uint(batchSize),
            n_input_channels: uint(nInputChannels),
            n_output_channels: uint(nOutputChannels),
            output_height: uint(outputHeight),
            output_width: uint(outputWidth),
            input_height: uint(inputHeight),
            input_width: uint(inputWidth),
            kernel_size: uint(kernelSize),
            stride_height: uint(strideHeight),
            stride_width: uint(strideWidth),
            padding: uint(padding),
            bias: (bias != nil),
            threadgroups_per_grid_dim4: 0
        )
        let convParamsGPU = MTLUtils.copyToGPU(
            dataPtr: &convParamsCPU, size: MemoryLayout<Conv2DLayerParams>.stride)

        for batchIdx in 0..<batchSize {
            let cmdBuffer = MTLCommons.mtlCommandQueue.makeCommandBuffer()!
            let cmdEncoder = cmdBuffer.makeComputeCommandEncoder()!
            assert(MTLUtils.addComputePipeline(cmdEncoder: cmdEncoder,
                                               kernelLibrary: MTLCommons.defaultLib,
                                               kernelFuncName: "conv2d_img2col") == true)
            
            var batchIdxCPU = uint(batchIdx)
            let batchIdxGPU = MTLUtils.copyToGPU(
                dataPtr: &batchIdxCPU, size: MemoryLayout<uint>.stride)

            cmdEncoder.setBuffer(img2colBuffer.dataGPU, offset: 0, index: 0)
            cmdEncoder.setBuffer(input.dataGPU, offset: 0, index: 1)
            cmdEncoder.setBuffer(convParamsGPU, offset: 0, index: 2)
            cmdEncoder.setBuffer(batchIdxGPU, offset: 0, index: 3)

            let groupSize = Conv2DLayerImg2col.GROUP_W * Conv2DLayerImg2col.GROUP_W
            let nthreadsPerBlock = MTLSize(width: groupSize, height: 1, depth: 1)
            let nblocks = MTLSize(
                width: (nInputChannels * outputWidth * outputHeight + groupSize - 1) / groupSize,
                height: 1, depth: 1)
            cmdEncoder.dispatchThreadgroups(nblocks, threadsPerThreadgroup: nthreadsPerBlock)
            
            cmdEncoder.endEncoding()

            cmdBuffer.commit()
            cmdBuffer.waitUntilCompleted()
        }
        return img2colBuffer
    }
}
