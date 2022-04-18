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
    
    public init(mtlBundle: MTLBundle, nInputChannels: Int, nOutputChannels: Int, bias: Bool,
                kernelSize: Int, strideHeight: Int, strideWidth: Int,
                padding: Int, paddingMode: PaddingMode, gpu: Bool,
                initKernels: [DataType]? = nil, initBias: [DataType]? = nil) {
        self.mtlBundle = mtlBundle
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
                shape: kernelShape, initValue: 1, mtlBundle.mtlDevice)
        } else {
            self.kernels = Tensor<DataType>(
                shape: kernelShape, data: initKernels!, mtlBundle.mtlDevice)
        }
        
        if (!bias) {
            self.bias = nil;
        } else {
            let biasShape = [1, nOutputChannels];
            if (initBias == nil) {
                self.bias = Tensor<DataType>(
                    shape: biasShape, initValue: 1, mtlBundle.mtlDevice)
            } else {
                self.bias = Tensor<DataType>(
                    shape: biasShape, data: initBias!, mtlBundle.mtlDevice)
            }
        }
    }
    
    public func forward(input: Tensor<DataType>) -> Tensor<DataType> {
        if (gpu) {
            return cpuForward(input: input)
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
            initValue: DataType.zero, mtlBundle.mtlDevice);

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
}
