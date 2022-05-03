//
//  AlexNetBenchmark.swift
//  MCNN iOS
//
//  Created by BerthCloud Chou on 2022/5/3.
//

import Foundation

public class AlexNetBenchmark {
    public typealias DataType = Float32
    
    private let batchSize: Int
    private let gpu: Bool
    
    public init(gpu: Bool) {
        self.batchSize = 1
        self.gpu = gpu
    }
    
    public func runConv2D() -> Metrics {
        let input: Tensor<DataType> = TensorBuilder.buildRandFloat32Tensor([batchSize, 3, 227, 227], 0, 256)

        var startTime = DispatchTime.now()
        if (gpu) {
            input.copyToGPU()
        }
        let network = Conv2DLayerImg2col(nInputChannels: 3, nOutputChannels: 96, bias: true,
                                         kernelSize: 11, strideHeight: 4, strideWidth: 4,
                                         padding: 0, paddingMode: PaddingMode.zeros, gpu: gpu)
        let initElapsedTime = DispatchTime.now().uptimeNanoseconds - startTime.uptimeNanoseconds
        
        startTime = DispatchTime.now()
        let forwardResult: Tensor<DataType> = network.forward(input: input)
        let forwardElapsedTime = DispatchTime.now().uptimeNanoseconds - startTime.uptimeNanoseconds
        if (gpu) {
            forwardResult.copyToCPU()
        }
        // forwardResult.printData()

        return Metrics(initElapseNanosecs: initElapsedTime, forwardElapseNanosecs: forwardElapsedTime)
    }
    
    public func runMaxPool2D() -> Metrics {
        let input: Tensor<DataType> = TensorBuilder.buildRandFloat32Tensor([batchSize, 3, 54, 54], 0, 1)

        var startTime = DispatchTime.now()
        if (gpu) {
            input.copyToGPU()
        }
        let network = MaxPool2DLayer(kernelSize: 3, strideHeight: 2, strideWidth: 2, padding: 0, gpu: gpu)
        let initElapsedTime = DispatchTime.now().uptimeNanoseconds - startTime.uptimeNanoseconds
        
        startTime = DispatchTime.now()
        let forwardResult: Tensor<DataType> = network.forward(input: input)
        let forwardElapsedTime = DispatchTime.now().uptimeNanoseconds - startTime.uptimeNanoseconds
        if (gpu) {
            forwardResult.copyToCPU()
        }
        // forwardResult.printData()

        return Metrics(initElapseNanosecs: initElapsedTime, forwardElapseNanosecs: forwardElapsedTime)
    }
    
    public func runFC() -> Metrics {
        let input: Tensor<DataType> = TensorBuilder.buildRandFloat32Tensor([batchSize, 9216], 0, 1)

        var startTime = DispatchTime.now()
        if (gpu) {
            input.copyToGPU()
        }
        let network = LinearLayer(nInputFeatures: 9216, nOutputFeatures: 4096, bias: true, gpu: gpu)
        let initElapsedTime = DispatchTime.now().uptimeNanoseconds - startTime.uptimeNanoseconds
        
        startTime = DispatchTime.now()
        let forwardResult: Tensor<DataType> = network.forward(input: input)
        let forwardElapsedTime = DispatchTime.now().uptimeNanoseconds - startTime.uptimeNanoseconds
        if (gpu) {
            forwardResult.copyToCPU()
        }
        // forwardResult.printData()

        return Metrics(initElapseNanosecs: initElapsedTime, forwardElapseNanosecs: forwardElapsedTime)
    }
    
    public func runReLu() -> Metrics {
        let input: Tensor<DataType> = TensorBuilder.buildRandFloat32Tensor([batchSize, 3, 55, 55], -1, 1)

        var startTime = DispatchTime.now()
        if (gpu) {
            input.copyToGPU()
        }
        let network = ReLu(gpu: gpu)
        let initElapsedTime = DispatchTime.now().uptimeNanoseconds - startTime.uptimeNanoseconds
        
        startTime = DispatchTime.now()
        let forwardResult: Tensor<DataType> = network.forward(input: input)
        let forwardElapsedTime = DispatchTime.now().uptimeNanoseconds - startTime.uptimeNanoseconds
        if (gpu) {
            forwardResult.copyToCPU()
        }
        // forwardResult.printData()

        return Metrics(initElapseNanosecs: initElapsedTime, forwardElapseNanosecs: forwardElapsedTime)
    }

}
