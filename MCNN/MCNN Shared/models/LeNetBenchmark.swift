//
//  LeNetBenchmark.swift
//  MCNN iOS
//
//  Created by BerthCloud Chou on 2022/5/2.
//

import Foundation

public class LeNetBenchmark {
    public typealias DataType = Float32
    
    private let batchSize: Int
    private let gpu: Bool
    
    public init(gpu: Bool) {
        self.batchSize = 1
        self.gpu = gpu
    }

    public func runFullNetwork() -> Metrics {
        let input: Tensor<DataType> = TensorBuilder.buildRandFloat32Tensor([batchSize, 1, 32, 32], 0, 256)
        
        var startTime = DispatchTime.now()
        if (gpu) {
            input.copyToGPU()
        }
        let network = LeNet5(gpu: gpu)
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
    
    public func runConv2D() -> Metrics {
        let input: Tensor<DataType> = TensorBuilder.buildRandFloat32Tensor([batchSize, 1, 32, 32], 0, 256)

        var startTime = DispatchTime.now()
        if (gpu) {
            input.copyToGPU()
        }
        let network = LeNet5.Conv2DLayer(nInputChannels: 1, nOutputChannels: 6, bias: true,
                                         kernelSize: 5, strideHeight: 1, strideWidth: 1,
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
        let input: Tensor<DataType> = TensorBuilder.buildRandFloat32Tensor([batchSize, 3, 28, 28], 0, 1)

        var startTime = DispatchTime.now()
        if (gpu) {
            input.copyToGPU()
        }
        let network = MaxPool2DLayer(kernelSize: 2, strideHeight: 2, strideWidth: 2, padding: 0, gpu: gpu)
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
        let input: Tensor<DataType> = TensorBuilder.buildRandFloat32Tensor([batchSize, 120], 0, 1)

        var startTime = DispatchTime.now()
        if (gpu) {
            input.copyToGPU()
        }
        let network = LinearLayer(nInputFeatures: 120, nOutputFeatures: 84, bias: true, gpu: gpu)
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
        let input: Tensor<DataType> = TensorBuilder.buildRandFloat32Tensor([batchSize, 3, 28, 28], -1, 1)

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
