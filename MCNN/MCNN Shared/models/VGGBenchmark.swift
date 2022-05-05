//
//  VGGBenchmark.swift
//  MCNN iOS
//
//  Created by BerthCloud Chou on 2022/5/3.
//

import Foundation

//
//  LeNetBenchmark.swift
//  MCNN iOS
//
//  Created by BerthCloud Chou on 2022/5/2.
//

import Foundation

public class VGGBenchmark {
    public typealias DataType = Float32
    public typealias VGGNetwork = VGG11
    
    private let batchSize: Int
    private let gpu: Bool
    
    public init(gpu: Bool) {
        self.batchSize = 1
        self.gpu = gpu
    }

    public func runFullNetwork() -> Metrics {
        let input: Tensor<DataType> = TensorBuilder.buildRandFloat32Tensor([batchSize, 3, 224, 224], 0, 256)
        
        var startTime = DispatchTime.now()
        if (gpu) {
            input.copyToGPU()
        }
        let network = VGGNetwork(gpu: gpu)
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
        let input: Tensor<DataType> = TensorBuilder.buildRandFloat32Tensor([batchSize, 3, 224, 224], 0, 256)

        var startTime = DispatchTime.now()
        if (gpu) {
            input.copyToGPU()
        }
        let network = VGGNetwork.Conv2DLayer(nInputChannels: 3, nOutputChannels: 64, bias: true,
                                        kernelSize: 3, strideHeight: 1, strideWidth: 1,
                                        padding: 1, paddingMode: PaddingMode.zeros, gpu: gpu)
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
        let input: Tensor<DataType> = TensorBuilder.buildRandFloat32Tensor([batchSize, 3, 112, 112], 0, 1)

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
        let input: Tensor<DataType> = TensorBuilder.buildRandFloat32Tensor([batchSize, 25088], 0, 1)

        var startTime = DispatchTime.now()
        if (gpu) {
            input.copyToGPU()
        }
        let network = LinearLayer(nInputFeatures: 25088, nOutputFeatures: 4096, bias: true, gpu: gpu)
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
        let input: Tensor<DataType> = TensorBuilder.buildRandFloat32Tensor([batchSize, 3, 112, 112], -1, 1)

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
