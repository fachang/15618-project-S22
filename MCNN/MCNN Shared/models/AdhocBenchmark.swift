//
//  AdhocBenchmark.swift
//  MCNN iOS
//
//  Created by BerthCloud Chou on 2022/5/3.
//

import Foundation

public class AdhocBenchmark {
    public typealias DataType = Float32
    
    private let gpu: Bool
    
    public init(gpu: Bool) {
        self.gpu = gpu
    }

    public func runFC_128_4096_4096() -> Metrics {
        let input: Tensor<DataType> = TensorBuilder.buildRandFloat32Tensor([128, 4096], 0, 1)

        var startTime = DispatchTime.now()
        if (gpu) {
            input.copyToGPU()
        }
        let network = LinearLayer(nInputFeatures: 4096, nOutputFeatures: 4096, bias: true, gpu: gpu)
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
    
    public func runConv2D_96_11_4_0() -> Metrics {
        let input: Tensor<DataType> = TensorBuilder.buildRandFloat32Tensor([1, 3, 227, 227], 0, 256)

        var startTime = DispatchTime.now()
        if (gpu) {
            input.copyToGPU()
        }
        let network = NetworkModuleProtocol.Conv2DLayer(nInputChannels: 3, nOutputChannels: 96, bias: true,
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
}
