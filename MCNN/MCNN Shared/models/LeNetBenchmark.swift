//
//  LeNetBenchmark.swift
//  MCNN iOS
//
//  Created by BerthCloud Chou on 2022/5/2.
//

import Foundation
import Metal

public class LeNetBenchmark {
    public typealias DataType = Float32
    
    private let batchSize: Int
    private let gpu: Bool
    
    public init(gpu: Bool) {
        self.batchSize = 1
        self.gpu = gpu
    }

    public func runFullNetwork() -> Metrics {
        let inputData: [DataType] = (0..<(batchSize * 1 * 32 * 32)).map { _ in .random(in: 0..<256) }
        
        var startTime = DispatchTime.now()
        let input = Tensor<DataType>(shape: [batchSize, 1, 32, 32], data: inputData)
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
}
