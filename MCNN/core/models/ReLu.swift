//
//  ReLu.swift
//  MCNN iOS
//
//  Created by BerthCloud Chou on 2022/4/12.
//

import Foundation
import Metal

public class ReLu: NetworkModuleProtocol {
    private let gpu: Bool
    
    public init(gpu: Bool) {
        self.gpu = gpu
    }
    
    public func forward(input: Tensor<DataType>) -> Tensor<DataType> {
        if (gpu) {
            return cpuForward(input: input)
        } else {
            return cpuForward(input: input)
        }
    }
    
    private func cpuForward(input: Tensor<DataType>) -> Tensor<DataType> {
        for i in 0..<input.data.count {
            if (input.data[i] < 0) {
                input.data[i] = 0
            }
        }
        return input
    }
}
