//
//  TensorBuilderCPU.swift
//  MCNN iOS
//
//  Created by BerthCloud Chou on 2022/4/11.
//

import Foundation

struct TensorBuilderCPU {
    static func buildRandFloat16(shape: [Int], start: Float16, end: Float16) -> Tensor<Float16> {
        let flattenSize: Int = shape.reduce(1, {result, i in return result * i})
        let result = Tensor<Float16>(shape: shape, initValue: 0)
        for i in 0..<flattenSize {
            result.data[i] = Float16.random(in: start..<end)
        }
        return result
    }
}
