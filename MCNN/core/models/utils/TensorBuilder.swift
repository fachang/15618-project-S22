//
//  TensorBuilder.swift
//  MCNN iOS
//
//  Created by BerthCloud Chou on 2022/5/3.
//

import Foundation

public struct TensorBuilder {
    public static func buildRandFloat32Tensor(_ shape: [Int], _ start: Float32,
                                              _ end: Float32) -> Tensor<Float32> {
        let flattenSize: Int = shape.reduce(1, {result, i in return result * i})
        let randData: [Float32] = (0..<flattenSize).map { _ in Float32.random(in: start..<end) }
        return Tensor<Float32>(shape: shape, data: randData)
    }
}
