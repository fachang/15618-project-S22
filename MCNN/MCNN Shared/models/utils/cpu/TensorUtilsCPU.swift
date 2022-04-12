//
//  TensorUtilsCPU.swift
//  MCNN iOS
//
//  Created by BerthCloud Chou on 2022/4/10.
//

import Foundation

struct TensorUtilsCPU<DataType: AdditiveArithmetic & Numeric> {
    static func matMul(t1: Tensor<DataType>, t2: Tensor<DataType>) -> Tensor<DataType> {
        let t1Shape: [Int] = t1.getShape()
        let t2Shape: [Int] = t2.getShape()
        assert(t1Shape.count == 2 && t2Shape.count == 2 && t1Shape[1] == t2Shape[0])
        
        let result: Tensor<DataType> = Tensor<DataType>(
            shape: [t1Shape[0], t2Shape[1]], initValue: DataType.zero)

        for i in 0..<t1Shape[0] {
            for j in 0..<t2Shape[1] {
                var sum: DataType = DataType.zero
                for k in 0..<t1Shape[1] {
                    sum += (t1.getData(idx: [i, k]) * t2.getData(idx: [k, j]))
                }
                result.setData(idx: [i, j], value: sum)
            }
        }
        return result
    }
    
    static func addRow(t: inout Tensor<DataType>, row: Tensor<DataType>) {
        let tShape: [Int] = t.getShape()
        let rowShape: [Int] = row.getShape()
        assert(rowShape.count == 2 && rowShape[0] == 1 &&
               tShape[tShape.count - 1] == rowShape[1])
        for j in 0..<rowShape[1] {
            row.getData(idx: [0, j])
        }
        for dim in 0..<tShape.count {
            
        }

    }
}