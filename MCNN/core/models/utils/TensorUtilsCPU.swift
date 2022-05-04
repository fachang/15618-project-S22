//
//  TensorUtilsCPU.swift
//  MCNN iOS
//
//  Created by BerthCloud Chou on 2022/4/10.
//

import Foundation

public struct TensorUtilsCPU<DataType: AdditiveArithmetic & Numeric> {
    public static func matMul(t1: Tensor<DataType>, t2: Tensor<DataType>) -> Tensor<DataType> {
        let t1Shape: [Int] = t1.getShape()
        let t2Shape: [Int] = t2.getShape()
        assert(t1Shape.count == 2 && t2Shape.count == 2 && t1Shape[1] == t2Shape[0])
        
        let result: Tensor<DataType> = Tensor<DataType>(
            shape: [t1Shape[0], t2Shape[1]], initValue: DataType.zero)

        /*
        for k in 0..<t1Shape[1] {
            for i in 0..<t1Shape[0] {
                // let tmp = t1.getData(idx: [i, k])
                let tmp = t1.data[i * t1Shape[1] + k]
                for j in 0..<t2Shape[1] {
                    /*
                    result.setData(idx: [i, j],
                                   value: result.getData(idx: [i, j]) +
                                          tmp * t2.getData(idx: [k, j]))
                    */
                    result.data[i * t2Shape[1] + j] += tmp * t2.data[k * t2Shape[1] + j]
                }
            }
        }
        */
        for i in 0..<t1Shape[0] {
            for j in 0..<t2Shape[1] {
                var sum: DataType = DataType.zero
                for k in 0..<t1Shape[1] {
                    // sum += (t1.getData(idx: [i, k]) * t2.getData(idx: [k, j]))
                    sum += (t1.data[i * t1Shape[1] + k] * t2.data[k * t2Shape[1] + j])
                }
                // result.setData(idx: [i, j], value: sum)
                result.data[i * t2Shape[1] + j] = sum
            }
        }
        return result
    }
    
    public static func matMul(result: inout Tensor<DataType>, resultDimStart: Int,
                              t1: Tensor<DataType>, t2: Tensor<DataType>) {
        let t1Shape: [Int] = t1.getShape()
        let t2Shape: [Int] = t2.getShape()
        assert(t1Shape.count == 2 && t2Shape.count == 2 && t1Shape[1] == t2Shape[0])

        /*
        for k in 0..<t1Shape[1] {
            for i in 0..<t1Shape[0] {
                // let tmp = t1.getData(idx: [i, k])
                let tmp = t1.data[i * t1Shape[1] + k]
                for j in 0..<t2Shape[1] {
                    /*
                    result.setData(idx: [resultDimStart, i, j],
                                   value: result.getData(idx: [resultDimStart, i, j]) +
                                          tmp * t2.getData(idx: [k, j]))
                    */
                    result.data[resultDimStart * t1Shape[0] * t2Shape[1] +
                                i * t2Shape[1] + j] += tmp * t2.data[k * t2Shape[1] + j]
                }
            }
        }
        */
        for i in 0..<t1Shape[0] {
            for j in 0..<t2Shape[1] {
                var sum: DataType = DataType.zero
                for k in 0..<t1Shape[1] {
                    // sum += (t1.getData(idx: [i, k]) * t2.getData(idx: [k, j]))
                    sum += (t1.data[i * t1Shape[1] + k] * t2.data[k * t2Shape[1] + j])
                }
                // result.setData(idx: [i, j], value: sum)
                result.data[resultDimStart * t1Shape[0] * t2Shape[1] + i * t2Shape[1] + j] = sum
            }
        }
    }
}
