//
//  Tensor.swift
//  MCNN iOS
//
//  Created by BerthCloud Chou on 2022/3/24.
//

import Foundation

public class Tensor<T>: CustomStringConvertible {
    
    public var data: [T]
    private var shape: [Int]

    public init(shape: [Int], initValue: T) {
        let flattenSize: Int = shape.reduce(1, {result, i in return result * i})
        self.data = [T](repeating: initValue, count: flattenSize)
        self.shape = shape
    }
    
    public var description: String {
        return self.data.description
    }
    
    public func reshape(shape: [Int]) {
        self.shape = shape
    }
    
    public func getShape() -> [Int] {
        return self.shape
    }
    
    public func getData(idx: [Int]) -> T {
        var realIdx = idx[0]
        for i in 1..<idx.count {
            realIdx = (realIdx * shape[i] + idx[i])
        }
        return data[realIdx]
    }
    
    public func setData(idx: [Int], value: T) {
        var realIdx = idx[0]
        for i in 1..<idx.count {
            realIdx = (realIdx * shape[i] + idx[i])
        }
        data[realIdx] = value
    }
}
