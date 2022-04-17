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
    
    public init(shape: [Int], data: [T]) {
        let flattenSize: Int = shape.reduce(1, {result, i in return result * i})
        assert(flattenSize == data.count)
        self.data = data
        self.shape = shape
    }
    
    public var description: String {
        return self.data.description
    }
    
    public func reshape(shape: [Int]) {
        let oldFlattenSize: Int = self.shape.reduce(1, {result, i in return result * i})
        let newFlattenSize: Int = shape.reduce(1, {result, i in return result * i})
        assert(oldFlattenSize == newFlattenSize)
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
    
    public func printData() {
        Tensor<T>.printDataHelper(arr: data, shape: shape, dimIdx: 0, arrDimStart: 0)
    }
    
    private static func printDataHelper(arr: [T], shape: [Int], dimIdx: Int, arrDimStart: Int) -> Int {
        var indent: String = ""
        for _ in 0..<dimIdx {
            indent += "    "
        }

        if (dimIdx == shape.count - 1) {
            print("\(indent)[", terminator: "")
            for i in 0..<shape[dimIdx] {
                print("\(arr[arrDimStart + i]), ", terminator: "")
            }
            print("],")
            return shape[dimIdx]
        }
        var arrShift = 0
        for _ in 0..<shape[dimIdx] {
            print("\(indent)[")
            arrShift += printDataHelper(arr: arr, shape: shape, dimIdx: dimIdx + 1,
                                        arrDimStart: arrDimStart + arrShift)
            print("\(indent)],")
        }
        return arrShift
    }
}
