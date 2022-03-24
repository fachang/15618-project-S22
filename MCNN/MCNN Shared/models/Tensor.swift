//
//  Tensor.swift
//  MCNN iOS
//
//  Created by BerthCloud Chou on 2022/3/24.
//

import Foundation

public class Tensor: CustomStringConvertible {
    
    public typealias DataType = Float32
    
    public var data: [DataType]

    public init(nDim: Int) {
        data = [DataType](repeating: 0, count: nDim )
    }
    
    public var description: String {
        return data.description
    }
}
