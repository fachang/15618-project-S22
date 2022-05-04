//
//  ForwardableProtocol.swift
//  MCNN iOS
//
//  Created by BerthCloud Chou on 2022/3/24.
//

import Foundation

public protocol NetworkModuleProtocol {
    typealias DataType = Float32
    typealias Conv2DLayer = Conv2DLayerImg2col
    func forward(input: Tensor<DataType>) -> Tensor<DataType>
}
