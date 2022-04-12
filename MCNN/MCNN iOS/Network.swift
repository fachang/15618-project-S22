//
//  Network.swift
//  MCNN iOS
//
//  Created by BerthCloud Chou on 2022/3/24.
//

import Foundation
import Metal

public class TestLinearNetwork {
    public typealias DataType = Float16
    
    private let mtlBundle: MTLBundle
    private let layers: [NetworkModuleProtocol]
    
    public init() {
        self.mtlBundle = MTLBundle();
        
        self.layers = [
            LinearLayer(
                mtlBundle: self.mtlBundle, nInputFeatures: 3, nOutputFeatures: 5, bias: true, gpu: false),
            LinearLayer(
                mtlBundle: self.mtlBundle, nInputFeatures: 5, nOutputFeatures: 2, bias: true, gpu: false),
        ];
    }
    
    public func forward(input: Tensor<DataType>) -> Tensor<DataType> {
        var curTensor: Tensor<DataType> = input
        for nnModule in self.layers {
            curTensor = nnModule.forward(input: curTensor)
        }
        return curTensor
    }
}

public class TestConvNetwork {
    public typealias DataType = Float16
    
    private let mtlBundle: MTLBundle
    private let layers: [NetworkModuleProtocol]
    
    public init() {
        self.mtlBundle = MTLBundle();
        
        self.layers = [
            Conv2DLayer(mtlBundle: self.mtlBundle, nInputChannels: 3, nOutputChannels: 2, bias: true,
                        kernelSize: 3, strideHeight: 2, strideWidth: 1,
                        padding: 1, paddingMode: PaddingMode.zeros, gpu: false)
        ];
    }
    
    public func forward(input: Tensor<DataType>) -> Tensor<DataType> {
        var curTensor: Tensor<DataType> = input
        for nnModule in self.layers {
            curTensor = nnModule.forward(input: curTensor)
        }
        return curTensor
    }
}
