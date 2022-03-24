//
//  Network.swift
//  MCNN iOS
//
//  Created by BerthCloud Chou on 2022/3/24.
//

import Foundation
import Metal

public class Network {
    
    private let mtlBundle: MTLBundle
    private let layers: [ForwardableProtocol]
    
    public init() {
        self.mtlBundle = MTLBundle();
        
        self.layers = [
            LinearLayer(mtlBundle: self.mtlBundle, nInputFeatures: 5, nOutputFeatures: 5),
            LinearLayer(mtlBundle: self.mtlBundle, nInputFeatures: 5, nOutputFeatures: 5),
        ];
    }
    
    public func forward(input: Tensor) -> Tensor {
        var curTensor: Tensor = input
        for nnModule in self.layers {
            curTensor = nnModule.forward(input: curTensor)
        }
        return curTensor
    }
}
