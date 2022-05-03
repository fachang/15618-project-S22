//
//  LeNetNetwork.swift
//  MCNN iOS
//
//  Created by BerthCloud Chou on 2022/5/2.
//

import Foundation

private class C1: NetworkModuleProtocol {

    private let layers: [NetworkModuleProtocol]

    public init(gpu: Bool = false) {
        self.layers = [
            Conv2DLayerImg2col(nInputChannels: 1, nOutputChannels: 6, bias: true,
                        kernelSize: 5, strideHeight: 1, strideWidth: 1,
                        padding: 0, paddingMode: PaddingMode.zeros, gpu: gpu),
            ReLu(gpu: gpu),
            MaxPool2DLayer(kernelSize: 2, strideHeight: 2, strideWidth: 2, padding: 0, gpu: gpu),
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

private class C2: NetworkModuleProtocol {

    private let layers: [NetworkModuleProtocol]

    public init(gpu: Bool = false) {
        self.layers = [
            Conv2DLayerImg2col(nInputChannels: 6, nOutputChannels: 16, bias: true,
                        kernelSize: 5, strideHeight: 1, strideWidth: 1,
                        padding: 0, paddingMode: PaddingMode.zeros, gpu: gpu),
            ReLu(gpu: gpu),
            MaxPool2DLayer(kernelSize: 2, strideHeight: 2, strideWidth: 2, padding: 0, gpu: gpu),
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

private class C3: NetworkModuleProtocol {

    private let layers: [NetworkModuleProtocol]

    public init(gpu: Bool = false) {
        self.layers = [
            Conv2DLayerImg2col(nInputChannels: 16, nOutputChannels: 120, bias: true,
                        kernelSize: 5, strideHeight: 1, strideWidth: 1,
                        padding: 0, paddingMode: PaddingMode.zeros, gpu: gpu),
            ReLu(gpu: gpu),
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

private class F4: NetworkModuleProtocol {

    private let layers: [NetworkModuleProtocol]

    public init(gpu: Bool = false) {
        self.layers = [
            LinearLayer(nInputFeatures: 120, nOutputFeatures: 84, bias: true, gpu: gpu),
            ReLu(gpu: gpu),
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

private class F5: NetworkModuleProtocol {

    private let layers: [NetworkModuleProtocol]

    public init(gpu: Bool = false) {
        self.layers = [
            LinearLayer(nInputFeatures: 84, nOutputFeatures: 10, bias: true, gpu: gpu),
            ReLu(gpu: gpu), //TODO: Change to Softmax after Softmax is implemented
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

public class LeNet5: NetworkModuleProtocol {

    private let c1: C1
    private let c2_1: C2
    private let c2_2: C2
    private let c3: C3
    private let f4: F4
    private let f5: F5
    
    public init(gpu: Bool = false) {
        c1 = C1(gpu: gpu)
        c2_1 = C2(gpu: gpu)
        c2_2 = C2(gpu: gpu)
        c3 = C3(gpu: gpu)
        f4 = F4(gpu: gpu)
        f5 = F5(gpu: gpu)
    }
    
    public func forward(input: Tensor<DataType>) -> Tensor<DataType> {
        var output = c1.forward(input: input)

        c2_1.forward(input: output) // TODO: add the outputs of c2_1 and c2_2
        output = c2_2.forward(input: output)

        output = c3.forward(input: output)
        let batchSize = input.getShape()[0]
        output.reshape(shape: [batchSize, output.data.count / batchSize])

        output = f4.forward(input: output)
        output = f5.forward(input: output)
        return output
    }
}
