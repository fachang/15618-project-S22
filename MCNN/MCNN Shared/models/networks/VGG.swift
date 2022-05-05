//
//  VGG.swift
//  MCNN iOS
//
//  Created by BerthCloud Chou on 2022/5/3.
//

import Foundation

public class VGG11: NetworkModuleProtocol {

    private let layers: [NetworkModuleProtocol]
    private let flattenLayers: [NetworkModuleProtocol]

    public init(gpu: Bool = false) {
        let maxpool = MaxPool2DLayer(kernelSize: 2, strideHeight: 2, strideWidth: 2, padding: 0, gpu: gpu)
        self.layers = [
            Conv2DLayer(nInputChannels: 3, nOutputChannels: 128, bias: true,
                        kernelSize: 3, strideHeight: 1, strideWidth: 1,
                        padding: 1, paddingMode: PaddingMode.zeros, gpu: gpu),
            ReLu(gpu: gpu),
            Conv2DLayer(nInputChannels: 128, nOutputChannels: 128, bias: true,
                        kernelSize: 3, strideHeight: 1, strideWidth: 1,
                        padding: 1, paddingMode: PaddingMode.zeros, gpu: gpu),
            ReLu(gpu: gpu),
            maxpool,
            Conv2DLayer(nInputChannels: 128, nOutputChannels: 256, bias: true,
                        kernelSize: 3, strideHeight: 1, strideWidth: 1,
                        padding: 1, paddingMode: PaddingMode.zeros, gpu: gpu),
            ReLu(gpu: gpu),
            Conv2DLayer(nInputChannels: 256, nOutputChannels: 256, bias: true,
                        kernelSize: 3, strideHeight: 1, strideWidth: 1,
                        padding: 1, paddingMode: PaddingMode.zeros, gpu: gpu),
            ReLu(gpu: gpu),
            maxpool,
            Conv2DLayer(nInputChannels: 256, nOutputChannels: 512, bias: true,
                        kernelSize: 3, strideHeight: 1, strideWidth: 1,
                        padding: 1, paddingMode: PaddingMode.zeros, gpu: gpu),
            ReLu(gpu: gpu),
            Conv2DLayer(nInputChannels: 512, nOutputChannels: 512, bias: true,
                        kernelSize: 3, strideHeight: 1, strideWidth: 1,
                        padding: 1, paddingMode: PaddingMode.zeros, gpu: gpu),
            ReLu(gpu: gpu),
            maxpool,
        ];

        self.flattenLayers = [
            LinearLayer(nInputFeatures: 8192, nOutputFeatures: 4096, bias: true, gpu: gpu),
            ReLu(gpu: gpu),
            // TODO: Dropout
            LinearLayer(nInputFeatures: 4096, nOutputFeatures: 4096, bias: true, gpu: gpu),
            ReLu(gpu: gpu),
            // TODO: Dropout
            LinearLayer(nInputFeatures: 4096, nOutputFeatures: 10, bias: true, gpu: gpu),
        ];
    }

    public func forward(input: Tensor<DataType>) -> Tensor<DataType> {
        var curTensor: Tensor<DataType> = input
        for nnModule in self.layers {
            curTensor = nnModule.forward(input: curTensor)
        }
        let batchSize = curTensor.getShape()[0]
        curTensor.reshape(shape: [batchSize, curTensor.data.count / batchSize])
        for nnModule in self.flattenLayers {
            curTensor = nnModule.forward(input: curTensor)
        }
        return curTensor
    }
}


public class VGG16: NetworkModuleProtocol {

    private let layers: [NetworkModuleProtocol]
    private let flattenLayers: [NetworkModuleProtocol]

    public init(gpu: Bool = false) {
        let maxpool = MaxPool2DLayer(kernelSize: 2, strideHeight: 2, strideWidth: 2, padding: 0, gpu: gpu)
        self.layers = [
            Conv2DLayer(nInputChannels: 3, nOutputChannels: 64, bias: true,
                        kernelSize: 3, strideHeight: 1, strideWidth: 1,
                        padding: 1, paddingMode: PaddingMode.zeros, gpu: gpu),
            ReLu(gpu: gpu),
            Conv2DLayer(nInputChannels: 64, nOutputChannels: 64, bias: true,
                        kernelSize: 3, strideHeight: 1, strideWidth: 1,
                        padding: 1, paddingMode: PaddingMode.zeros, gpu: gpu),
            ReLu(gpu: gpu),
            maxpool,
            Conv2DLayer(nInputChannels: 64, nOutputChannels: 128, bias: true,
                        kernelSize: 3, strideHeight: 1, strideWidth: 1,
                        padding: 1, paddingMode: PaddingMode.zeros, gpu: gpu),
            ReLu(gpu: gpu),
            Conv2DLayer(nInputChannels: 128, nOutputChannels: 128, bias: true,
                        kernelSize: 3, strideHeight: 1, strideWidth: 1,
                        padding: 1, paddingMode: PaddingMode.zeros, gpu: gpu),
            ReLu(gpu: gpu),
            maxpool,
            Conv2DLayer(nInputChannels: 128, nOutputChannels: 256, bias: true,
                        kernelSize: 3, strideHeight: 1, strideWidth: 1,
                        padding: 1, paddingMode: PaddingMode.zeros, gpu: gpu),
            ReLu(gpu: gpu),
            Conv2DLayer(nInputChannels: 256, nOutputChannels: 256, bias: true,
                        kernelSize: 3, strideHeight: 1, strideWidth: 1,
                        padding: 1, paddingMode: PaddingMode.zeros, gpu: gpu),
            ReLu(gpu: gpu),
            Conv2DLayer(nInputChannels: 256, nOutputChannels: 256, bias: true,
                        kernelSize: 3, strideHeight: 1, strideWidth: 1,
                        padding: 1, paddingMode: PaddingMode.zeros, gpu: gpu),
            ReLu(gpu: gpu),
            maxpool,
            Conv2DLayer(nInputChannels: 256, nOutputChannels: 512, bias: true,
                        kernelSize: 3, strideHeight: 1, strideWidth: 1,
                        padding: 1, paddingMode: PaddingMode.zeros, gpu: gpu),
            ReLu(gpu: gpu),
            Conv2DLayer(nInputChannels: 512, nOutputChannels: 512, bias: true,
                        kernelSize: 3, strideHeight: 1, strideWidth: 1,
                        padding: 1, paddingMode: PaddingMode.zeros, gpu: gpu),
            ReLu(gpu: gpu),
            Conv2DLayer(nInputChannels: 512, nOutputChannels: 512, bias: true,
                        kernelSize: 3, strideHeight: 1, strideWidth: 1,
                        padding: 1, paddingMode: PaddingMode.zeros, gpu: gpu),
            ReLu(gpu: gpu),
            maxpool,
            Conv2DLayer(nInputChannels: 512, nOutputChannels: 512, bias: true,
                        kernelSize: 3, strideHeight: 1, strideWidth: 1,
                        padding: 1, paddingMode: PaddingMode.zeros, gpu: gpu),
            ReLu(gpu: gpu),
            Conv2DLayer(nInputChannels: 512, nOutputChannels: 512, bias: true,
                        kernelSize: 3, strideHeight: 1, strideWidth: 1,
                        padding: 1, paddingMode: PaddingMode.zeros, gpu: gpu),
            ReLu(gpu: gpu),
            Conv2DLayer(nInputChannels: 512, nOutputChannels: 512, bias: true,
                        kernelSize: 3, strideHeight: 1, strideWidth: 1,
                        padding: 1, paddingMode: PaddingMode.zeros, gpu: gpu),
            ReLu(gpu: gpu),
            maxpool,
        ];

        self.flattenLayers = [
            LinearLayer(nInputFeatures: 25088, nOutputFeatures: 4096, bias: true, gpu: gpu),
            ReLu(gpu: gpu),
            // TODO: Dropout
            LinearLayer(nInputFeatures: 4096, nOutputFeatures: 4096, bias: true, gpu: gpu),
            ReLu(gpu: gpu),
            // TODO: Dropout
            LinearLayer(nInputFeatures: 4096, nOutputFeatures: 10, bias: true, gpu: gpu),
        ];
    }

    public func forward(input: Tensor<DataType>) -> Tensor<DataType> {
        var curTensor: Tensor<DataType> = input
        for nnModule in self.layers {
            curTensor = nnModule.forward(input: curTensor)
        }
        let batchSize = curTensor.getShape()[0]
        curTensor.reshape(shape: [batchSize, curTensor.data.count / batchSize])
        for nnModule in self.flattenLayers {
            curTensor = nnModule.forward(input: curTensor)
        }
        return curTensor
    }
}
