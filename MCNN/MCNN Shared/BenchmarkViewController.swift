//
//  CNNDemoViewController.swift
//  MCNN iOS
//
//  Created by BerthCloud Chou on 2022/3/22.
//

import UIKit
import MetalKit

// Our iOS specific view controller
class BenchmarkViewController: UIViewController {

    public typealias DataType = Float32
    
    @IBOutlet var benchmarkTextLabel: UILabel!
    @IBOutlet var runBtn: UIButton!

    override func viewDidLoad() {
        super.viewDidLoad()
        
        benchmarkTextLabel.text = "CNN Forward Benchmark"
        runBtn.setTitle("RUN", for: .normal)
    }
    
    @IBAction func runBenchmark(sender: UIButton) {
        // runLinearNetworkBenchmark()
        // runConvNetworkBenchmark()
        runBigConvNetworkBenchmark()
    }
    
    private func runLinearNetworkBenchmark() {
        var startTime = DispatchTime.now()
        let network: TestLinearNetwork = TestLinearNetwork()
        let initElapsedTime = Double(
                                    DispatchTime.now().uptimeNanoseconds -
                                    startTime.uptimeNanoseconds) / 1000000000;
        
        startTime = DispatchTime.now()
        
        let input = Tensor<DataType>(shape: [10, 3], initValue: 1)
        input.copyToGPU()
        let forwardResult: Tensor<DataType> = network.forward(input: input)
        forwardResult.copyToCPU()
        
        let forwardElapsedTime = Double(
                                    DispatchTime.now().uptimeNanoseconds -
                                    startTime.uptimeNanoseconds) / 1000000000;

        forwardResult.printData()
        benchmarkTextLabel.text =
            "Metrics:\n" +
            "Init Elapsed Time: \(initElapsedTime) sec\n" +
            "Forward Elapsed Time: \(forwardElapsedTime) sec\n";
    }
    
    private func runConvNetworkBenchmark() {
        // Test CPU version
        var startTime = DispatchTime.now()
        var input: Tensor<DataType> = Tensor<DataType>(shape: [2, 3, 5, 5], initValue: 1)
        var network: TestConvNetwork = TestConvNetwork()
        var initElapsedTime = Double(
            DispatchTime.now().uptimeNanoseconds - startTime.uptimeNanoseconds) / 1000000000;

        startTime = DispatchTime.now()
        var forwardResult: Tensor<DataType> = network.forward(input: input)
        var forwardElapsedTime = Double(
            DispatchTime.now().uptimeNanoseconds - startTime.uptimeNanoseconds) / 1000000000;

        forwardResult.printData()
        var metricString =
            "Metrics:\n" +
            "Init Elapsed Time: \(initElapsedTime) sec\n" +
            "Forward Elapsed Time: \(forwardElapsedTime) sec\n";
        
        metricString += "--------------\n"
        
        // Test GPU version
        startTime = DispatchTime.now()
        input = Tensor<DataType>(shape: [2, 3, 5, 5], initValue: 1)
        input.copyToGPU()
        network = TestConvNetwork(gpu: true)
        initElapsedTime = Double(
            DispatchTime.now().uptimeNanoseconds - startTime.uptimeNanoseconds) / 1000000000;
        
        startTime = DispatchTime.now()
        forwardResult = network.forward(input: input)
        forwardResult.copyToCPU()
        forwardElapsedTime = Double(
            DispatchTime.now().uptimeNanoseconds - startTime.uptimeNanoseconds) / 1000000000;

        forwardResult.printData()
        metricString += (
            "Metrics:\n" +
            "Init Elapsed Time: \(initElapsedTime) sec\n" +
            "Forward Elapsed Time: \(forwardElapsedTime) sec\n"
        );
        
        benchmarkTextLabel.text = metricString;
    }
    
    private func runBigConvNetworkBenchmark() {
        let inputData: [DataType] = [
            -0.4242, -0.4242, -0.4242, -0.4242, -0.4242, -0.4242, -0.4242, -0.4242,
            -0.4242, -0.4242, -0.4242, -0.4242, -0.4242, -0.4242, -0.4242, -0.4242,
            -0.4242, -0.4242, -0.4242, -0.4242, -0.4242, -0.4242, -0.4242, -0.4242,
            -0.4242, -0.4242, -0.4242, -0.4242, -0.4242, -0.4242, -0.4242, -0.4242,
            -0.4242, -0.4242, -0.4242, -0.4242, -0.4242, -0.4242, -0.4242, -0.4242,
            -0.4242, -0.4242, -0.4242, -0.4242, -0.4242, -0.4242, -0.4242, -0.4242,
            -0.4242, -0.4242, -0.4242, -0.4242, -0.4242, -0.4242, -0.4242, -0.4242,
            -0.4242, -0.4242, -0.4242, -0.4242, -0.4242, -0.4242, -0.4242, -0.4242,
            -0.4242, -0.4242, -0.4242, -0.4242, -0.4242, -0.4242, -0.4242, -0.4242,
            -0.4242, -0.4242, -0.4242, -0.4242, -0.4242, -0.4242, -0.4242, -0.4242,
            -0.4242, -0.4242, -0.4242, -0.4242, -0.4242, -0.4242, -0.4242, -0.4242,
            -0.4242, -0.4242, -0.4242, -0.4242, -0.4242, -0.4242, -0.4242, -0.4242,
            -0.4242, -0.4242, -0.4242, -0.4242, -0.4242, -0.4242, -0.4242, -0.4242,
            -0.4242, -0.4242, -0.4242, -0.4242, -0.4242, -0.4242, -0.4242, -0.4242,
            -0.4242, -0.4242, -0.4242, -0.4242, -0.4242, -0.4242, -0.4242, -0.4242,
            -0.4242, -0.4242, -0.4242, -0.4242, -0.4242, -0.4242, -0.4242, -0.4242,
            -0.4242, -0.4242, -0.4242, -0.4242, -0.4242, -0.4242, -0.4242, -0.4242,
            -0.4242, -0.4242, -0.4242, -0.4242, -0.4242, -0.4242, -0.4242, -0.4242,
            -0.4242, -0.4242, -0.4242, -0.4242, -0.4242, -0.4242, -0.4242, -0.4242,
            -0.4242, -0.4242, -0.4242, -0.4242, -0.4242, -0.4242, -0.4242, -0.4242,
            -0.4242, -0.4242, -0.4242, -0.4242, -0.4242, -0.4242, -0.4242, -0.4242,
            -0.4242, -0.4242, -0.4242, -0.4242, -0.4242, -0.4242, -0.4242, -0.4242,
            -0.4242, -0.4242, -0.4242, -0.4242, -0.4242, -0.4242, -0.4242, -0.4242,
            -0.4242, -0.4242, -0.4242, -0.4242, -0.4242, -0.4242, -0.4242, -0.4242,
            -0.4242, -0.4242, -0.4242, -0.4242, -0.4242, -0.4242, -0.4242, -0.4242,
            -0.4242, -0.4242, -0.4242, -0.4242, -0.4242, -0.4242, -0.4242, -0.4242,
            -0.4242, -0.4242, -0.4242, -0.4242, -0.4242, -0.4242, -0.4242, -0.3860,
            -0.4115, -0.4242, -0.4242, -0.3351, -0.4242,  0.0467, -0.4242, -0.4242,
            -0.4242, -0.4242, -0.4242, -0.4242, -0.4242, -0.4242, -0.4242, -0.4242,
            -0.4242, -0.4242, -0.4242, -0.4242, -0.4242, -0.4115, -0.3988, -0.4242,
            -0.0806,  0.6450, -0.2842, -0.4242, -0.4242, -0.4242, -0.4242, -0.4242,
            -0.4242,  1.0904, -0.4242, -0.4242, -0.4242, -0.4242, -0.4242, -0.4242,
            -0.4242, -0.4242, -0.4242, -0.4242, -0.4242, -0.4242, -0.4242, -0.4242,
            -0.4242, -0.4115, -0.4242, -0.4242,  0.6959,  1.3959,  0.9759, -0.4242,
            -0.4242, -0.4242, -0.4242, -0.1442,  0.7595,  0.9250, -0.4242, -0.4242,
            -0.4242, -0.4242, -0.4242, -0.4242, -0.4242, -0.4242, -0.4242, -0.4242,
            -0.4242, -0.4242, -0.4242, -0.4242, -0.4242, -0.3733, -0.4242,  0.2504,
             1.2177,  1.1032,  1.4468,  1.8032,  1.5741,  1.6887,  1.2941,  1.5359,
             1.7141,  1.3577, -0.4242, -0.4242, -0.4242, -0.4242, -0.4242, -0.4242,
            -0.4242, -0.4242, -0.4242, -0.4242, -0.4242, -0.4242, -0.4242, -0.4242,
            -0.3988, -0.4242, -0.2842,  1.3196,  1.2305,  1.2050,  1.6123,  1.8160,
             1.5996,  1.7014,  1.8414,  1.4723,  1.4978,  1.4087, -0.4242, -0.4242,
            -0.4242, -0.4242, -0.4242, -0.4242, -0.4242, -0.4242, -0.4115, -0.4242,
            -0.3988, -0.4115, -0.4242, -0.3860, -0.4242, -0.4242,  1.0395,  1.0268,
             0.9250,  1.3196,  1.7141,  1.5232,  1.5614,  1.6759,  1.7014,  1.3959,
             1.5741,  1.5868, -0.2842, -0.4242, -0.4242, -0.4242, -0.4242, -0.4242,
            -0.4115, -0.4242, -0.4242, -0.4242, -0.4242, -0.4242, -0.3860, -0.4242,
            -0.4242,  0.7086,  1.3450,  0.7213,  0.7722,  1.5232,  1.4723,  1.2432,
             1.4978,  1.7269,  1.7650,  1.3959,  1.5996,  1.7269,  0.1867, -0.4242,
            -0.4242, -0.4242, -0.4242, -0.4242, -0.4242, -0.4242, -0.3988, -0.3733,
            -0.4115, -0.4242, -0.4242, -0.4242,  0.8232,  1.3068,  0.9759,  0.9632,
             0.9759,  1.6378,  1.2941,  1.4087,  1.4723,  1.5996,  1.7014,  1.4087,
             1.5868,  1.7269,  1.0904, -0.4242, -0.4242, -0.4242, -0.3988, -0.3988,
            -0.4115, -0.3988, -0.4242, -0.4242, -0.4242, -0.4242, -0.0933,  0.9504,
             1.0650,  0.8359,  0.9886,  1.0650,  1.3068,  1.5614,  1.2814,  1.5359,
             1.5359,  1.5614,  1.6123,  1.3705,  1.4468,  1.5614,  1.8414, -0.4242,
            -0.3860, -0.4242, -0.4242, -0.4242, -0.4242, -0.4242, -0.4242, -0.1569,
             0.2504,  0.7468,  1.0650,  0.9886,  0.8868,  1.0395,  1.2177,  1.2814,
             1.3959,  1.5359,  1.6759,  1.7396,  1.5359,  1.4978,  1.5359,  1.3959,
             1.3323,  1.4850,  1.6759,  0.1231, -0.4242, -0.4242, -0.1315,  0.2631,
             0.4031,  0.5431,  0.6577,  1.0777,  1.2050,  1.1414,  0.9886,  1.0141,
             1.0777,  1.1923,  1.1668,  1.3450,  1.2686,  1.3068,  1.6123,  1.3577,
             1.5487,  1.6250,  1.4087,  1.5487,  1.7650,  1.6250,  1.9814,  0.3649,
            -0.4242,  0.4413,  0.7722,  0.7213,  0.9886,  1.0268,  0.9886,  1.0268,
             1.0395,  1.1923,  1.2941,  1.3068,  1.3959,  1.1795,  1.1923,  1.4978,
             1.5359,  1.3959,  1.4596,  1.1668,  1.6378,  1.6378,  1.4087,  1.3323,
             1.5232,  1.6378,  2.0705,  0.3140,  0.4668,  1.7269,  1.2177,  0.8995,
             0.8232,  0.8486,  0.7722,  0.8104,  0.8232,  0.8741,  0.9504,  0.9250,
             1.0904,  1.1032,  1.2177,  1.4723,  1.5614,  1.7014,  1.9942,  1.9942,
             2.0705,  2.0960,  2.0960,  1.9560,  2.0832,  1.9814,  1.9178,  0.0340,
            -0.2206,  1.1795,  1.7523,  1.9687,  1.9687,  1.9178,  1.7523,  1.5232,
             1.2941,  1.1032,  1.1795,  1.1923,  1.4341,  1.9305,  2.0578,  2.2360,
             2.2233,  2.8215,  2.2360,  1.8287,  2.6942,  2.7833,  2.7706,  2.7706,
             2.7197,  2.3760,  2.1978,  0.1995, -0.4242, -0.4242, -0.4242, -0.2715,
             0.4286,  0.9250,  1.6632,  1.9305,  2.1087,  2.2487,  2.2614,  2.2487,
             2.2233,  1.9942,  1.4850,  0.6195, -0.3224, -0.4242, -0.4242, -0.4242,
             1.8414,  2.2233,  1.9687,  1.8032,  1.6378,  1.5868,  1.4978, -0.2842,
            -0.4242, -0.4242, -0.4242, -0.4242, -0.4242, -0.4242, -0.4242, -0.4242,
            -0.4242, -0.4242, -0.4242, -0.4242, -0.4242, -0.4242, -0.4242, -0.4242,
            -0.4242, -0.4242, -0.4242, -0.4242, -0.4242, -0.4242, -0.4242, -0.4242,
            -0.4242, -0.4242, -0.4242, -0.4242, -0.4242, -0.4242, -0.4242, -0.4242,
            -0.4242, -0.4242, -0.4242, -0.4242, -0.4242, -0.4242, -0.4242, -0.4242,
            -0.4242, -0.4242, -0.4242, -0.4242, -0.4242, -0.4242, -0.4242, -0.4242,
            -0.4242, -0.4242, -0.4242, -0.4242, -0.4242, -0.4242, -0.4242, -0.4242,
            -0.4242, -0.4242, -0.4242, -0.4242, -0.4242, -0.4242, -0.4242, -0.4242,
            -0.4242, -0.4242, -0.4242, -0.4242, -0.4242, -0.4242, -0.4242, -0.4242,
            -0.4242, -0.4242, -0.4242, -0.4242, -0.4242, -0.4242, -0.4242, -0.4242,
            -0.4242, -0.4242, -0.4242, -0.4242, -0.4242, -0.4242, -0.4242, -0.4242,
            -0.4242, -0.4242, -0.4242, -0.4242, -0.4242, -0.4242, -0.4242, -0.4242,
            -0.4242, -0.4242, -0.4242, -0.4242, -0.4242, -0.4242, -0.4242, -0.4242,
            -0.4242, -0.4242, -0.4242, -0.4242, -0.4242, -0.4242, -0.4242, -0.4242,
            -0.4242, -0.4242, -0.4242, -0.4242, -0.4242, -0.4242, -0.4242, -0.4242,
            -0.4242, -0.4242, -0.4242, -0.4242, -0.4242, -0.4242, -0.4242, -0.4242,
            -0.4242, -0.4242, -0.4242, -0.4242, -0.4242, -0.4242, -0.4242, -0.4242,
            -0.4242, -0.4242, -0.4242, -0.4242, -0.4242, -0.4242, -0.4242, -0.4242,
            -0.4242, -0.4242, -0.4242, -0.4242, -0.4242, -0.4242, -0.4242, -0.4242,
            -0.4242, -0.4242, -0.4242, -0.4242, -0.4242, -0.4242, -0.4242, -0.4242,
            -0.4242, -0.4242, -0.4242, -0.4242, -0.4242, -0.4242, -0.4242, -0.4242
        ]
        
        // Test CPU version
        var startTime = DispatchTime.now()
        var network: TestBigConvNetwork = TestBigConvNetwork()
        var initElapsedTime = Double(
            DispatchTime.now().uptimeNanoseconds - startTime.uptimeNanoseconds) / 1000000000;
        
        startTime = DispatchTime.now()
        var input: Tensor<DataType> = Tensor<DataType>(shape: [1, 1, 28, 28], data: inputData)
        var forwardResult: Tensor<DataType> = network.forward(input: input)
        var forwardElapsedTime = Double(
            DispatchTime.now().uptimeNanoseconds - startTime.uptimeNanoseconds) / 1000000000;

        forwardResult.printData()
        var metricString =
            "Metrics:\n" +
            "Init Elapsed Time: \(initElapsedTime) sec\n" +
            "Forward Elapsed Time: \(forwardElapsedTime) sec\n"
        
        metricString += "--------------\n"
        
        // Test GPU version
        startTime = DispatchTime.now()
        network = TestBigConvNetwork(gpu: true)
        initElapsedTime = Double(
            DispatchTime.now().uptimeNanoseconds - startTime.uptimeNanoseconds) / 1000000000;
        
        startTime = DispatchTime.now()
        input = Tensor<DataType>(shape: [1, 1, 28, 28], data: inputData)
        input.copyToGPU()
        forwardResult = network.forward(input: input)
        forwardResult.copyToCPU()
        forwardElapsedTime = Double(
            DispatchTime.now().uptimeNanoseconds - startTime.uptimeNanoseconds) / 1000000000;

        forwardResult.printData()
        metricString += (
            "Metrics:\n" +
            "Init Elapsed Time: \(initElapsedTime) sec\n" +
            "Forward Elapsed Time: \(forwardElapsedTime) sec\n"
        )
        
        benchmarkTextLabel.text = metricString
    }
}
