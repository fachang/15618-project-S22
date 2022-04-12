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

    @IBOutlet var benchmarkTextLabel: UILabel!
    @IBOutlet var runBtn: UIButton!

    override func viewDidLoad() {
        super.viewDidLoad()
        
        benchmarkTextLabel.text = "CNN Forward Benchmark"
        runBtn.setTitle("RUN", for: .normal)
    }
    
    @IBAction func runBenchmark(sender: UIButton) {
        // runLinearNetworkBenchmark()
        runConvNetworkBenchmark()
    }
    
    private func runLinearNetworkBenchmark() {
        var startTime = DispatchTime.now()
        let network: TestLinearNetwork = TestLinearNetwork()
        let initElapsedTime = Double(
                                    DispatchTime.now().uptimeNanoseconds -
                                    startTime.uptimeNanoseconds) / 1000000000;
        
        startTime = DispatchTime.now()
        let forwardResult: Tensor<Float16> = network.forward(
                                    input: Tensor<Float16>(shape: [10, 3], initValue: 1))
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
        var startTime = DispatchTime.now()
        let network: TestConvNetwork = TestConvNetwork()
        let initElapsedTime = Double(
                                    DispatchTime.now().uptimeNanoseconds -
                                    startTime.uptimeNanoseconds) / 1000000000;
        
        startTime = DispatchTime.now()
        let input: Tensor<Float16> = Tensor<Float16>(shape: [2, 3, 5, 5], initValue: 1)
        let forwardResult: Tensor<Float16> = network.forward(input: input)
        let forwardElapsedTime = Double(
                                    DispatchTime.now().uptimeNanoseconds -
                                    startTime.uptimeNanoseconds) / 1000000000;

        forwardResult.printData()
        benchmarkTextLabel.text =
            "Metrics:\n" +
            "Init Elapsed Time: \(initElapsedTime) sec\n" +
            "Forward Elapsed Time: \(forwardElapsedTime) sec\n";
    }
}
