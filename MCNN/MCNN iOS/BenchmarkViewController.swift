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
        var startTime = DispatchTime.now()
        let network: Network = Network()
        let initElapsedTime = Double(
                                    DispatchTime.now().uptimeNanoseconds -
                                    startTime.uptimeNanoseconds) / 1000000000;
        
        startTime = DispatchTime.now()
        let forwardResult: Tensor = network.forward(input: Tensor(nDim: 5))
        let forwardElapsedTime = Double(
                                    DispatchTime.now().uptimeNanoseconds -
                                    startTime.uptimeNanoseconds) / 1000000000;

        benchmarkTextLabel.text =
            "Result:\n\(forwardResult)\n\n" +
            "Metrics:\n" +
            "Init Elapsed Time: \(initElapsedTime) sec\n" +
            "Forward Elapsed Time: \(forwardElapsedTime) sec\n";
    }
}
