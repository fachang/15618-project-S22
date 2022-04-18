//
//  LinearLayerParams.swift
//  MCNN iOS
//
//  Created by BerthCloud Chou on 2022/4/17.
//

import Foundation

public struct LinearLayerParams {
    let batch_size: UInt32;
    let n_input_channel: UInt32;
    let n_output_channel: UInt32;
    let bias: Bool;
}
