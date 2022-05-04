//
//  Metrics.swift
//  MCNN iOS
//
//  Created by BerthCloud Chou on 2022/5/2.
//

import Foundation

public struct Metrics: CustomStringConvertible {
    public let initElapseNanosecs: UInt64
    public let forwardElapseNanosecs: UInt64
    
    public var description: String {
        return (
            "Metrics:\n" +
            "Init Elapsed Time: \(Double(initElapseNanosecs) / 1000000) ms\n" +
            "Forward Elapsed Time: \(Double(forwardElapseNanosecs) / 1000000) ms\n"
        )
    }
}
