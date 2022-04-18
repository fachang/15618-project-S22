//
//  MTLBundle.swift
//  MCNN iOS
//
//  Created by BerthCloud Chou on 2022/3/24.
//

import Foundation
import Metal

internal struct MTLCommons {
    internal static let mtlDevice: MTLDevice! = MTLCreateSystemDefaultDevice()
    internal static let mtlCommandQueue: MTLCommandQueue! = mtlDevice.makeCommandQueue()
    internal static let defaultLib: MTLLibrary! = mtlDevice.makeDefaultLibrary()
}
