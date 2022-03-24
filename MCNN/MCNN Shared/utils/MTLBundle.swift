//
//  MTLBundle.swift
//  MCNN iOS
//
//  Created by BerthCloud Chou on 2022/3/24.
//

import Foundation
import Metal

public class MTLBundle {
    internal let mtlDevice: MTLDevice!
    internal let mtlCommandQueue: MTLCommandQueue!
    internal let mtlLibrary: MTLLibrary!
    
    public init() {
        mtlDevice = MTLCreateSystemDefaultDevice()
        mtlCommandQueue = mtlDevice.makeCommandQueue()
        mtlLibrary = mtlDevice.makeDefaultLibrary()
    }
}
