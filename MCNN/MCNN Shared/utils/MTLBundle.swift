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
    
    public init(mtlDevice: MTLDevice) {
        self.mtlDevice = mtlDevice
        mtlCommandQueue = mtlDevice.makeCommandQueue()
        mtlLibrary = mtlDevice.makeDefaultLibrary()
    }
    
    public func copyToGPU(ptr: UnsafeRawPointer, size: Int) -> MTLBuffer? {
        return mtlDevice.makeBuffer(
            bytes: ptr, length: size,
            options: MTLResourceOptions.cpuCacheModeWriteCombined)
    }
}
