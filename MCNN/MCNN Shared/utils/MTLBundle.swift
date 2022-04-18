//
//  MTLBundle.swift
//  MCNN iOS
//
//  Created by BerthCloud Chou on 2022/3/24.
//

import Foundation
import Metal

public class MTLBundle {
    internal static let mtlDevice: MTLDevice! = MTLCreateSystemDefaultDevice()
    internal static let mtlCommandQueue: MTLCommandQueue! = mtlDevice.makeCommandQueue()
    internal static let mtlLibrary: MTLLibrary! = mtlDevice.makeDefaultLibrary()
    
    public static func copyToGPU(ptr: UnsafeRawPointer, size: Int) -> MTLBuffer? {
        return mtlDevice.makeBuffer(
            bytes: ptr, length: size,
            options: MTLResourceOptions.cpuCacheModeWriteCombined)
    }
}
