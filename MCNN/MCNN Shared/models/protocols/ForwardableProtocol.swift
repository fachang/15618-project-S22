//
//  ForwardableProtocol.swift
//  MCNN iOS
//
//  Created by BerthCloud Chou on 2022/3/24.
//

import Foundation

public protocol ForwardableProtocol {
    func forward(input: Tensor) -> Tensor
}
