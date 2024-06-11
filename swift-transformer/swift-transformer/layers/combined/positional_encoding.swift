import Foundation
import Accelerate

class PositionalEncoding {
    var dModel: Int
    var dropoutRate: Float
    var maxLen: Int
    var dataType: [Float]
    var pe: [[Float]]
    
    init(maxLen: Int, dModel: Int, dropoutRate: Float = 0.1, dataType: [Float] = []) {
        self.dModel = dModel
        self.dropoutRate = dropoutRate
        self.maxLen = maxLen
        self.dataType = dataType
        
        var pe = [[Float]](repeating: [Float](repeating: 0.0, count: dModel), count: maxLen)
        
        for pos in 0..<maxLen {
            for i in stride(from: 0, to: dModel, by: 2) {
                let divTerm = exp(Float(i) * (-log(10000.0) / Float(dModel)))
                pe[pos][i] = sin(Float(pos) * divTerm)
                if i + 1 < dModel {
                    pe[pos][i + 1] = cos(Float(pos) * divTerm)
                }
            }
        }
        self.pe = pe
    }
    
    func forward(x: inout [[Float]]) {
        let batchSize = x.count
        let seqLen = x[0].count / dModel
        
        for b in 0..<batchSize {
            for s in 0..<seqLen {
                let xIndex = s * dModel
                let peIndex = s
                let range = xIndex..<min(xIndex + dModel, x[b].count)
                
                if range.upperBound <= x[b].count && peIndex < pe.count {
                    var temp = [Float](repeating: 0, count: dModel)
                    x[b].withUnsafeBufferPointer { xBuffer in
                        pe[peIndex].withUnsafeBufferPointer { peBuffer in
                            vDSP_vadd(xBuffer.baseAddress! + xIndex, 1, peBuffer.baseAddress!, 1, &temp, 1, vDSP_Length(dModel))
                        }
                    }
                    x[b].replaceSubrange(range, with: temp)
                } else {
                    print("Range \(range) exceeds bounds for batch \(b), sequence \(s)")
                }
            }
        }
    }
    
    func backward(error: [[Float]]) -> [[Float]] {
        return error
    }
}
