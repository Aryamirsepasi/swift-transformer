import Foundation
import Accelerate

//needed
class PositionalEncoding {
    var dModel: Int
    var dropoutRate: Float
    var maxLen: Int
    var dataType: [Float]
    var pe: [[[Float]]]
    
    init(maxLen: Int, dModel: Int, dropoutRate: Float = 0.1, dataType: [Float] = []) {
        self.dModel = dModel
        self.dropoutRate = dropoutRate
        self.maxLen = maxLen
        self.dataType = dataType
        
        var pe = [[[Float]]](repeating: [[Float]](repeating: [Float](repeating: 0.0, count: dModel), count: 1), count: maxLen)
        
        for pos in 0..<maxLen {
            for i in stride(from: 0, to: dModel, by: 2) {
                let divTerm = exp(Float(i) * (-log(10000.0) / Float(dModel)))
                pe[pos][0][i] = sin(Float(pos) * divTerm)
                if i + 1 < dModel {
                    pe[pos][0][i + 1] = cos(Float(pos) * divTerm)
                }
            }
        }
        self.pe = pe
    }
    
    func forward(x: [[[Float]]]) -> [[[Float]]] {
        var result = x
        let batchSize = x.count
        let seqLen = x[0].count
        
        for b in 0..<batchSize {
            for s in 0..<seqLen {
                let peIndex = s
                for d in 0..<dModel {
                    if peIndex < pe.count && d < pe[peIndex][0].count {
                        result[b][s][d] += pe[peIndex][0][d]
                    }
                }
            }
        }
        return result
    }

    func backward(error: [[[Float]]]) -> [[[Float]]] {
        return error
    }
}
