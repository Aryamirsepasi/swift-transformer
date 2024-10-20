import Foundation
import Accelerate
import MLX

//needed
class PositionalEncoding {
    var dModel: Int
    var dropoutRate: Float
    var maxLen: Int
    var dataType: DType
    var pe: MLXArray
    
    init(maxLen: Int, dModel: Int, dropoutRate: Float = 0.1, dataType: DType) {
        
        self.dModel = dModel
        self.dropoutRate = dropoutRate
        self.maxLen = maxLen
        self.dataType = dataType
        
        let pe = MLX.zeros([maxLen, dModel])
        
        let position = MLXArray(0 ..< maxLen)[0..., .newAxis]
        
        let divTermValues = stride(from: 0, to: dModel, by: 2).map { Float($0) * (-log(10000.0) / Float(dModel)) }
        
        let divTerm = MLX.exp(MLXArray(divTermValues))
        
        pe[0..., .stride(from: 0, to: pe.shape[1], by: 2)] = MLX.sin(position * divTerm, stream: .gpu)
        pe[0..., .stride(from: 1, to: pe.shape[1], by: 2)] = MLX.cos(position * divTerm, stream: .gpu)
        
        self.pe = pe[0..., .newAxis, 0...]
        
    }
    
    func forward(x: MLXArray) -> MLXArray {
        
        
        let xvar = x
        
        xvar += self.pe[..<xvar.shape[0], 0...]
        
        return xvar
    }
    
    func backward(error: MLXArray) -> MLXArray {
        return error
    }
}
