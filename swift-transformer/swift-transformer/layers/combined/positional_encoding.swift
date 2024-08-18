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
        
        var pe = MLX.zeros([maxLen, dModel])
        
        var position = MLXArray(0 ..< maxLen)[0..., .newAxis]
        
        var divTermValues = stride(from: 0, to: dModel, by: 2).map { Float($0) * (-log(10000.0) / Float(dModel)) }
        var divTerm = MLX.exp(MLXArray(divTermValues))
        
        pe[0..., 0..<2] = MLX.sin(position * divTerm)
        pe[0..., 1..<2] = MLX.cos(position * divTerm)
        
        self.pe = pe[0..., .newAxis, 0...]
    }
    
    func forward(x: MLXArray) -> MLXArray {
        var xvar = x
        
        xvar += self.pe[..<x.shape[0], 0...]
        
        return xvar
    }

    func backward(error: MLXArray) -> MLXArray {
        return error
    }
}
