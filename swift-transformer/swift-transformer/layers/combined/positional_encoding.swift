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
        
        print("entered positionalEncoding init")
        
        self.dModel = dModel
        self.dropoutRate = dropoutRate
        self.maxLen = maxLen
        self.dataType = dataType
        
        print("step11")
        var pe = MLX.zeros([maxLen, dModel])
        
        print("step12")
        var position = MLXArray(0 ..< maxLen)[0..., .newAxis]
        
        print("step13")
        var divTermValues = stride(from: 0, to: dModel, by: 2).map { Float($0) * (-log(10000.0) / Float(dModel)) }
        
        print("step14")
        var divTerm = MLX.exp(MLXArray(divTermValues))
        
        print("step15")
        pe[0..., .stride(from: 0, to: pe.shape[1], by: 2)] = MLX.sin(position * divTerm)
        pe[0..., .stride(from: 1, to: pe.shape[1], by: 2)] = MLX.cos(position * divTerm)
        
        print("step16")
        self.pe = pe[0..., .newAxis, 0...]
        
        print ("exited positionalEncoding init")
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
