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
        
        // Create position array [maxLen, 1]
        let position = MLXArray(0 ..< maxLen).reshaped([maxLen, 1]).asType(DType.float32)
        
        // Create div term for each dimension pair
        let divTermValues = stride(from: 0, to: dModel, by: 2).map { i -> Float in
            exp(Float(i) * (-log(10000.0) / Float(dModel)))
        }
        let divTerm = MLXArray(divTermValues).asType(DType.float32)  // Shape: [dModel/2]
        
        // Compute sin and cos positional encodings
        let angles = position * divTerm  // Shape: [maxLen, dModel/2]
        let sinPE = MLX.sin(angles, stream: .gpu)  // Shape: [maxLen, dModel/2]
        let cosPE = MLX.cos(angles, stream: .gpu)  // Shape: [maxLen, dModel/2]
        
        // Interleave sin and cos to create [maxLen, dModel]
        // Stack along a new axis then reshape
        let stacked = MLX.stacked([sinPE, cosPE], axis: 2)  // Shape: [maxLen, dModel/2, 2]
        self.pe = stacked.reshaped([maxLen, dModel]).asType(dataType)  // Shape: [maxLen, dModel]
    }
    
    func forward(x: MLXArray) -> MLXArray {
        autoreleasepool {
            // x has shape [batch_size, seq_len, dModel]
            // pe has shape [maxLen, dModel]
            // We need to add pe[:seq_len, :] to x, broadcasting over batch dimension
            let seqLen = x.shape[1]
            
            // Get positional encoding for this sequence length
            let posEncoding = self.pe[..<seqLen, 0...]  // Shape: [seq_len, dModel]
            
            // Add positional encoding (broadcasts over batch dimension)
            return x + posEncoding
        }
    }
    
    func backward(error: MLXArray) -> MLXArray {
        return error
    }
}
