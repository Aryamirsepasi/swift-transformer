import Foundation
import Accelerate
import MLX

protocol LossFunction {
    func loss(y: MLXArray, t: MLXArray) -> MLXArray
    func derivative(y: MLXArray, t: MLXArray) -> MLXArray
}


class CrossEntropy: LossFunction {
    var ignore_index: Int
    let log_softmax = LogSoftmax()
    
    init(ignore_index: Int = 0) {
        self.ignore_index = ignore_index
    }
    
    private func createOneHot(indices: MLXArray, numClasses: Int) -> MLXArray {
        let batchSize = indices.shape[0]
        let oneHot = MLX.zeros([batchSize, numClasses], stream: .gpu).asType(DType.float32)
        
        // Set 1s at the specified indices
        oneHot[MLXArray(0..<indices.count), indices] = MLXArray(1.0)
        
        return oneHot
    }
    
    func loss(y: MLXArray, t: MLXArray) -> MLXArray {
        autoreleasepool {
            // Apply log softmax
            let logits = self.log_softmax.forward(x: y)
            let tInt = t.asType(Int.self)
            let numClasses = y.shape[y.ndim - 1]
            
            // Create mask for padding tokens
            let mask = (t .!= self.ignore_index).asType(y.dtype)
            
            // Create one-hot encoding of target
            let oneHot = createOneHot(indices: tInt, numClasses: numClasses)
            
            // Calculate negative log likelihood
            let nll = -MLX.sum(oneHot * logits, axis: -1)
            
            // Apply mask and calculate mean
            return (nll * mask).mean()
        }
    }
    
    func derivative(y: MLXArray, t: MLXArray) -> MLXArray {
        autoreleasepool {
            let probabilities = MLX.softmax(y, axis: -1)
            let tInt = t.asType(Int.self)
            let numClasses = y.shape[y.ndim - 1]
            
            // Create one-hot encoding of target
            let oneHot = createOneHot(indices: tInt, numClasses: numClasses)
            
            var grad = probabilities - oneHot
            
            // Apply ignore_index mask
            let mask = (t .!= self.ignore_index).asType(y.dtype).reshaped([-1, 1])
            grad = grad * mask
            
            return grad / Float(t.shape[0])
        }
    }
}

let lossFunctions: [String: LossFunction] = [
    "cross_entropy": CrossEntropy()
]
