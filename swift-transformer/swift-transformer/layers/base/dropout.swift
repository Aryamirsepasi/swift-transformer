import Foundation
import Accelerate
import MLX
import MLXRandom

//needed
class Dropout {
    var rate: Float
    var inputShape: [Int]
    var outputShape: [Int]
    var mask: MLXArray?
    var dataType: DType
    
    init(rate: Float = 0.1, dataType: DType = DType.float32) {
        
        self.rate = rate
        self.inputShape = []
        self.outputShape = []
        self.mask = nil
        self.dataType = dataType
        
    }
    
    func build() {
        
        self.outputShape = self.inputShape
        
    }
    
    func forward(X: MLXArray, training: Bool = true) -> MLXArray {
        
        if training {
            if self.mask == nil || self.mask!.shape != X.shape {
                // Only generate a new mask if necessary
                self.mask = MLXRandom.bernoulli(1 - self.rate, X.shape, stream: .gpu).asType(self.dataType)
            }
            
            let res = X * self.mask!
            
            return res
        }
        
        return X
        
        
        
    }
    
    func backward(_ error: MLXArray) -> MLXArray {
        
        let outputError = error * self.mask!
        
        return outputError
    }
    
}
