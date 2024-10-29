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
    private var currentMaskShape: [Int]? // Track current mask shape
    
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
        autoreleasepool {
            
            guard training else { return X }
            
            // Only generate new mask if shape changes
            if self.currentMaskShape != X.shape {
                self.mask = MLXRandom.bernoulli(1 - self.rate, X.shape, stream: .gpu)
                    .asType(self.dataType)
                self.currentMaskShape = X.shape
            }
            
            return X * self.mask!
        }
        
        
        
    }
    
    func backward(_ error: MLXArray) -> MLXArray {
        autoreleasepool {
            
            let outputError = error * self.mask!
            
            return outputError
        }
    }
    
}
