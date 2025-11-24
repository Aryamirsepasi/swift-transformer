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
    private var scale: Float  // Scale factor for inverted dropout
    
    init(rate: Float = 0.1, dataType: DType = DType.float32) {
        
        self.rate = rate
        self.inputShape = []
        self.outputShape = []
        self.mask = nil
        self.dataType = dataType
        // Inverted dropout: scale by 1/(1-p) during training so no scaling needed at inference
        self.scale = rate < 1.0 ? 1.0 / (1.0 - rate) : 1.0
        
    }
    
    func build() {
        
        self.outputShape = self.inputShape
        
    }
    
    func forward(X: MLXArray, training: Bool = true) -> MLXArray {
        autoreleasepool {
            
            guard training && self.rate > 0 else { return X }
            
            // Always generate new mask for each forward pass during training
            // This ensures different dropout patterns for different forward calls
            self.mask = MLXRandom.bernoulli(1 - self.rate, X.shape, stream: .gpu)
                .asType(self.dataType)
            self.currentMaskShape = X.shape
            
            // Apply inverted dropout: mask and scale
            return (X * self.mask!) * self.scale
        }
    }
    
    func backward(_ error: MLXArray) -> MLXArray {
        autoreleasepool {
            guard let mask = self.mask else { return error }
            
            // Gradient flows through kept units, scaled by the same factor
            let outputError = (error * mask) * self.scale
            
            return outputError
        }
    }
    
}
