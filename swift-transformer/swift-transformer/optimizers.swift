import Foundation
import Accelerate
import MLX

protocol Optimizer {
    var alpha: Float { get set }
    func update(gradient: MLXArray, weights: MLXArray, v: MLXArray, m: MLXArray, vHat: MLXArray, mHat: MLXArray,  t: Int)-> (MLXArray,MLXArray, MLXArray, MLXArray, MLXArray, Int)
}

class SGD: Optimizer {
    var alpha: Float
    
    init(alpha: Float = 0.001) {
        self.alpha = alpha
    }
    
    func update(gradient: MLXArray, weights:  MLXArray, v:  MLXArray, m:  MLXArray, vHat:  MLXArray, mHat:  MLXArray, t: Int) -> (MLXArray,MLXArray, MLXArray, MLXArray, MLXArray, Int) {
                
        weights -= gradient * alpha
                
        return (weights, v, m, vHat, mHat, t)
        
    }
}


class Adam: Optimizer { //after
    var alpha: Float
    var beta: Float
    var beta2: Float
    var epsilon: Float
    
    init(alpha: Float = 0.001, beta: Float = 0.9, beta2: Float = 0.999, epsilon: Float = 1e-9) {
        self.alpha = alpha
        self.beta = beta
        self.beta2 = beta2
        self.epsilon = epsilon
    }
    
    func update(gradient: MLXArray, weights:  MLXArray, v:  MLXArray, m:  MLXArray, vHat:  MLXArray, mHat:  MLXArray, t: Int) -> (MLXArray, MLXArray, MLXArray, MLXArray, MLXArray, Int) {
        
        let mvar = MLX.add((beta * m),((1 - beta) * gradient))
        
        let vvar = MLX.add((beta2 * v),((1 - beta2) * MLX.pow(gradient, 2, stream: .gpu)))
        
        let mHatvar = mvar / (1 - Float(pow(Double(beta), Double(t))))
        let vHatvar = vvar / (1 - Float(pow(Double(beta2), Double(t))))
        
        
        let temp = alpha * mHatvar / (MLX.sqrt(vHatvar, stream: .gpu) + epsilon)
        
        weights -= temp
        
        return (weights, v, m, vHat, mHat , t)
    }
}


class Noam: Optimizer {
    var alpha: Float
    var optimizer: Optimizer
    var modelDim: Float
    var scaleFactor: Float
    var warmupSteps: Int
    var stepsNum: Int
    
    init(optimizer: Optimizer, modelDim: Float, scaleFactor: Float = 1, warmupSteps: Int = 4000) {
        self.optimizer = optimizer
        self.modelDim = modelDim
        self.scaleFactor = scaleFactor
        self.warmupSteps = warmupSteps
        self.alpha = self.optimizer.alpha
        self.stepsNum = 0
    }
    
    func update(gradient: MLXArray, weights:  MLXArray, v:  MLXArray, m:  MLXArray, vHat:  MLXArray, mHat:  MLXArray, t: Int) -> (MLXArray, MLXArray, MLXArray, MLXArray, MLXArray, Int) {
                
        stepsNum += 1
        let arg1 = pow(Float(stepsNum), -0.5)
        let arg2 = Float(stepsNum) * pow(Float(warmupSteps), -1.5)
        let lr = scaleFactor * pow(modelDim, -0.5) * min(arg1, arg2)
        optimizer.alpha = lr
                
        return self.optimizer.update(gradient: gradient, weights: weights, v: v, m: m, vHat: vHat, mHat: mHat, t: t)
    }
}


let optimizers: [String: Optimizer] = [
    "sgd": SGD(),
    //"noam": Noam(optimizer: any Optimizer, modelDim: <#Float#>),
    "adam": Adam(),
]

