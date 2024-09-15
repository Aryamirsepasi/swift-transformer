import Foundation
import Accelerate
import MLX

protocol Optimizer {
    var alpha: Float { get set }
    func update(gradient: MLXArray, weights: inout MLXArray, v: inout MLXArray, m: inout MLXArray, vHat: inout MLXArray, mHat: inout MLXArray, t: Int) -> (MLXArray, MLXArray, MLXArray, MLXArray, MLXArray, Int)
}

class SGD: Optimizer {
    var alpha: Float
    
    init(alpha: Float = 0.001) {
        self.alpha = alpha
    }
    
    func update(gradient: MLXArray, weights: inout MLXArray, v: inout MLXArray, m: inout MLXArray, vHat: inout MLXArray, mHat: inout MLXArray, t: Int) -> (MLXArray,MLXArray, MLXArray, MLXArray, MLXArray, Int) {
        
        //print("entered SGD update")

        
        weights -= gradient * alpha
        
        
        //print("exited SGD update")

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
    
    func update(gradient: MLXArray, weights: inout MLXArray, v: inout MLXArray, m: inout MLXArray, vHat: inout MLXArray, mHat: inout MLXArray, t: Int) -> (MLXArray, MLXArray, MLXArray, MLXArray, MLXArray, Int) {
        
        //print("entered Adam update")
        
        //print("m: ", m)
        //print("beta: ", beta)
        //print("gradient: ", gradient)


        m = beta * m + (1 - beta) * gradient

        v = beta2 * v + (1 - beta2) * MLX.pow(gradient, 2)

        mHat = m / (1 - Float(pow(Double(beta), Double(t))))
        vHat = v / (1 - Float(pow(Double(beta2), Double(t))))


        var temp = alpha * mHat / (MLX.sqrt(vHat) + epsilon)

        weights -= temp
        

        //print("exited Adam update")

        return (weights, v, m, vHat, mHat , t)
    }
}


class Noam: Optimizer {
    var alpha: Float
    var optimizer: Optimizer
    var modelDim: Float
    var scaleFactor: Float
    var warmupSteps: Int
    var stepsNum: Int = 0
    
    init(optimizer: Optimizer, modelDim: Float, scaleFactor: Float = 1, warmupSteps: Int = 4000) {
        self.optimizer = optimizer
        self.modelDim = modelDim
        self.scaleFactor = scaleFactor
        self.warmupSteps = warmupSteps
        self.alpha = 0.0  // Initial alpha value, will be updated
    }
    
    func update(gradient: MLXArray, weights: inout MLXArray, v: inout MLXArray, m: inout MLXArray, vHat: inout MLXArray, mHat: inout MLXArray, t: Int) -> (MLXArray, MLXArray, MLXArray, MLXArray, MLXArray, Int) {
        
        //print("entered Noam update")

        stepsNum += 1
        var model_dim_component = Float(pow(modelDim, -0.5))
        var steps_num_component = Int(pow(Double(stepsNum), -0.5))
        var warmup_component = stepsNum * Int(pow(Double(warmupSteps), -1.5))
            
        var min_component = min(steps_num_component, warmup_component)
            
        var lr = scaleFactor * Float(model_dim_component) * Float(min_component)
            
        self.optimizer.alpha = lr
        
        //print("exited Noam update")

        return self.optimizer.update(gradient: gradient, weights: &weights, v: &v, m: &m, vHat: &vHat, mHat: &mHat, t: t)
    }
}


let optimizers: [String: Optimizer] = [
    "sgd": SGD(),
    //"noam": Noam(optimizer: any Optimizer, modelDim: <#Float#>),
    "adam": Adam(),
]
