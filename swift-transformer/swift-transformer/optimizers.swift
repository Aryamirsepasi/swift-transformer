import Foundation
import Accelerate

protocol Optimizer {
    var alpha: Float { get set }
    func update(gradient: [Float], weights: inout [Float], v: inout [Float], m: inout [Float], vHat: inout [Float], mHat: inout [Float], t: Int) -> ([Float], [Float], [Float], [Float], [Float], Int)
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
    
    func update(gradient: [Float], weights: inout [Float], v: inout [Float], m: inout [Float], vHat: inout [Float], mHat: inout [Float], t: Int) -> ([Float], [Float], [Float], [Float], [Float], Int) {
        var oneMinusBeta = 1 - beta
        var oneMinusBeta2 = 1 - beta2
         
        // Update biased first moment estimate
        vDSP_vsmul(m, 1, &beta, &m, 1, vDSP_Length(m.count))
        var scaledGradient = [Float](repeating: 0, count: gradient.count)
        vDSP_vsmul(gradient, 1, &oneMinusBeta, &scaledGradient, 1, vDSP_Length(gradient.count))
        vDSP_vadd(m, 1, scaledGradient, 1, &m, 1, vDSP_Length(m.count))
        
        // Update biased second raw moment estimate
        var gradSquared = [Float](repeating: 0, count: gradient.count)
        vDSP_vsq(gradient, 1, &gradSquared, 1, vDSP_Length(gradient.count))
        vDSP_vsmul(v, 1, &beta2, &v, 1, vDSP_Length(v.count))
        var scaledGradSquared = [Float](repeating: 0, count: gradSquared.count)
        vDSP_vsmul(gradSquared, 1, &oneMinusBeta2, &scaledGradSquared, 1, vDSP_Length(gradSquared.count))
        vDSP_vadd(v, 1, scaledGradSquared, 1, &v, 1, vDSP_Length(v.count))
        
        // Compute bias-corrected first moment estimate
        var mHatCorrected = m
        let betaTPower = pow(beta, Float(t))
        var betaCorrection = 1 / (1 - betaTPower)
        vDSP_vsmul(m, 1, &betaCorrection, &mHatCorrected, 1, vDSP_Length(m.count))
        
        // Compute bias-corrected second raw moment estimate
        var vHatCorrected = v
        let beta2TPower = pow(beta2, Float(t))
        var beta2Correction = 1 / (1 - beta2TPower)
        vDSP_vsmul(v, 1, &beta2Correction, &vHatCorrected, 1, vDSP_Length(v.count))
        
        // Update weights
        var sqrtVHatPlusEpsilon = [Float](repeating: epsilon, count: vHatCorrected.count)
        vDSP_vsadd(vHatCorrected, 1, &epsilon, &sqrtVHatPlusEpsilon, 1, vDSP_Length(vHatCorrected.count))
        vvsqrtf(&sqrtVHatPlusEpsilon, sqrtVHatPlusEpsilon, [Int32(sqrtVHatPlusEpsilon.count)])
        var updateValues = [Float](repeating: 0, count: weights.count)
        vDSP_vdiv(sqrtVHatPlusEpsilon, 1, mHatCorrected, 1, &updateValues, 1, vDSP_Length(updateValues.count))
        var negAlpha: Float = -alpha
        vDSP_vsma(updateValues, 1, &negAlpha, weights, 1, &weights, 1, vDSP_Length(weights.count))
        
        return (weights, v, m, vHatCorrected, mHatCorrected , t)
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
    
    func update(gradient: [Float], weights: inout [Float], v: inout [Float], m: inout [Float], vHat: inout [Float], mHat: inout [Float], t: Int) -> ([Float], [Float], [Float], [Float], [Float], Int) {
        stepsNum += 1
        let factor = scaleFactor * pow(modelDim, -0.5) * min(pow(Float(stepsNum), -0.5), Float(stepsNum) * pow(Float(warmupSteps), -1.5))
        optimizer.alpha = factor
        self.alpha = factor
        return optimizer.update(gradient: gradient, weights: &weights, v: &v, m: &m, vHat: &vHat, mHat: &mHat, t: t)
    }
}


let optimizers: [String: Optimizer] = [
    "adam": Adam(),
]
