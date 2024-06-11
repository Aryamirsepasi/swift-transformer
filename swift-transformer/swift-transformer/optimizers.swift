import Foundation
import Accelerate

protocol Optimizer {
    var alpha: Float { get set }
    func update(gradient: [Float], weights: inout [Float], v: inout [Float], m: inout [Float], vHat: inout [Float], mHat: inout [Float], t: Int) -> ([Float], [Float], [Float], [Float], [Float], Int)
}

class SGD: Optimizer {
    var alpha: Float
    
    init(alpha: Float = 0.001) {
        self.alpha = alpha
    }
    
    func update(gradient: [Float], weights: inout [Float], v: inout [Float], m: inout [Float], vHat: inout [Float], mHat: inout [Float], t: Int) -> ([Float], [Float], [Float], [Float], [Float], Int) {
        cblas_saxpy(Int32(weights.count), -alpha, gradient, 1, &weights, 1)
        return (weights, v, m, vHat, mHat, t)
    }
}

class Momentum: Optimizer {
    var alpha: Float
    var beta: Float
    
    init(alpha: Float = 0.01, beta: Float = 0.9) {
        self.alpha = alpha
        self.beta = beta
    }
    
    func update(gradient: [Float], weights: inout [Float], v: inout [Float], m: inout [Float], vHat: inout [Float], mHat: inout [Float], t: Int) -> ([Float], [Float], [Float], [Float], [Float], Int) {
        var betaV = [Float](repeating: 0, count: v.count)
        vDSP_vsmul(v, 1, &beta, &betaV, 1, vDSP_Length(v.count))
        var oneMinusBetaGrad = [Float](repeating: 0, count: gradient.count)
        var oneMinusBeta = 1 - beta
        vDSP_vsmul(gradient, 1, &oneMinusBeta, &oneMinusBetaGrad, 1, vDSP_Length(gradient.count))
        vDSP_vadd(betaV, 1, oneMinusBetaGrad, 1, &v, 1, vDSP_Length(v.count))
        var negAlpha: Float = -alpha
        vDSP_vsma(v, 1, &negAlpha, weights, 1, &weights, 1, vDSP_Length(weights.count))
        return (weights, v, m, vHat, mHat, t)
    }
}

class RMSProp: Optimizer {
    var alpha: Float
    var beta: Float
    var epsilon: Float
    
    init(alpha: Float = 0.01, beta: Float = 0.9, epsilon: Float = 1e-9) {
        self.alpha = alpha
        self.beta = beta
        self.epsilon = epsilon
    }
    
    func update(gradient: [Float], weights: inout [Float], v: inout [Float], m: inout [Float], vHat: inout [Float], mHat: inout [Float], t: Int) -> ([Float], [Float], [Float], [Float], [Float], Int) {
        var gradSquared = [Float](repeating: 0, count: gradient.count)
        vDSP_vsq(gradient, 1, &gradSquared, 1, vDSP_Length(gradient.count))
        var betaV = [Float](repeating: 0, count: v.count)
        vDSP_vsmul(v, 1, &beta, &betaV, 1, vDSP_Length(v.count))
        var oneMinusBetaGradSquared = [Float](repeating: 0, count: gradSquared.count)
        var oneMinusBeta = 1 - beta
        vDSP_vsmul(gradSquared, 1, &oneMinusBeta, &oneMinusBetaGradSquared, 1, vDSP_Length(gradSquared.count))
        vDSP_vadd(betaV, 1, oneMinusBetaGradSquared, 1, &v, 1, vDSP_Length(v.count))
        
        var sqrtVPlusEpsilon = [Float](repeating: epsilon, count: v.count)
        vDSP_vsadd(v, 1, &epsilon, &sqrtVPlusEpsilon, 1, vDSP_Length(v.count))
        vvsqrtf(&sqrtVPlusEpsilon, sqrtVPlusEpsilon, [Int32(sqrtVPlusEpsilon.count)])
        
        var alphaGrad = [Float](repeating: 0, count: gradient.count)
        vDSP_vsdiv(gradient, 1, &sqrtVPlusEpsilon, &alphaGrad, 1, vDSP_Length(gradient.count))
        cblas_saxpy(Int32(weights.count), -alpha, alphaGrad, 1, &weights, 1)
        return (weights, v, m, vHat, mHat, t)
    }
}

class Adam: Optimizer {
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

class Nadam: Optimizer {
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
        vDSP_vsmul(gradSquared, 1, &oneMinusBeta, &scaledGradSquared, 1, vDSP_Length(gradSquared.count))
        vDSP_vadd(v, 1, scaledGradSquared, 1, &v, 1, vDSP_Length(v.count))
        
        // Compute bias-corrected first moment estimate
        var mHat = m
        let betaTPower = pow(beta, Float(t))
        var betaCorrection = 1 / (1 - betaTPower)
        vDSP_vsmul(m, 1, &betaCorrection, &mHat, 1, vDSP_Length(m.count))
        
        // Compute bias-corrected second raw moment estimate
        var vHat = v
        let beta2TPower = pow(beta2, Float(t))
        var beta2Correction = 1 / (1 - beta2TPower)
        vDSP_vsmul(v, 1, &beta2Correction, &vHat, 1, vDSP_Length(v.count))
        
        // Compute mHatImmediate
        var mHatImmediate = [Float](repeating: 0, count: m.count)
        vDSP_vadd(mHat, 1, scaledGradient, 1, &mHatImmediate, 1, vDSP_Length(m.count))
        
        // Update weights
        var sqrtVHatPlusEpsilon = [Float](repeating: epsilon, count: vHat.count)
        vDSP_vsadd(vHat, 1, &epsilon, &sqrtVHatPlusEpsilon, 1, vDSP_Length(vHat.count))
        vvsqrtf(&sqrtVHatPlusEpsilon, sqrtVHatPlusEpsilon, [Int32(sqrtVHatPlusEpsilon.count)])
        var updateValues = [Float](repeating: 0, count: weights.count)
        vDSP_vdiv(sqrtVHatPlusEpsilon, 1, mHatImmediate, 1, &updateValues, 1, vDSP_Length(updateValues.count))
        var negAlpha: Float = -alpha
        vDSP_vsma(updateValues, 1, &negAlpha, weights, 1, &weights, 1, vDSP_Length(weights.count))
        
        return (weights, v, m, vHat, mHat, t)
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
    "sgd": SGD(),
    "momentum": Momentum(),
    "rmsprop": RMSProp(),
    "adam": Adam(),
    "nadam": Nadam(),
]
