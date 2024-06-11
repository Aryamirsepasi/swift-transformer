import Foundation
import Accelerate

protocol LossFunction {
    func loss(y: [Float], t: [Float]) -> [Float]
    func derivative(y: [Float], t: [Float]) -> [Float]
}

class MSE: LossFunction {
    func loss(y: [Float], t: [Float]) -> [Float] {
        var result = [Float](repeating: 0.0, count: y.count)
        var difference = [Float](repeating: 0.0, count: y.count)
        vDSP_vsub(t, 1, y, 1, &difference, 1, vDSP_Length(y.count))
        vDSP_vsq(difference, 1, &result, 1, vDSP_Length(y.count))
        return result
    }
    
    func derivative(y: [Float], t: [Float]) -> [Float] {
        var result = [Float](repeating: 0.0, count: y.count)
        var scale: Float = -2.0 / Float(y.count)
        vDSP_vsub(t, 1, y, 1, &result, 1, vDSP_Length(y.count))
        vDSP_vsmul(result, 1, &scale, &result, 1, vDSP_Length(y.count))
        return result
    }
}

class BinaryCrossEntropy: LossFunction {
    func loss(y: [Float], t: [Float]) -> [Float] {
        let epsilon: Float = 1e-8
        var result = [Float](repeating: 0.0, count: y.count)
        var temp1 = [Float](repeating: 0.0, count: y.count)
        var temp2 = [Float](repeating: 0.0, count: y.count)
        
        // Compute log(y + epsilon)
        var yPlusEps = y.map { $0 + epsilon }
        vvlogf(&temp1, yPlusEps, [Int32(y.count)])
        
        // Compute log(1 - y + epsilon)
        var oneMinusYPlusEps = y.map { 1 - $0 + epsilon }
        vvlogf(&temp2, oneMinusYPlusEps, [Int32(y.count)])
        
        // Calculate -t * log(y + epsilon)
        vDSP_vmul(t, 1, temp1, 1, &temp1, 1, vDSP_Length(y.count))
        
        // Calculate -(1 - t) * log(1 - y + epsilon)
        let oneMinusT = t.map { 1 - $0 }
        vDSP_vmul(oneMinusT, 1, temp2, 1, &temp2, 1, vDSP_Length(y.count))
        
        // Sum both results into `result`
        vDSP_vadd(temp1, 1, temp2, 1, &result, 1, vDSP_Length(y.count))
        
        // Negate the result as we need -(result)
        vDSP_vneg(result, 1, &result, 1, vDSP_Length(y.count))
        
        return result
    }
    
    func derivative(y: [Float], t: [Float]) -> [Float] {
        let epsilon: Float = 1e-8
        var result = [Float](repeating: 0.0, count: y.count)
        var yPlusEps = y.map { $0 + epsilon }
        var oneMinusYPlusEps = y.map { 1 - $0 + epsilon }
        vDSP_vdiv(t, 1, yPlusEps, 1, &result, 1, vDSP_Length(y.count))
        var temp = [Float](repeating: 0.0, count: y.count)
        vDSP_vdiv(t.map { 1 - $0 }, 1, oneMinusYPlusEps, 1, &temp, 1, vDSP_Length(y.count))
        vDSP_vsub(result, 1, temp, 1, &result, 1, vDSP_Length(y.count))
        vDSP_vneg(result, 1, &result, 1, vDSP_Length(y.count))
        return result
    }
}

class CategoricalCrossEntropy: LossFunction {
    var ignoreIndex: Int?
    
    init(ignoreIndex: Int? = nil) {
        self.ignoreIndex = ignoreIndex
    }
    
    func loss(y: [Float], t: [Float]) -> [Float] {
        var result = [Float](repeating: 0.0, count: y.count)
        for i in 0..<y.count {
            if let ignore = ignoreIndex, t[i] == Float(ignore) {
                result[i] = 0.0
            } else {
                result[i] = -t[i] * log(y[i])
            }
        }
        return result
    }
    
    func derivative(y: [Float], t: [Float]) -> [Float] {
        var result = [Float](repeating: 0.0, count: y.count)
        for i in 0..<y.count {
            if let ignore = ignoreIndex, t[i] == Float(ignore) {
                result[i] = 0.0
            } else {
                result[i] = -t[i] / y[i]
            }
        }
        return result
    }
}

class CrossEntropy: LossFunction {
    var ignoreIndex: Int?
    let logSoftmax = LogSoftmax()
    
    init(ignoreIndex: Int? = nil) {
        self.ignoreIndex = ignoreIndex
    }
    
    func loss(y: [Float], t: [Float]) -> [Float] {
        let logSoft = logSoftmax.forward(x: y)
        var result = [Float](repeating: 0.0, count: y.count)
        for i in 0..<t.count {
            let index = Int(t[i])
            if index != ignoreIndex {
                result[i] = -logSoft[index]
            }
        }
        return result
    }
    
    func derivative(y: [Float], t: [Float]) -> [Float] {
        let batchSize = y.count
        let err: Float = 1 / Float(batchSize)
        let nllLossDer = [Float](repeating: -err, count: y.count)
        
        // Use reshape to handle the shape transformation
        let reshapedT = reshape(t, newShape: [batchSize, 1])
        
        let outputErr = logSoftmax.backward(grad: nllLossDer)
        var result = [Float](repeating: 0.0, count: y.count)
        for i in 0..<t.count {
            let index = Int(t[i])
            if reshapedT[i] == Float(ignoreIndex ?? -1) {
                result[i] = 0
            } else {
                result[i] = outputErr[i]
            }
        }
        return result
    }
}

let lossFunctions: [String: LossFunction] = [
    "mse": MSE(),
    "binary_crossentropy": BinaryCrossEntropy(),
    "categorical_crossentropy": CategoricalCrossEntropy()
]
