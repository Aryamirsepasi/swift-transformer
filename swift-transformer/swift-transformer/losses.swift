import Foundation
import Accelerate

protocol LossFunction {
    func loss(y: [[Float]], t: [Float]) -> [Float]
    func derivative(y: [[Float]], t: [Float]) -> [[Float]]
}

class MSE: LossFunction {
    func loss(y: [[Float]], t: [Float]) -> [Float] {
        return zip(y, t).map { (yRow, tVal) in
            zip(yRow, Array(repeating: tVal, count: yRow.count)).map { (yVal, tVal) in
                pow(tVal - yVal, 2)
            }.reduce(0, +) / Float(yRow.count)
        }
    }
    
    func derivative(y: [[Float]], t: [Float]) -> [[Float]] {
        return zip(y, t).map { (yRow, tVal) in
            zip(yRow, Array(repeating: tVal, count: yRow.count)).map { (yVal, tVal) in
                -2 * (tVal - yVal) / Float(yRow.count)
            }
        }
    }
}

class BinaryCrossEntropy: LossFunction {
    func loss(y: [[Float]], t: [Float]) -> [Float] {
        let epsilon: Float = 1e-8
        return zip(y, t).map { (yRow, tVal) in
            zip(yRow, Array(repeating: tVal, count: yRow.count)).map { (yVal, tVal) in
                let yPlusEps = yVal + epsilon
                return -(tVal * log(yPlusEps) + (1 - tVal) * log(1 - yPlusEps))
            }.reduce(0, +) / Float(yRow.count)
        }
    }
    
    func derivative(y: [[Float]], t: [Float]) -> [[Float]] {
        let epsilon: Float = 1e-8
        return zip(y, t).map { (yRow, tVal) in
            zip(yRow, Array(repeating: tVal, count: yRow.count)).map { (yVal, tVal) in
                let yPlusEps = yVal + epsilon
                return -tVal / yPlusEps + (1 - tVal) / (1 - yPlusEps)
            }
        }
    }
}

class CategoricalCrossEntropy: LossFunction {
    var ignoreIndex: Int?
    
    init(ignoreIndex: Int? = nil) {
        self.ignoreIndex = ignoreIndex
    }
    
    func loss(y: [[Float]], t: [Float]) -> [Float] {
        return zip(y, t).map { (yRow, tVal) in
            zip(yRow, Array(repeating: tVal, count: yRow.count)).map { (yVal, tVal) in
                if let ignore = ignoreIndex, Int(tVal) == ignore {
                    return 0.0
                } else {
                    return -tVal * log(yVal)
                }
            }.reduce(0, +) / Float(yRow.count)
        }
    }
    
    func derivative(y: [[Float]], t: [Float]) -> [[Float]] {
        return zip(y, t).map { (yRow, tVal) in
            zip(yRow, Array(repeating: tVal, count: yRow.count)).map { (yVal, tVal) in
                if let ignore = ignoreIndex, Int(tVal) == ignore {
                    return 0.0
                } else {
                    return -tVal / yVal
                }
            }
        }
    }
}

class CrossEntropy: LossFunction { //needed CrossEntropyLoss
    var ignoreIndex: Int?
    let logSoftmax = LogSoftmax()
    
    init(ignoreIndex: Int? = nil) {
        self.ignoreIndex = ignoreIndex
    }
    
    func loss(y: [[Float]], t: [Float]) -> [Float] {
        let logSoft = logSoftmax.forward(x: [y])[0]
        return zip(logSoft, t).map { (logSoftRow, tVal) in
            zip(logSoftRow, Array(repeating: tVal, count: logSoftRow.count)).map { (logSoftVal, tVal) in
                if Int(tVal) == ignoreIndex {
                    return 0.0
                } else {
                    return -logSoftVal
                }
            }.reduce(0, +) / Float(logSoftRow.count)
        }
    }
    
    func derivative(y: [[Float]], t: [Float]) -> [[Float]] {
        let batchSize = y.count
        let err: Float = 1 / Float(batchSize)
        let nllLossDer = Array(repeating: Array(repeating: -err, count: y[0].count), count: y.count)
        
        let outputErr = logSoftmax.backward(grad: [nllLossDer])[0]
        return zip(outputErr, t).map { (outputErrRow, tVal) in
            zip(outputErrRow, Array(repeating: tVal, count: outputErrRow.count)).map { (outputErrVal, tVal) in
                if Int(tVal) == ignoreIndex {
                    return 0.0
                } else {
                    return outputErrVal
                }
            }
        }
    }
}

let lossFunctions: [String: LossFunction] = [
    "mse": MSE(),
    "binary_crossentropy": BinaryCrossEntropy(),
    "categorical_crossentropy": CategoricalCrossEntropy(),
    "cross_entropy": CrossEntropy()
]
