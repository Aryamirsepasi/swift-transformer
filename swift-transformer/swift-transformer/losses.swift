import Foundation
import Accelerate

protocol LossFunction {
    func loss(y: [[Float]], t: [[Float]]) -> [[Float]]
    func derivative(y: [[Float]], t: [[Float]]) -> [[Float]]
}

class MSE: LossFunction {
    func loss(y: [[Float]], t: [[Float]]) -> [[Float]] {
        return zip(y, t).map { (yRow, tRow) in
            var result = [Float](repeating: 0.0, count: yRow.count)
            var difference = [Float](repeating: 0.0, count: yRow.count)
            vDSP_vsub(tRow, 1, yRow, 1, &difference, 1, vDSP_Length(yRow.count))
            vDSP_vsq(difference, 1, &result, 1, vDSP_Length(yRow.count))
            return result
        }
    }
    
    func derivative(y: [[Float]], t: [[Float]]) -> [[Float]] {
        return zip(y, t).map { (yRow, tRow) in
            var result = [Float](repeating: 0.0, count: yRow.count)
            var scale: Float = -2.0 / Float(yRow.count)
            vDSP_vsub(tRow, 1, yRow, 1, &result, 1, vDSP_Length(yRow.count))
            vDSP_vsmul(result, 1, &scale, &result, 1, vDSP_Length(yRow.count))
            return result
        }
    }
}

class BinaryCrossEntropy: LossFunction {
    func loss(y: [[Float]], t: [[Float]]) -> [[Float]] {
        let epsilon: Float = 1e-8
        return zip(y, t).map { (yRow, tRow) in
            var result = [Float](repeating: 0.0, count: yRow.count)
            var temp1 = [Float](repeating: 0.0, count: yRow.count)
            var temp2 = [Float](repeating: 0.0, count: yRow.count)
            
            var yPlusEps = yRow.map { $0 + epsilon }
            vvlogf(&temp1, yPlusEps, [Int32(yRow.count)])
            
            var oneMinusYPlusEps = yRow.map { 1 - $0 + epsilon }
            vvlogf(&temp2, oneMinusYPlusEps, [Int32(yRow.count)])
            
            vDSP_vmul(tRow, 1, temp1, 1, &temp1, 1, vDSP_Length(yRow.count))
            
            let oneMinusT = tRow.map { 1 - $0 }
            vDSP_vmul(oneMinusT, 1, temp2, 1, &temp2, 1, vDSP_Length(yRow.count))
            
            vDSP_vadd(temp1, 1, temp2, 1, &result, 1, vDSP_Length(yRow.count))
            
            vDSP_vneg(result, 1, &result, 1, vDSP_Length(yRow.count))
            
            return result
        }
    }
    
    func derivative(y: [[Float]], t: [[Float]]) -> [[Float]] {
        let epsilon: Float = 1e-8
        return zip(y, t).map { (yRow, tRow) in
            var result = [Float](repeating: 0.0, count: yRow.count)
            var yPlusEps = yRow.map { $0 + epsilon }
            var oneMinusYPlusEps = yRow.map { 1 - $0 + epsilon }
            vDSP_vdiv(tRow, 1, yPlusEps, 1, &result, 1, vDSP_Length(yRow.count))
            var temp = [Float](repeating: 0.0, count: yRow.count)
            vDSP_vdiv(tRow.map { 1 - $0 }, 1, oneMinusYPlusEps, 1, &temp, 1, vDSP_Length(yRow.count))
            vDSP_vsub(result, 1, temp, 1, &result, 1, vDSP_Length(yRow.count))
            vDSP_vneg(result, 1, &result, 1, vDSP_Length(yRow.count))
            return result
        }
    }
}

class CategoricalCrossEntropy: LossFunction {
    var ignoreIndex: Int?
    
    init(ignoreIndex: Int? = nil) {
        self.ignoreIndex = ignoreIndex
    }
    
    func loss(y: [[Float]], t: [[Float]]) -> [[Float]] {
        return zip(y, t).map { (yRow, tRow) in
            var result = [Float](repeating: 0.0, count: yRow.count)
            for i in 0..<yRow.count {
                if let ignore = ignoreIndex, tRow[i] == Float(ignore) {
                    result[i] = 0.0
                } else {
                    result[i] = -tRow[i] * log(yRow[i])
                }
            }
            return result
        }
    }
    
    func derivative(y: [[Float]], t: [[Float]]) -> [[Float]] {
        return zip(y, t).map { (yRow, tRow) in
            var result = [Float](repeating: 0.0, count: yRow.count)
            for i in 0..<yRow.count {
                if let ignore = ignoreIndex, tRow[i] == Float(ignore) {
                    result[i] = 0.0
                } else {
                    result[i] = -tRow[i] / yRow[i]
                }
            }
            return result
        }
    }
}

class CrossEntropy: LossFunction {
    var ignoreIndex: Int?
    let logSoftmax = LogSoftmax()
    
    init(ignoreIndex: Int? = nil) {
        self.ignoreIndex = ignoreIndex
    }
    
    func loss(y: [[Float]], t: [[Float]]) -> [[Float]] {
        let logSoft = logSoftmax.forward(x: y)
        var result = [[Float]](repeating: [Float](repeating: 0.0, count: y[0].count), count: y.count)
        for i in 0..<t.count {
            for j in 0..<t[i].count {
                let index = Int(t[i][j])
                if index != ignoreIndex {
                    result[i][j] = -logSoft[i][index]
                }
            }
        }
        return result
    }
    
    func derivative(y: [[Float]], t: [[Float]]) -> [[Float]] {
        let batchSize = y.count
        let err: Float = 1 / Float(batchSize)
        let nllLossDer = [[Float]](repeating: [Float](repeating: -err, count: y[0].count), count: y.count)
        
        let outputErr = logSoftmax.backward(grad: nllLossDer)
        var result = [[Float]](repeating: [Float](repeating: 0.0, count: y[0].count), count: y.count)
        for i in 0..<t.count {
            for j in 0..<t[i].count {
                let index = Int(t[i][j])
                if t[i][j] == Float(ignoreIndex ?? -1) {
                    result[i][j] = 0
                } else {
                    result[i][j] = outputErr[i][j]
                }
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
