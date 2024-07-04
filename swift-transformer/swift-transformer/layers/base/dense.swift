import Foundation
import Accelerate
import Matft

class Dense {
    var unitsNum: Int
    var inputsNum: Int?
    var useBias: Bool
    var w: [Float]
    var b: [Float]
    var optimizer: Optimizer?
    var dataType: [Float]
    
    var v, m, vHat, mHat: [Float]
    var vb, mb, vbHat, mbHat: [Float]
    var gradW: [Float]
    var gradB: [Float]
    
    init(unitsNum: Int, inputsNum: Int? = nil, useBias: Bool = true, dataType: [Float]) {
        self.unitsNum = unitsNum
        self.inputsNum = inputsNum
        self.useBias = useBias
        self.dataType = dataType
        self.w = []
        self.b = []
        self.v = []
        self.m = []
        self.vHat = []
        self.mHat = []
        self.vb = []
        self.mb = []
        self.vbHat = []
        self.mbHat = []
        self.gradW = []
        self.gradB = []
        
        self.build()
    }
    
    func setOptimizer(optimizer: Optimizer) {
        self.optimizer = optimizer
    }
    
    func build() {
        guard let inputsNum = self.inputsNum else {
            return
        }
        
        let stdv = 1 / sqrt(Float(inputsNum))
        self.w = uniform(-stdv, stdv, inputsNum * unitsNum)
        self.b = zeros((1, unitsNum)).flatMap { $0 }
        
        self.v = zerosLike(w)
        self.m = zerosLike(w)
        self.vHat = zerosLike(w)
        self.mHat = zerosLike(w)
        
        self.vb = zerosLike(b)
        self.mb = zerosLike(b)
        self.vbHat = zerosLike(b)
        self.mbHat = zerosLike(b)
    }
    
    func forward(_ X: [[[Float]]], training: Bool = true) -> [[[Float]]] {
        guard let inputsNum = self.inputsNum else {
            fatalError("inputsNum is nil")
        }
        
        let batchSize = X.count
        let seqLen = X[0].count
        
        var result = [[[Float]]](repeating: [[Float]](repeating: [Float](repeating: 0, count: unitsNum), count: seqLen), count: batchSize)
        
        for b in 0..<batchSize {
            for s in 0..<seqLen {
                var singleResult = [Float](repeating: 0, count: unitsNum)
                vDSP_mmul(X[b][s], 1, w, 1, &singleResult, 1, vDSP_Length(1), vDSP_Length(unitsNum), vDSP_Length(inputsNum))
                result[b][s] = singleResult
            }
        }
        
        if useBias {
            for bIndex in 0..<batchSize {
                for s in 0..<seqLen {
                    result[bIndex][s].withUnsafeMutableBufferPointer { resultBuffer in
                        b.withUnsafeBufferPointer { biasBuffer in
                            vDSP_vadd(resultBuffer.baseAddress!, 1, biasBuffer.baseAddress!, 1, resultBuffer.baseAddress!, 1, vDSP_Length(unitsNum))
                        }
                    }
                }
            }
        }

        
        return result
    }
    
    func backward(_ error: [[[Float]]]) -> [[[Float]]] {
        guard let inputsNum = self.inputsNum else {
            return []
        }
        
        let batchSize = error.count
        let seqLen = error[0].count
        
        var outputError = [[[Float]]](repeating: [[Float]](repeating: [Float](repeating: 0, count: inputsNum), count: seqLen), count: batchSize)
        var transposedW = [Float](repeating: 0, count: inputsNum * unitsNum)
        
        vDSP_mtrans(w, 1, &transposedW, 1, vDSP_Length(inputsNum), vDSP_Length(unitsNum))
        
        for b in 0..<batchSize {
            for s in 0..<seqLen {
                var singleError = [Float](repeating: 0, count: inputsNum)
                vDSP_mmul(error[b][s], 1, transposedW, 1, &singleError, 1, vDSP_Length(1), vDSP_Length(inputsNum), vDSP_Length(unitsNum))
                outputError[b][s] = singleError
            }
        }
        
        gradW = zerosLike(w)
        gradB = zerosLike(b)
        
        for b in 0..<batchSize {
            for s in 0..<seqLen {
                var tempGradW = [Float](repeating: 0, count: inputsNum * unitsNum)
                vDSP_mmul(error[b][s], 1, w, 1, &tempGradW, 1, vDSP_Length(unitsNum), vDSP_Length(inputsNum), 1)
                vDSP_vadd(gradW, 1, tempGradW, 1, &gradW, 1, vDSP_Length(gradW.count))
                
                vDSP_vadd(gradB, 1, error[b][s], 1, &gradB, 1, vDSP_Length(unitsNum))
            }
        }
        
        return outputError
    }
    
    func updateWeights(layerNum: Int) -> Int {
        if let optimizer = self.optimizer {
            var templayerNum = layerNum
            (w, v, m, vHat, mHat, templayerNum) = optimizer.update(gradient: gradW, weights: &w, v: &v, m: &m, vHat: &vHat, mHat: &mHat, t: layerNum)
            if useBias {
                (b, vb, mb, vbHat, mbHat, templayerNum) = optimizer.update(gradient: gradB, weights: &b, v: &vb, m: &mb, vHat: &vbHat, mHat: &mbHat, t: layerNum)
            }
        }
        return layerNum + 1
    }
    
    func getGrads() -> ([Float], [Float]) {
        return (gradW, gradB)
    }
    
    func setGrads(grads: ([Float], [Float])) {
        (gradW, gradB) = grads
    }
}
