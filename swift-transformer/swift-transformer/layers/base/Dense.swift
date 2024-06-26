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
    var outputShape: (Int, Int)
    
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
        self.outputShape = (1, unitsNum)
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
    
    func forward(_ X: [[Float]], training: Bool = true) -> [[Float]] {
        guard let inputsNum = self.inputsNum else {
            fatalError("inputsNum is nil")
        }
        
        let rows = X.count
        let cols = X[0].count
        
        var result = [[Float]](repeating: [Float](repeating: 0, count: unitsNum), count: rows)
        
        for i in 0..<rows {
            var singleResult = [Float](repeating: 0, count: unitsNum)
            vDSP_mmul(X[i], 1, w, 1, &singleResult, 1, vDSP_Length(1), vDSP_Length(unitsNum), vDSP_Length(cols))
            result[i] = singleResult
        }
        
        if useBias {
            for i in 0..<rows {
                var biasAddedResult = [Float](repeating: 0, count: unitsNum)
                vDSP_vadd(result[i], 1, b, 1, &biasAddedResult, 1, vDSP_Length(unitsNum))
                result[i] = biasAddedResult
            }
        }
        
        return result
    }
    
    func backward(_ error: [[Float]]) -> [[Float]] {
        guard let inputsNum = self.inputsNum else {
            return []
        }
        
        let rows = error.count
        let cols = error[0].count
        
        var outputError = [[Float]](repeating: [Float](repeating: 0, count: inputsNum), count: rows)
        var transposedW = [Float](repeating: 0, count: inputsNum * unitsNum)
        
        vDSP_mtrans(w, 1, &transposedW, 1, vDSP_Length(inputsNum), vDSP_Length(unitsNum))
        
        for i in 0..<rows {
            var singleError = [Float](repeating: 0, count: inputsNum)
            vDSP_mmul(error[i], 1, transposedW, 1, &singleError, 1, vDSP_Length(1), vDSP_Length(inputsNum), vDSP_Length(unitsNum))
            outputError[i] = singleError
        }
        
        gradW = zerosLike(w)
        gradB = zerosLike(b)
        
        for i in 0..<rows {
            var tempGradW = [Float](repeating: 0, count: inputsNum * unitsNum)
            vDSP_mmul(error[i], 1, w, 1, &tempGradW, 1, vDSP_Length(unitsNum), vDSP_Length(inputsNum), 1)
            vDSP_vadd(gradW, 1, tempGradW, 1, &gradW, 1, vDSP_Length(gradW.count))
            
            vDSP_vadd(gradB, 1, error[i], 1, &gradB, 1, vDSP_Length(unitsNum))
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
