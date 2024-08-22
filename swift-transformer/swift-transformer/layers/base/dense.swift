import Foundation
import Accelerate
import MLX
import MLXRandom

 //needed
class Dense {
    var unitsNum: Int
    var inputsNum: Int
    var useBias: Bool
    var outputShape: (Int,Int)
    var w: MLXArray
    var b: MLXArray
    var optimizer: Optimizer?
    var dataType: DType
    
    var inputData, outputData: MLXArray
    var batchSize: Int
    
    var v, m, vHat, mHat: MLXArray
    var vb, mb, vbHat, mbHat: MLXArray
    var gradW: MLXArray
    var gradB: MLXArray
    
    init(unitsNum: Int, inputsNum: Int = 0, useBias: Bool = true, dataType: DType = DType.float32) {
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
        
        self.inputData = []
        self.outputData = []
        self.batchSize = 0
        self.outputShape = (0,0)
        
        self.build()
    }
    
    func setOptimizer(optimizer: Optimizer) {
        self.optimizer = optimizer
    }
    
    func build() {
        
        var stdv = 1 / sqrt(Float(inputsNum))
        // need replacement for uniform
        /*func randomUniform(low: Float, high: Float, count: Int) -> [Float] {
                return (0..<count).map { _ in
                    Float.random(in: low...high)
                }
        }
        let wArray = randomUniform(low: -stdv, high: stdv, count: inputsNum * unitsNum)*/
        self.w = MLXRandom.uniform(low: -stdv, high: stdv,[self.inputsNum, self.unitsNum], dtype: self.dataType)
        self.b = MLX.zeros([unitsNum]).asType(dataType)
        
        self.v = MLX.zeros(like:w).asType(dataType)
        self.m = MLX.zeros(like:w).asType(dataType)
        self.vHat = MLX.zeros(like:w).asType(dataType)
        self.mHat = MLX.zeros(like:w).asType(dataType)
        
        self.vb = MLX.zeros(like:b).asType(dataType)
        self.mb = MLX.zeros(like:b).asType(dataType)
        self.vbHat = MLX.zeros(like:b).asType(dataType)
        self.mbHat = MLX.zeros(like:b).asType(dataType)
        
        self.outputShape = (1, self.unitsNum)
    }
    
    func forward(X: MLXArray , training: Bool = true) -> MLXArray {
        self.inputData = X
        
        self.batchSize = self.inputData.count
        
        self.outputData = MLX.zeros([batchSize, self.b.count])
        
        for i in 0..<self.batchSize {
            for j in 0..<self.b.count {
                for k in 0..<self.w.count {
                    self.outputData[i,j] = MLX.sum(self.inputData[i, k] * self.w[k, j])
                }
            }
        }
        
        self.outputData += self.b

        return self.outputData
    }
    
    func backward(_ error: MLXArray) -> MLXArray {
        self.gradW = MLX.sum(MLX.matmul(self.inputData.transposed(0, 2, 1), error), axes: [0])
        
        self.gradB = MLX.sum(error, axes: [0,1])
        
        var outputError = MLX.zeros([error.shape[0], error.shape[1], self.w.shape[0]])

        for i in 0..<error.shape[0] {
            for j in 0..<error.shape[1] {
                for k in 0..<self.w.shape[0] {
                    for l in 0..<self.w.shape[1] {
                        outputError[i,j,k] = MLX.sum(error[i, j, l] * self.w[k, l])
                    }
                }
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
    
    func getGrads() -> (MLXArray, MLXArray) {
        return (gradW, gradB)
    }
    
    func setGrads(grads: (MLXArray, MLXArray)) {
        (gradW, gradB) = grads
    }
}
