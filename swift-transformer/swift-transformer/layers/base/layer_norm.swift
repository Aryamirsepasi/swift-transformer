import Foundation
import Accelerate
import MLX

//needed
class LayerNormalization {
    var normalizedShape: [Int]
    var epsilon: Float
    var gamma: MLXArray
    var beta: MLXArray
    var mean: MLXArray
    var variance: MLXArray
    var optimizer: Optimizer?
    var dataType: DType
    var featureSize: Int
    var inputData: MLXArray
    var outputData: MLXArray
    var gradGamma: MLXArray
    var gradBeta: MLXArray
    var vg, mg, vgHat, mgHat: MLXArray
    var vb, mb, vbHat, mbHat: MLXArray
    var stddevInv: MLXArray
    var XCentered: MLXArray
    var XHatT: MLXArray
    var XHat: MLXArray
    var normalizedAxis: [Int]

    init(normalizedShape: [Int] = [], epsilon: Float = 0.001, dataType: DType = DType.float32) {
        self.normalizedShape = normalizedShape
        self.epsilon = epsilon
        self.dataType = dataType
        self.gamma = []
        self.beta = []
        self.vg = []
        self.mg = []
        self.vgHat = []
        self.mgHat = []
        self.vb = []
        self.mb = []
        self.vbHat = []
        self.mbHat = []
        self.XHat = []
        self.XCentered = []
        self.normalizedAxis = []
        self.stddevInv = []
        self.gradBeta = []
        self.gradGamma = []
        self.outputData = []
        self.inputData = []
        self.variance = []
        self.mean = []
        self.featureSize = 0
        self.XHatT = []
        
        build()
    }

    func setOptimizer(optimizer: Optimizer) {
        self.optimizer = optimizer
    }

    func build() {
                
        if (self.normalizedShape != []){
            
            self.gamma = MLX.ones(self.normalizedShape).asType(self.dataType)
            self.beta = MLX.zeros(self.normalizedShape).asType(self.dataType)
            
            self.vg = MLX.zeros(like: self.gamma).asType(self.dataType)
            self.mg = MLX.zeros(like: self.gamma).asType(self.dataType)
            
            self.vgHat = MLX.zeros(like: self.gamma).asType(self.dataType)
            self.mgHat = MLX.zeros(like: self.gamma).asType(self.dataType)
            
            self.vb = MLX.zeros(like: self.gamma).asType(self.dataType)
            self.mb = MLX.zeros(like: self.gamma).asType(self.dataType)
            
            self.vbHat = MLX.zeros(like: self.gamma).asType(self.dataType)
            self.mbHat = MLX.zeros(like: self.gamma).asType(self.dataType)
            
            
        }
    }

    func forward(X: MLXArray) -> MLXArray {
        
        print ("entered layer_norm forward")

        self.inputData = X
        var x_T = self.inputData.T
        
        if (self.normalizedShape == []){
            var temp : [Int] = []
            for i in 1..<self.inputData.shape.count{
                temp.append(self.inputData.shape[i])
            }
            self.normalizedShape = temp
            
            self.build()
        }
        
        self.normalizedAxis = Array(0..<(self.inputData.ndim - self.gamma.ndim))
        
        self.featureSize = self.gamma.size
        
        self.mean = MLX.mean(x_T, axes: [0])
        self.variance = MLX.variance(x_T, axes: [0])
        
        self.XCentered = (x_T - self.mean)
        self.stddevInv = 1 / MLX.sqrt(self.variance + self.epsilon)
        
        self.XHatT = self.XCentered * self.stddevInv
        self.XHat = self.XHatT.T
        
        self.outputData = self.gamma * self.XHat + self.beta
        
        print ("exited layer_norm forward")

        return self.outputData
    }

    func backward(error: MLXArray) -> MLXArray {
        var errorT = error.T
        
        var temp1 = (1 / self.featureSize) * MLX.expandedDimensions(self.gamma, axes: self.normalizedAxis).T
        var temp2 = temp1 * self.stddevInv * (self.featureSize * errorT - MLX.sum(errorT, axis: 0) - self.XCentered * MLX.pow(self.stddevInv, 2) * MLX.sum(errorT * self.XCentered, axis: 0))
        
        var outputError = temp2
        
        outputError = outputError.T
        
        self.gradGamma = MLX.sum(error * self.XHat, axes: self.normalizedAxis)
        self.gradBeta = MLX.sum(error, axes: self.normalizedAxis)

        return outputError
    }

    func updateWeights(layerNum: Int) -> Int {
        if let optimizer = self.optimizer {
            var layerNum = layerNum
            (gamma, vg, mg, vgHat, mgHat, layerNum) = optimizer.update(gradient: gradGamma, weights: &gamma, v: &vg, m: &mg, vHat: &vgHat, mHat: &mgHat, t: layerNum)
            (beta, vb, mb, vbHat, mbHat, layerNum) = optimizer.update(gradient: gradBeta, weights: &beta, v: &vb, m: &mb, vHat: &vbHat, mHat: &mbHat, t: layerNum)
        }

        return layerNum
    }
    

    func getGrads() -> (MLXArray, MLXArray) {
        return (self.gradGamma, self.gradBeta)
    }

    func setGrads(grads: (MLXArray, MLXArray)) {
        self.gradGamma = grads.0
        self.gradBeta = grads.1
    }
}
