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
        
        print ("entered layer_norm init")
        
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
        
        print ("exited layer_norm init")
        
    }
    
    func setOptimizer(optimizer: Optimizer) {
        print ("entered layer_norm setOptimizer")
    
        self.optimizer = optimizer
        
        print ("exited layer_norm setOptimizer")
        
    }
    
    func build() {
        
        print ("entered layer_norm build")
        
        
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
        
        print ("exited layer_norm build")
        
    }
    
    func forward(X: MLXArray) -> MLXArray {
        
        print ("entered layer_norm forward")
        
        self.inputData = X
        var x_T =  self.inputData.T
        
        if (self.normalizedShape == []){
            /*var temp : [Int] = []
            for i in 1..<self.inputData.shape.count{
                temp.append(self.inputData.shape[i])
            }*/
            self.normalizedShape = Array(self.inputData.shape[1...])
            
            self.build()
        }
        
        self.normalizedAxis = Array(0..<( self.inputData.ndim - self.gamma.ndim))
        
        self.featureSize = self.gamma.size
        //print("gamma size: ", self.gamma.size)
        //print("featureSize after gamma size: ", self.featureSize)
        
        self.mean = MLX.mean(x_T, axes: [0], stream: .gpu)
        self.variance = MLX.variance(x_T, axes: [0], stream: .gpu)
        
        self.XCentered = (x_T -  self.mean)
        self.stddevInv = 1 / MLX.sqrt( self.variance +  self.epsilon)
        
        self.XHatT = self.XCentered * self.stddevInv
        self.XHat =  self.XHatT.T
        
        self.outputData = self.gamma * self.XHat + self.beta
        
        print ("exited layer_norm forward")
        
        return  self.outputData
    }
    
    func backward(error: MLXArray) -> MLXArray {
        
        print ("entered layer_norm backward")
        
        let errorT = error.T
        let gammaExpanded = MLX.expandedDimensions(self.gamma, axes: self.normalizedAxis, stream: .gpu).T
        let temp1 = (1 / Float(self.featureSize)) * gammaExpanded
        let temp2 = temp1 * self.stddevInv
        
        // Correctly compute temp3
        let sumErrorT = MLX.sum(errorT, axis: 0, stream: .gpu)
        let sumErrorTXCentered = MLX.sum(errorT * self.XCentered, axis: 0, stream: .gpu)
        let temp3 = self.featureSize * errorT - sumErrorT - self.XCentered * MLX.pow(self.stddevInv, 2) * sumErrorTXCentered
        
        let outputError = temp2 * temp3
        self.gradGamma = MLX.sum(error * self.XHat, axes: self.normalizedAxis, stream: .gpu)
        self.gradBeta = MLX.sum(error, axes: self.normalizedAxis, stream: .gpu)
        
        print ("exited layer_norm backward")
        
        return outputError.T
    }
    
    
    func updateWeights(layerNum: Int) -> Int {
        
        
        print("entered layer_norm updateWeights")
        
        if let optimizer =  self.optimizer {
            var layerNum = layerNum
            (self.gamma,  self.vg,  self.mg,  self.vgHat,  self.mgHat, layerNum) = optimizer.update(gradient: self.gradGamma, weights: &self.gamma, v: &self.vg, m: &self.mg, vHat: &self.vgHat, mHat: &self.mgHat, t: layerNum)
            (self.beta,  self.vb,  self.mb,  self.vbHat,  self.mbHat, layerNum) = optimizer.update(gradient: self.gradBeta, weights: &self.beta, v: &self.vb, m: &self.mb, vHat: &self.vbHat, mHat: &self.mbHat, t: layerNum)
        }
        
        print("exited layer_norm updateWeights")
        
        
        return layerNum
    }
    
    
    func getGrads() -> (MLXArray, MLXArray) {
        
        print ("entered layer_norm getGrads")
        
        return (self.gradGamma, self.gradBeta)
    }
    
    func setGrads(grads: (MLXArray, MLXArray)) {
        
        print ("entered layer_norm setGrads")
        
        self.gradGamma = grads.0
        self.gradBeta = grads.1
        
        print ("exited layer_norm setGrads")
        
    }
}
