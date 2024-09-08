import Foundation
import Accelerate
import MLX

class LayerNormGlobalVars {
    static let shared = LayerNormGlobalVars() // Singleton instance

    // State variables
    var featureSize: Int = 0
    var stddevInv: MLXArray = []
    var XCentered: MLXArray = []
    var XHat: MLXArray = []
    var gamma: MLXArray = []
    var beta: MLXArray = []
    var gradGamma: MLXArray = []
    var gradBeta: MLXArray = []
    var normalizedAxis: [Int] = []
    var mean: MLXArray = []
    var variance: MLXArray = []
    var optimizer: Optimizer?
    var inputData: MLXArray = []
    var outputData: MLXArray = []
    var vg: MLXArray = []
    var mg: MLXArray = []
    var vgHat: MLXArray = []
    var mgHat: MLXArray = []
    var vb: MLXArray = []
    var mb: MLXArray = []
    var vbHat: MLXArray = []
    var mbHat: MLXArray = []
    var XHatT: MLXArray = []

    private init() {} // Private initializer to prevent creating multiple instances
}

//needed
class LayerNormalization {
    var normalizedShape: [Int]
    var epsilon: Float
    //var gamma: MLXArray
    //var beta: MLXArray
    //var mean: MLXArray
    //var variance: MLXArray
    //var optimizer: Optimizer?
    var dataType: DType
    //var featureSize: Int
    //var inputData: MLXArray
    //var outputData: MLXArray
    //var gradGamma: MLXArray
    //var gradBeta: MLXArray
    //var vg, mg, vgHat, mgHat: MLXArray
    //var vb, mb, vbHat, mbHat: MLXArray
    //var stddevInv: MLXArray
    //var XCentered: MLXArray
    //var XHatT: MLXArray
    //var XHat: MLXArray
    //var normalizedAxis: [Int]

    init(normalizedShape: [Int] = [], epsilon: Float = 0.001, dataType: DType = DType.float32) {
        self.normalizedShape = normalizedShape
        self.epsilon = epsilon
        self.dataType = dataType
        //self.gamma = []
        //self.beta = []
        LayerNormGlobalVars.shared.vg = []
        LayerNormGlobalVars.shared.mg = []
        LayerNormGlobalVars.shared.vgHat = []
        LayerNormGlobalVars.shared.mgHat = []
        LayerNormGlobalVars.shared.vb = []
        LayerNormGlobalVars.shared.mb = []
        LayerNormGlobalVars.shared.vbHat = []
        LayerNormGlobalVars.shared.mbHat = []
        //self.XHat = []
        //self.XCentered = []
        //self.normalizedAxis = []
        //self.stddevInv = []
        //self.gradBeta = []
        //self.gradGamma = []
        LayerNormGlobalVars.shared.outputData = []
        LayerNormGlobalVars.shared.inputData = []
        LayerNormGlobalVars.shared.variance = []
        LayerNormGlobalVars.shared.mean = []
        //self.featureSize = 0
        LayerNormGlobalVars.shared.XHatT = []
                
        build()
    }

    func setOptimizer(optimizer: Optimizer) {
        LayerNormGlobalVars.shared.optimizer = optimizer
    }

    func build() {
                
        if (self.normalizedShape != []){
            
            LayerNormGlobalVars.shared.gamma = MLX.ones(self.normalizedShape).asType(self.dataType)
            LayerNormGlobalVars.shared.beta = MLX.zeros(self.normalizedShape).asType(self.dataType)
            
            LayerNormGlobalVars.shared.vg = MLX.zeros(like: LayerNormGlobalVars.shared.gamma).asType(self.dataType)
            LayerNormGlobalVars.shared.mg = MLX.zeros(like: LayerNormGlobalVars.shared.gamma).asType(self.dataType)
            
            LayerNormGlobalVars.shared.vgHat = MLX.zeros(like: LayerNormGlobalVars.shared.gamma).asType(self.dataType)
            LayerNormGlobalVars.shared.mgHat = MLX.zeros(like: LayerNormGlobalVars.shared.gamma).asType(self.dataType)
            
            LayerNormGlobalVars.shared.vb = MLX.zeros(like: LayerNormGlobalVars.shared.gamma).asType(self.dataType)
            LayerNormGlobalVars.shared.mb = MLX.zeros(like: LayerNormGlobalVars.shared.gamma).asType(self.dataType)
            
            LayerNormGlobalVars.shared.vbHat = MLX.zeros(like: LayerNormGlobalVars.shared.gamma).asType(self.dataType)
            LayerNormGlobalVars.shared.mbHat = MLX.zeros(like: LayerNormGlobalVars.shared.gamma).asType(self.dataType)
            
            
        }
    }

    func forward(X: MLXArray) -> MLXArray {
        
        print ("entered layer_norm forward")

        LayerNormGlobalVars.shared.inputData = X
        var x_T =  LayerNormGlobalVars.shared.inputData.T
        
        if (self.normalizedShape == []){
            var temp : [Int] = []
            for i in 1..<LayerNormGlobalVars.shared.inputData.shape.count{
                temp.append(LayerNormGlobalVars.shared.inputData.shape[i])
            }
            self.normalizedShape = temp
            
            self.build()
        }
        
        LayerNormGlobalVars.shared.normalizedAxis = Array(0..<( LayerNormGlobalVars.shared.inputData.ndim - LayerNormGlobalVars.shared.gamma.ndim))
        
        LayerNormGlobalVars.shared.featureSize = LayerNormGlobalVars.shared.gamma.size
        //print("gamma size: ", self.gamma.size)
        //print("featureSize after gamma size: ", self.featureSize)

        LayerNormGlobalVars.shared.mean = MLX.mean(x_T, axes: [0])
        LayerNormGlobalVars.shared.variance = MLX.variance(x_T, axes: [0])
        
        LayerNormGlobalVars.shared.XCentered = (x_T -  LayerNormGlobalVars.shared.mean)
        LayerNormGlobalVars.shared.stddevInv = 1 / MLX.sqrt( LayerNormGlobalVars.shared.variance +  self.epsilon)
        
        LayerNormGlobalVars.shared.XHatT = LayerNormGlobalVars.shared.XCentered * LayerNormGlobalVars.shared.stddevInv
        LayerNormGlobalVars.shared.XHat =  LayerNormGlobalVars.shared.XHatT.T
        
        LayerNormGlobalVars.shared.outputData = LayerNormGlobalVars.shared.gamma * LayerNormGlobalVars.shared.XHat + LayerNormGlobalVars.shared.beta
        
        print ("exited layer_norm forward")

        return  LayerNormGlobalVars.shared.outputData
    }

    func backward(error: MLXArray) -> MLXArray {
        
        print("entered layer_norm backward")

        var errorT = error.T
        
        //print("featureSize: ", self.featureSize)
        
        var temp1 = (1 / LayerNormGlobalVars.shared.featureSize) * MLX.expandedDimensions(LayerNormGlobalVars.shared.gamma, axes: LayerNormGlobalVars.shared.normalizedAxis).T
        
        print("first part passed")
        var temp2 = temp1 * LayerNormGlobalVars.shared.stddevInv
        
        print("second part passed")
        var temp3 = LayerNormGlobalVars.shared.featureSize * errorT - MLX.sum(errorT, axis: 0)
        print("third part passed")
        
        print("errorT: ", errorT.shape)
        print("temp3: ", temp3.shape)
        print("XCentered: ", LayerNormGlobalVars.shared.XCentered.shape)
        print("stddevInv power: ", MLX.pow(LayerNormGlobalVars.shared.stddevInv, 2).shape)

        var temp4 = temp3 - (LayerNormGlobalVars.shared.XCentered * MLX.pow(LayerNormGlobalVars.shared.stddevInv, 2))
        print("fourth part passed")

        var temp5 = temp4 * MLX.sum(errorT * LayerNormGlobalVars.shared.XCentered, axis: 0)
        
        print("fifth part passed")
        
        var temp6 = temp2 * temp5


        var outputError = temp6
        
        outputError = outputError.T
        
        LayerNormGlobalVars.shared.gradGamma = MLX.sum(error * LayerNormGlobalVars.shared.XHat, axes: LayerNormGlobalVars.shared.normalizedAxis)
        LayerNormGlobalVars.shared.gradBeta = MLX.sum(error, axes: LayerNormGlobalVars.shared.normalizedAxis)

        print("exited layer_norm backward")

        return outputError
    }

    func updateWeights(layerNum: Int) -> Int {
        if let optimizer =  LayerNormGlobalVars.shared.optimizer {
            var layerNum = layerNum
            (LayerNormGlobalVars.shared.gamma,  LayerNormGlobalVars.shared.vg,  LayerNormGlobalVars.shared.mg,  LayerNormGlobalVars.shared.vgHat,  LayerNormGlobalVars.shared.mgHat, layerNum) = optimizer.update(gradient: LayerNormGlobalVars.shared.gradGamma, weights: &LayerNormGlobalVars.shared.gamma, v: &LayerNormGlobalVars.shared.vg, m: &LayerNormGlobalVars.shared.mg, vHat: &LayerNormGlobalVars.shared.vgHat, mHat: &LayerNormGlobalVars.shared.mgHat, t: layerNum)
            (LayerNormGlobalVars.shared.beta,  LayerNormGlobalVars.shared.vb,  LayerNormGlobalVars.shared.mb,  LayerNormGlobalVars.shared.vbHat,  LayerNormGlobalVars.shared.mbHat, layerNum) = optimizer.update(gradient: LayerNormGlobalVars.shared.gradBeta, weights: &LayerNormGlobalVars.shared.beta, v: &LayerNormGlobalVars.shared.vb, m: &LayerNormGlobalVars.shared.mb, vHat: &LayerNormGlobalVars.shared.vbHat, mHat: &LayerNormGlobalVars.shared.mbHat, t: layerNum)
        }

        return layerNum
    }
    

    func getGrads() -> (MLXArray, MLXArray) {
        return (LayerNormGlobalVars.shared.gradGamma, LayerNormGlobalVars.shared.gradBeta)
    }

    func setGrads(grads: (MLXArray, MLXArray)) {
        LayerNormGlobalVars.shared.gradGamma = grads.0
        LayerNormGlobalVars.shared.gradBeta = grads.1
    }
}
