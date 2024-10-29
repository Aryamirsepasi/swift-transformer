import Foundation
import Accelerate
import MLX
import MLXRandom

class Dense {
    // Instance-specific properties
    var unitsNum: Int
    var inputsNum: Int
    var useBias: Bool
    var dataType: DType
    
    var w: MLXArray
    var b: MLXArray
    var optimizer: Optimizer?
    var v: MLXArray
    var m: MLXArray
    var vHat: MLXArray
    var mHat: MLXArray
    var vb: MLXArray
    var mb: MLXArray
    var vbHat: MLXArray
    var mbHat: MLXArray
    
    var inputData: MLXArray?
    var outputData: MLXArray
    var gradW: MLXArray
    var gradB: MLXArray
    var batchSize: Int
    
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
        self.outputData = []
        self.gradW = []
        self.gradB = []
        self.batchSize = 0
        
        self.build()
        
    }
    
    func setOptimizer(optimizer: Optimizer) {
        
        self.optimizer = optimizer
        
    }
    
    func build() {
        
        let stdv = 1 / sqrt(Float(self.inputsNum))
        self.w = MLXRandom.uniform(low: -stdv, high: stdv, [self.inputsNum, self.unitsNum], dtype: self.dataType)
        self.b = MLX.zeros([self.unitsNum]).asType(self.dataType)
        
        self.v = MLX.zeros(like: self.w).asType(self.dataType)
        self.m = MLX.zeros(like: self.w).asType(self.dataType)
        self.vHat = MLX.zeros(like: self.w).asType(self.dataType)
        self.mHat = MLX.zeros(like: self.w).asType(self.dataType)
        
        self.vb = MLX.zeros(like: self.b).asType(self.dataType)
        self.mb = MLX.zeros(like: self.b).asType(self.dataType)
        self.vbHat = MLX.zeros(like: self.b).asType(self.dataType)
        self.mbHat = MLX.zeros(like: self.b).asType(self.dataType)
        
        
    }
    
    func forward(X: MLXArray, training: Bool = true) -> MLXArray {
        return autoreleasepool {
            
            self.inputData = X // Set inputData
            
            // Perform batched matrix multiplication
            let weightedSum = MLX.matmul(self.inputData!, self.w)
            
            // Add the bias term, broadcasting over the correct dimension
            self.outputData = MLX.add(weightedSum,self.b)
            
            return self.outputData
        }
    }
    
    func backward(error: MLXArray) -> MLXArray {
        return autoreleasepool {
            
            guard let inputData = self.inputData else {
                fatalError("Input data not set. Ensure forward pass is called before backward pass.")
            }
            
            // Compute gradients for weights and biases
            self.gradW = MLX.sum(MLX.matmul(inputData.transposed(0, 2, 1, stream: .gpu), error, stream: .gpu), axes: [0], stream: .gpu)
            self.gradB = MLX.sum(error, axes: [0, 1], stream: .gpu)
            
            // Perform matrix multiplication equivalent to np.dot(error, self.w.T)
            let outputError = MLX.matmul(error, self.w.T, stream: .gpu)
            return outputError
        }
    }
    
    func updateWeights(layerNum: Int) -> Int {
        autoreleasepool {
            
            if let optimizer = self.optimizer {
                var templayerNum = layerNum
                (self.w, self.v, self.m, self.vHat, self.mHat, templayerNum) = optimizer.update(gradient: self.gradW, weights: self.w, v: self.v, m: self.m, vHat: self.vHat, mHat: self.mHat, t: layerNum)
                if self.useBias {
                    (self.b, self.vb, self.mb, self.vbHat, self.mbHat, templayerNum) = optimizer.update(gradient: self.gradB, weights: self.b, v: self.vb, m: self.mb, vHat: self.vbHat, mHat: self.mbHat, t: layerNum)
                }
            }
            
            return layerNum + 1
        }
    }
    
    func getGrads() -> (MLXArray, MLXArray) {
        
        return (self.gradW, self.gradB)
    }
    
    func setGrads(grads: (MLXArray, MLXArray)) {
        
        (self.gradW, self.gradB) = grads
        
    }
}
