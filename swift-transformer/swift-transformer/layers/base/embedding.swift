import Foundation
import Accelerate
import MLX
import MLXRandom

class Embedding {
    var inputDim: Int
    var inputData: MLXArray
    var outputData: MLXArray
    var outputDim: Int
    var gradW: MLXArray
    var gradB: MLXArray
    var w: MLXArray
    var optimizer: Optimizer?
    var v: MLXArray
    var m: MLXArray
    var vHat: MLXArray
    var mHat: MLXArray
    var inputLabels: MLXArray
    var dataType: DType
    var batchSize: Int
    var currentInputLength: Int
    
    
    init(inputDim: Int, outputDim: Int, dataType: DType = DType.float32) {
        
        self.inputDim = inputDim
        self.outputDim = outputDim
        
        self.dataType = dataType
        self.w = []
        self.v = []
        self.m = []
        self.vHat = []
        self.mHat = []
        self.inputData = []
        self.outputData = []
        self.gradB = []
        self.gradW = []
        self.inputLabels = []
        self.batchSize = 0
        self.currentInputLength = 0
        
        self.build()
        
    }
    
    func setOptimizer(optimizer: Optimizer) {
        
        self.optimizer = optimizer
        
    }
    
    func build() {
        
        self.w = MLXRandom.normal([self.inputDim, self.outputDim], loc: 0, scale: pow(Float(self.inputDim), -0.5), stream: .gpu).asType(self.dataType)
        
        self.v = MLX.zeros(like: self.w).asType(self.dataType)
        self.m = MLX.zeros(like: self.w).asType(self.dataType)
        
        self.vHat = MLX.zeros(like: self.w).asType(self.dataType)
        self.mHat = MLX.zeros(like: self.w).asType(self.dataType)
        
    }
    
    func prepareLabels(batchLabels: MLXArray) -> MLXArray {
            return autoreleasepool {
                let batchLabelsVar = batchLabels.asType(DType.int32)
                let prepareBatchLabels = MLX.zeros([batchLabelsVar.size, self.inputDim])
                
                // Optimize one-hot encoding to avoid intermediate arrays
                prepareBatchLabels[MLXArray(0..<batchLabelsVar.size),
                                 batchLabelsVar.reshaped([1,-1])] = MLXArray(1)
                
                return prepareBatchLabels.reshaped([self.batchSize,
                                                  self.currentInputLength,
                                                  self.inputDim],
                                                 stream: .gpu)
                                        .asType(self.dataType)
            }
        }
    
    func forward(X: MLXArray) -> MLXArray {
        
        autoreleasepool {
            self.inputData = X
            
            for i in 0..<self.inputData.count{
                let arr = self.inputData[i]
                
                if !(MLX.equal(self.inputData[0].count, arr.count).all().item()){
                    fatalError("Input sequences must be of the same length")
                }
            }
            
            self.currentInputLength = self.inputData[0].count
            self.batchSize = self.inputData.count
            
            self.inputData = self.prepareLabels(batchLabels: self.inputData)
            
            self.outputData = MLX.matmul(self.inputData, self.w, stream: .gpu)
            
            
            return self.outputData
        }
        
    }
    
    func backward(error: MLXArray) -> MLXArray {
        autoreleasepool {
            
            self.gradW = MLX.matmul(MLX.transposed(self.inputData, axes: [0,2,1], stream: .gpu), error, stream: .gpu).sum(axis: 0, stream: .gpu)
            
            return []
        }
    }
    
    func updateWeights(layerNum: Int) -> Int {
        autoreleasepool {
            
            if let optimizer = optimizer {
                var templayerNum = layerNum
                (w, v, m, vHat, mHat, templayerNum) = optimizer.update(gradient: gradW, weights: w, v: v, m: m, vHat: vHat, mHat: mHat, t: layerNum)
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
