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
    var gradWeights: MLXArray
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
        self.gradWeights = []
        self.batchSize = 0
        self.currentInputLength = 0
        
        self.build()
    }
    
    func setOptimizer(optimizer: Optimizer) {
        self.optimizer = optimizer
    }
    
    func build() {
        
        self.w = MLXRandom.normal([self.inputDim, self.outputDim], loc: 0, scale: pow(Float(self.inputDim), -0.5)).asType(self.dataType)

        self.v = MLX.zeros(like: self.w).asType(self.dataType)
        self.m = MLX.zeros(like: self.w).asType(self.dataType)
        
        self.vHat = MLX.zeros(like: self.w).asType(self.dataType)
        self.mHat = MLX.zeros(like: self.w).asType(self.dataType)
        
        
    }
    
    func prepareLabels(batchLabels: MLXArray) -> MLXArray {
        // Convert batch labels to integers
        var batchLabelsVar = batchLabels.asType(DType.int32)
        
        // Prepare an empty tensor for the one-hot encoding
        var prepareBatchLabels = MLX.zeros([batchLabelsVar.size, self.inputDim])
        
        // Generate range of indices using the initializer
        let indices = MLXArray(0..<batchLabelsVar.size).asType(DType.int32)
        let reshapedLabels = batchLabelsVar.reshaped([batchLabelsVar.size])

        // Perform one-hot encoding manually
        for i in 0..<indices.size {
            let index = indices[i].item(Int.self)
            let label = reshapedLabels[i].item(Int.self)
            prepareBatchLabels[index, label] = MLXArray(1)
        }
        
        // Reshape the tensor to the desired dimensions
        return prepareBatchLabels.reshaped([self.batchSize, self.currentInputLength, self.inputDim]).asType(self.dataType)
    }


    func forward(X: MLXArray) -> MLXArray {
        
        print ("entered embedding forward")

        self.inputData = X
        
        for i in 0..<self.inputData.count{
            var arr = self.inputData[i]
            
            if !(MLX.equal(self.inputData[0].count, arr.count).all().item()){
                fatalError("Input sequences must be of the same length")
            }
        }
        
        self.currentInputLength = self.inputData[0].count
        self.batchSize = self.inputData.count
        
        self.inputData = self.prepareLabels(batchLabels: self.inputData)
        
        self.outputData = MLX.matmul(self.inputData, self.w)
        
        print ("exited embedding forward")

        return self.outputData
        
    }
    
    func backward(error: MLXArray) -> MLXArray {
        
        self.gradW = MLX.matmul(MLX.transposed(self.inputData, axes: [0,2,1]), error).logSumExp(axis: 0)
        
        return []
    }

    func updateWeights(layerNum: Int) -> Int {
        if let optimizer = optimizer {
            var templayerNum = layerNum
            (w, v, m, vHat, mHat, templayerNum) = optimizer.update(gradient: gradWeights, weights: &w, v: &v, m: &m, vHat: &vHat, mHat: &mHat, t: layerNum)
        }
        return layerNum + 1
    }
    
    func getGrads() -> (MLXArray, MLXArray) {
        return (self.gradW, self.gradB)
    }

    func setGrads(grads: (MLXArray, MLXArray)) {
        (self.gradW, self.gradB) = grads
    }
}

// Helper function to generate normal distribution
struct NormalDistribution {
    var mean: Float
    var standardDeviation: Float
    
    func next() -> Float {
        let u1 = Float.random(in: 0..<1)
        let u2 = Float.random(in: 0..<1)
        let r = sqrt(-2 * log(u1))
        let theta = 2 * Float.pi * u2
        return r * sin(theta) * standardDeviation + mean
    }
}
