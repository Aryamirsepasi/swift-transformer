import Foundation
import Accelerate
import MLX
import MLXRandom

class DenseGlobalVars {
    static let shared = DenseGlobalVars() // Singleton instance

    // State variables
    var outputShape: (Int, Int) = (0, 0)
    var w: MLXArray = []
    var b: MLXArray = []
    var optimizer: Optimizer?
    var v: MLXArray = []
    var m: MLXArray = []
    var vHat: MLXArray = []
    var mHat: MLXArray = []
    var vb: MLXArray = []
    var mb: MLXArray = []
    var vbHat: MLXArray = []
    var mbHat: MLXArray = []
    var XHatT: MLXArray = []
    var gradW: MLXArray = []
    var gradB: MLXArray = []
    var inputData: MLXArray = []
    var outputData: MLXArray = []
    var batchSize: Int = 0

    private init() {} // Private initializer to prevent creating multiple instances
}
class Dense {
    var unitsNum: Int
    var inputsNum: Int
    var useBias: Bool
    //var outputShape: (Int, Int)
    //var w: MLXArray
    //var b: MLXArray
    //var optimizer: Optimizer?
    var dataType: DType
    
    //var inputData, outputData: MLXArray
    //var batchSize: Int
    
    //var v, m, vHat, mHat: MLXArray
    //var vb, mb, vbHat, mbHat: MLXArray
    //var gradW: MLXArray
    //var gradB: MLXArray
    
    init(unitsNum: Int, inputsNum: Int = 0, useBias: Bool = true, dataType: DType = DType.float32) {
        self.unitsNum = unitsNum
        self.inputsNum = inputsNum
        self.useBias = useBias
        self.dataType = dataType
        /*self.w = []
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
        self.outputShape = (0, 0)*/
        
        // Ensure inputsNum is set correctly before calling build()
        if self.inputsNum > 0 {
            self.build()
        }
    }
    
    func setOptimizer(optimizer: Optimizer) {
        DenseGlobalVars.shared.optimizer = optimizer
    }
    
    func build() {
        // Check if inputsNum is valid
        guard inputsNum > 0 else {
            fatalError("inputsNum must be greater than 0 before calling build()")
        }
        
        let stdv = 1 / sqrt(Float(inputsNum))
        DenseGlobalVars.shared.w = MLXRandom.uniform(low: -stdv, high: stdv, [self.inputsNum, self.unitsNum], dtype: self.dataType)
        DenseGlobalVars.shared.b = MLX.zeros([unitsNum]).asType(dataType)
        
        DenseGlobalVars.shared.v = MLX.zeros(like: DenseGlobalVars.shared.w).asType(dataType)
        DenseGlobalVars.shared.m = MLX.zeros(like: DenseGlobalVars.shared.w).asType(dataType)
        DenseGlobalVars.shared.vHat = MLX.zeros(like: DenseGlobalVars.shared.w).asType(dataType)
        DenseGlobalVars.shared.mHat = MLX.zeros(like: DenseGlobalVars.shared.w).asType(dataType)
        
        DenseGlobalVars.shared.vb = MLX.zeros(like: DenseGlobalVars.shared.b).asType(dataType)
        DenseGlobalVars.shared.mb = MLX.zeros(like: DenseGlobalVars.shared.b).asType(dataType)
        DenseGlobalVars.shared.vbHat = MLX.zeros(like: DenseGlobalVars.shared.b).asType(dataType)
        DenseGlobalVars.shared.mbHat = MLX.zeros(like: DenseGlobalVars.shared.b).asType(dataType)
        
        DenseGlobalVars.shared.outputShape = (1, self.unitsNum)
    }
    
    func forward(X: MLXArray, training: Bool = true) -> MLXArray {
        print("entered dense forward")

        DenseGlobalVars.shared.inputData = X
        
        //print(self.w)
        //print(self.b)
        //print("inputData shape: ", self.inputData.shape)
        //print("w shape: ", self.w.shape)
        //print("b shape: ", self.b.shape)
        
        DenseGlobalVars.shared.batchSize = DenseGlobalVars.shared.inputData.count
        
        //print(self.batchSize)
        // Ensure outputData has the correct size before using it
        DenseGlobalVars.shared.outputData = MLX.tensordot(DenseGlobalVars.shared.inputData, DenseGlobalVars.shared.w) + DenseGlobalVars.shared.b
        
        //print(self.outputData)

        print("exited dense forward")

        return DenseGlobalVars.shared.outputData
    }
    
    func backward(error: MLXArray) -> MLXArray {
        print("entered dense backward")
        print("DENSE ERROR: ", error.shape)

        // Compute gradients for weights and biases
        DenseGlobalVars.shared.gradW = MLX.sum(MLX.matmul(DenseGlobalVars.shared.inputData.transposed(0, 2, 1), error), axes: [0])
        DenseGlobalVars.shared.gradB = MLX.sum(error, axes: [0, 1])
        
        // Perform matrix multiplication equivalent to np.dot(error, self.w.T)
        // Transpose w to match the required shape for multiplication
        let wT = DenseGlobalVars.shared.w.transposed(1, 0) // wT shape: [7802, 256]
        
        // Perform matrix multiplication
        let outputError = MLX.matmul(error, wT)

        print("exited dense backward")
        print("DENSE BACKWARD OUTPUT: ", outputError.shape)

        return outputError
    }

    
    func updateWeights(layerNum: Int) -> Int {
        if let optimizer = DenseGlobalVars.shared.optimizer {
            var templayerNum = layerNum
            (DenseGlobalVars.shared.w, DenseGlobalVars.shared.v, DenseGlobalVars.shared.m, DenseGlobalVars.shared.vHat, DenseGlobalVars.shared.mHat, templayerNum) = optimizer.update(gradient: DenseGlobalVars.shared.gradW, weights: &DenseGlobalVars.shared.w, v: &DenseGlobalVars.shared.v, m: &DenseGlobalVars.shared.m, vHat: &DenseGlobalVars.shared.vHat, mHat: &DenseGlobalVars.shared.mHat, t: layerNum)
            if useBias {
                (DenseGlobalVars.shared.b, DenseGlobalVars.shared.vb, DenseGlobalVars.shared.mb, DenseGlobalVars.shared.vbHat, DenseGlobalVars.shared.mbHat, templayerNum) = optimizer.update(gradient: DenseGlobalVars.shared.gradB, weights: &DenseGlobalVars.shared.b, v: &DenseGlobalVars.shared.vb, m: &DenseGlobalVars.shared.mb, vHat: &DenseGlobalVars.shared.vbHat, mHat: &DenseGlobalVars.shared.mbHat, t: layerNum)
            }
        }
        return layerNum + 1
    }
    
    func getGrads() -> (MLXArray, MLXArray) {
        return (DenseGlobalVars.shared.gradW, DenseGlobalVars.shared.gradB)
    }
    
    func setGrads(grads: (MLXArray, MLXArray)) {
        (DenseGlobalVars.shared.gradW, DenseGlobalVars.shared.gradB) = grads
    }
}
