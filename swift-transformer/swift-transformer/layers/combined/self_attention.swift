import Foundation
import Accelerate
import MLX

//needed
class MultiHeadAttention {
    var dModel: Int
    var headsNum: Int
    var dataType: DType
    
    var dK: Int
    var dQ: Int
    var dV: Int
    var scale: Float
    
    var KLinear: Dense
    var QLinear: Dense
    var VLinear: Dense
    var OLinear: Dense
    
    var activation: Softmax
    var dropout: Dropout
    
    var K: MLXArray
    var Q: MLXArray
    var V: MLXArray
    var dropoutAttention: MLXArray
    var mask: MLXArray
    
    var keyLen, queryLen, valueLen: Int
    
    init(dModel: Int = 512, headsNum: Int = 8, dropoutRate: Float = 0.1, dataType: DType) {
        self.dModel = dModel
        self.headsNum = headsNum
        self.dataType = dataType
        
        self.dK = self.dModel / headsNum
        self.dQ = self.dModel / headsNum
        self.dV = self.dModel / headsNum
        
        /*print("dModel: ", self.dModel)
        print("dK: ", self.dK)
        print("headsNum: ", self.headsNum)
        print("unitsNum: ", self.dK * self.headsNum)
        */
        self.scale = sqrt(Float(dK))
        
        self.KLinear = Dense(unitsNum: dK * headsNum, inputsNum: dModel, useBias: false, dataType: dataType)
        self.QLinear = Dense(unitsNum: dQ * headsNum, inputsNum: dModel, useBias: false, dataType: dataType)
        self.VLinear = Dense(unitsNum: dV * headsNum, inputsNum: dModel, useBias: false, dataType: dataType)
        self.OLinear = Dense(unitsNum: dV * headsNum, inputsNum: dModel, useBias: true, dataType: dataType)
        
        self.activation = Softmax()
        self.dropout = Dropout(rate: dropoutRate, dataType: dataType)
        
        self.K = []
        self.Q = []
        self.V = []
        
        self.dropoutAttention = []
        
        self.mask = []
        
        self.keyLen = 0
        self.queryLen = 0
        self.valueLen = 0
    }
    
    func splitHeadsForward(x:MLXArray) -> MLXArray {
        //print ("entered self_attention splitHeadsForward")

        let batchSize = x.shape[0]
        
        return x.reshaped([batchSize, -1, self.headsNum, self.dK]).transposed(0,2,1,3)
    }
    
    func splitHeadsBackward(x: MLXArray) -> MLXArray {
        //print ("entered self_attention splitHeadsBackward")

        let batchSize = x.shape[0]
        
        return x.transposed(0,2,1,3).reshaped([batchSize, -1, self.headsNum * self.dK])
    }
    
    func groupHeadsForward(x: MLXArray) -> MLXArray {
        //print ("entered self_attention groupHeadsForward")

        let batchSize = x.shape[0]
        
        return x.transposed(0,2,1,3).reshaped([batchSize, -1, self.headsNum * self.dK])
    }
    
    func groupHeadsBackward(x: MLXArray) -> MLXArray {
        //print ("entered self_attention groupHeadsBackward")

        let batchSize = x.shape[0]
        
        return x.reshaped([batchSize, -1, self.headsNum, self.dK]).transposed(0,2,1,3)
    }
    
    func forward(query: MLXArray, key: MLXArray, value: MLXArray, mask: MLXArray, training: Bool = true) -> (MLXArray, MLXArray) {
        
        //print ("entered self_attention forward")

        self.keyLen = key.shape[1]
        self.queryLen = query.shape[1]
        self.valueLen = value.shape[1]
        
        //print(key.shape)
        //print(query.shape)
        //print(value.shape)

        let K = KLinear.forward(X: key, training: training)
        let Q = QLinear.forward(X: query, training: training)
        let V = VLinear.forward(X: value, training: training)
        
        //print("K shape: ", K.shape)
        
        self.K = splitHeadsForward(x: K)
        self.Q = splitHeadsForward(x: Q)
        self.V = splitHeadsForward(x: V)
        
        //print("got here?")
        //print("Q: ", self.Q.shape)
        //print("Q: ", self.K.transposed(0,1,3,2).shape)
        //print("scale: ", self.scale)

        var energy = MLX.matmul(self.Q, self.K.transposed(0,1,3,2)) / self.scale
        
        // Assign the mask
        self.mask = mask
        
        // Corrected: Apply the new axis and ellipsis correctly
        self.mask = self.mask[0..., .newAxis, .ellipsis]
        
        // Handle negative infinity
        let negativeInfinity = Double.leastNormalMagnitude * -1
        
        // Apply the mask
        energy = MLX.which(self.mask .== 0, negativeInfinity, energy)
        
        var attention = self.activation.forward(x: energy)
        
        self.dropoutAttention = self.dropout.forward(X: attention, training: training)
        var output = MLX.matmul(self.dropoutAttention, self.V)
        
        var concat_output = self.groupHeadsForward(x: output)
        
        var O = self.OLinear.forward(X: concat_output)
        
        //print ("exited self_attention forward")

        return (O, attention)
    }

    
    func backward(error: MLXArray) -> (MLXArray,MLXArray,MLXArray) {
        
        //print("entered self_attention backward")

        var error = self.OLinear.backward(error: error)
        
        error = self.groupHeadsBackward(x: error)
        
        var VError = MLX.matmul(self.dropoutAttention.transposed(0,1,3,2), error)
                
        error = MLX.matmul(error, self.V.transposed(0,1,3,2))
        error = self.dropout.backward(error)
        error = self.activation.backward(grad: error)
        
        // if self.mask is not None:
        error = MLX.which(self.mask .== 0, 0, error)
        
        error /= self.scale
        
        var QError = MLX.matmul(error, self.K)
        var KError = MLX.matmul(self.Q.transposed(0,1,3,2), error)
        KError = KError.transposed(0,1,3,2)
        
        VError = self.splitHeadsBackward(x: VError)
        QError = self.splitHeadsBackward(x: QError)
        KError = self.splitHeadsBackward(x: KError)
        
        VError = self.VLinear.backward(error: VError)
        QError = self.QLinear.backward(error: QError)
        KError = self.KLinear.backward(error: KError)
        
        //print("exited self_attention backward")

        return (QError, KError, VError)
    }
    
    func setOptimizer(optimizer: Optimizer) {
        KLinear.setOptimizer(optimizer: optimizer)
        QLinear.setOptimizer(optimizer: optimizer)
        VLinear.setOptimizer(optimizer: optimizer)
        OLinear.setOptimizer(optimizer: optimizer)
    }
    
    func updateWeights(layerNum: Int) -> Int {
        
        //print("entered self_attention updateWeights")

        var layerNum = KLinear.updateWeights(layerNum: layerNum)
        layerNum = QLinear.updateWeights(layerNum: layerNum)
        layerNum = VLinear.updateWeights(layerNum: layerNum)
        layerNum = OLinear.updateWeights(layerNum: layerNum)
        
        //print("exited self_attention updateWeights")

        return layerNum
    }
}
