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
    var mask: MLXArray?
    
    var keyLen, queryLen, valueLen: Int
    
    init(dModel: Int = 512, headsNum: Int = 8, dropoutRate: Float = 0.1, dataType: DType) {
        
        self.dModel = dModel
        self.headsNum = headsNum
        self.dataType = dataType
        
        self.dK = self.dModel / headsNum
        self.dQ = self.dModel / headsNum
        self.dV = self.dModel / headsNum
        
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
        
        let batchSize = x.shape[0]
        
        return x.reshaped([batchSize, -1, self.headsNum, self.dK], stream: .gpu).transposed(0,2,1,3, stream: .gpu)
    }
    
    func splitHeadsBackward(x: MLXArray) -> MLXArray {
        
        let batchSize = x.shape[0]
        
        return x.transposed(0,2,1,3, stream: .gpu).reshaped([batchSize, -1, self.headsNum * self.dK], stream: .gpu)
    }
    
    func groupHeadsForward(x: MLXArray) -> MLXArray {
        
        let batchSize = x.shape[0]
        
        return x.transposed(0,2,1,3, stream: .gpu).reshaped([batchSize, -1, self.headsNum * self.dK], stream: .gpu)
    }
    
    func groupHeadsBackward(x: MLXArray) -> MLXArray {
        
        let batchSize = x.shape[0]
        
        return x.reshaped([batchSize, -1, self.headsNum, self.dK], stream: .gpu).transposed(0,2,1,3, stream: .gpu)
    }
    
    func forward(query: MLXArray, key: MLXArray, value: MLXArray, mask: MLXArray, training: Bool = true) -> (MLXArray, MLXArray) {
        autoreleasepool {
            
            self.keyLen = key.shape[1]
            self.queryLen = query.shape[1]
            self.valueLen = value.shape[1]
            
            let K = KLinear.forward(X: key)
            let Q = QLinear.forward(X: query)
            let V = VLinear.forward(X: value)
            
            self.K = splitHeadsForward(x: K)
            self.Q = splitHeadsForward(x: Q)
            self.V = splitHeadsForward(x: V)
            
            // Compute attention scores: Q @ K^T / sqrt(d_k)
            var energy = MLX.matmul(self.Q, self.K.transposed(0,1,3,2), stream: .gpu) / self.scale
            
            // Store the mask and apply it
            // Mask should be broadcastable to [batch, heads, query_len, key_len]
            if mask.size > 0 {
                // Expand mask to have heads dimension if needed
                // Input mask is typically [batch, 1, seq_len] or [batch, seq_len, seq_len]
                var expandedMask = mask
                if mask.ndim == 3 {
                    // Add heads dimension: [batch, 1, q_len, k_len] or [batch, 1, 1, k_len]
                    expandedMask = mask.expandedDimensions(axis: 1)
                }
                self.mask = expandedMask
                
                // Apply mask: use -inf for masked positions (where mask is 0/false)
                // MLX.where selects from second arg when condition is true, third when false
                let negInf = MLXArray(Float(-1e9))  // Use large negative value instead of -inf for numerical stability
                energy = MLX.where(expandedMask .== 0, negInf, energy, stream: .gpu)
            } else {
                self.mask = nil
            }
            
            let attention = self.activation.forward(x: energy)
            
            self.dropoutAttention = self.dropout.forward(X: attention, training: training)
            let output = MLX.matmul(self.dropoutAttention, self.V, stream: .gpu)
            
            let concat_output = self.groupHeadsForward(x: output)
            
            let O = self.OLinear.forward(X: concat_output)
            
            return (O, attention)
        }
    }
    
    
    func backward(error: MLXArray) -> (MLXArray,MLXArray,MLXArray) {
        autoreleasepool {
            
            var error = self.OLinear.backward(error: error)
            
            error = self.groupHeadsBackward(x: error)
            
            var VError = MLX.matmul(self.dropoutAttention.transposed(0,1,3,2), error)
            
            error = MLX.matmul(error, self.V.transposed(0,1,3,2), stream: .gpu)
            error = self.dropout.backward(error)
            error = self.activation.backward(grad: error)
            
            // Apply mask to gradients (zero out gradients for masked positions)
            if let mask = self.mask {
                error = MLX.where(mask .== 0, MLXArray(Float(0)), error, stream: .gpu)
            }
            error = error / self.scale
            
            var QError = MLX.matmul(error, self.K, stream: .gpu)
            var KError = MLX.matmul(self.Q.transposed(0,1,3,2), error, stream: .gpu)
            KError = KError.transposed(0,1,3,2, stream: .gpu)
            
            VError = self.splitHeadsBackward(x: VError)
            QError = self.splitHeadsBackward(x: QError)
            KError = self.splitHeadsBackward(x: KError)
            
            VError = self.VLinear.backward(error: VError)
            QError = self.QLinear.backward(error: QError)
            KError = self.KLinear.backward(error: KError)
            
            return (QError, KError, VError)
        }
    }
    
    func setOptimizer(optimizer: Optimizer) {
        
        KLinear.setOptimizer(optimizer: optimizer)
        QLinear.setOptimizer(optimizer: optimizer)
        VLinear.setOptimizer(optimizer: optimizer)
        OLinear.setOptimizer(optimizer: optimizer)
        
    }
    
    func updateWeights(layerNum: Int) -> Int {
        
        var layerNum = KLinear.updateWeights(layerNum: layerNum)
        layerNum = QLinear.updateWeights(layerNum: layerNum)
        layerNum = VLinear.updateWeights(layerNum: layerNum)
        layerNum = OLinear.updateWeights(layerNum: layerNum)
        
        return layerNum
    }
}
