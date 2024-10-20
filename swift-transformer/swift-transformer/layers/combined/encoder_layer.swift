import Foundation
import Accelerate
import MLX
import MLXNN

//needed
class EncoderLayer {
    var selfAttentionNorm: LayerNormalization
    var ffLayerNorm: LayerNormalization
    var selfAttention: MultiHeadAttention
    var positionWiseFeedForward: PositionwiseFeedforward
    var dropout: Dropout

    init(dModel: Int, headsNum: Int, dFF: Int, dropoutRate: Float, dataType: DType) {
        
        self.selfAttentionNorm = LayerNormalization(normalizedShape: [dModel], epsilon: 1e-6, dataType: dataType)
        self.ffLayerNorm = LayerNormalization(normalizedShape: [dModel], epsilon: 1e-6, dataType: dataType)
        self.selfAttention = MultiHeadAttention(dModel: dModel, headsNum: headsNum, dropoutRate: dropoutRate, dataType: dataType)
        self.positionWiseFeedForward = PositionwiseFeedforward(dModel: dModel, dFF: dFF, dropoutRate: dropoutRate)
        self.dropout = Dropout(rate: dropoutRate, dataType: dataType)
        
    }

    func forward(src: MLXArray, srcMask: MLXArray, training: Bool) -> MLXArray {

        var (_src, _) = self.selfAttention.forward(query: src, key: src, value: src, mask: srcMask, training: training)
        
        var srcvar = self.selfAttentionNorm.forward(X: MLX.add(src,self.dropout.forward(X: _src, training: training)))
        
        _src = self.positionWiseFeedForward.forward(X: src, training: training)
        
        srcvar = self.ffLayerNorm.forward(X: MLX.add(src,self.dropout.forward(X: _src, training: training)))
        
        return srcvar
    }

    func backward(error: MLXArray) -> MLXArray {
        
        var errorvar = self.ffLayerNorm.backward(error: error)
        
        var _error = self.positionWiseFeedForward.backward(error: self.dropout.backward(errorvar))
        
        errorvar = self.selfAttentionNorm.backward(error:errorvar + _error)
        
        var _error2, _error3 : MLXArray
        
        (_error, _error2, _error3) = self.selfAttention.backward(error: self.dropout.backward(errorvar))
        
        return _error + _error2 + _error3 + error
        
    }

    func setOptimizer(_ optimizer: Optimizer) {
        
        selfAttentionNorm.setOptimizer(optimizer: optimizer)
        ffLayerNorm.setOptimizer(optimizer: optimizer)
        selfAttention.setOptimizer(optimizer: optimizer)
        positionWiseFeedForward.setOptimizer(optimizer: optimizer)
        
    }

    func updateWeights(layerNum: Int) -> Int {
        
        var layerNum = layerNum
        layerNum = selfAttentionNorm.updateWeights(layerNum: layerNum)
        layerNum = ffLayerNorm.updateWeights(layerNum: layerNum)
        layerNum = selfAttention.updateWeights(layerNum: layerNum)
        layerNum = positionWiseFeedForward.updateWeights(startingLayerNum: layerNum)
        
        return layerNum
    }
}
