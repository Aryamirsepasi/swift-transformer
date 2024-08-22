import Foundation
import Accelerate
import MLX
//needed
class DecoderLayer {
    var selfAttentionNorm: LayerNormalization
    var encAttnLayerNorm: LayerNormalization
    var ffLayerNorm: LayerNormalization
    var selfAttention: MultiHeadAttention
    var encoderAttention: MultiHeadAttention
    var positionWiseFeedForward: PositionwiseFeedforward
    var dropout: Dropout

    init(dModel: Int, headsNum: Int, dFF: Int, dropoutRate: Float, dataType: DType) {
                
        self.selfAttentionNorm = LayerNormalization(normalizedShape: [dModel], epsilon: 1e-6, dataType: dataType)
        self.encAttnLayerNorm = LayerNormalization(normalizedShape: [dModel], epsilon: 1e-6, dataType: dataType)
        self.ffLayerNorm = LayerNormalization(normalizedShape: [dModel], epsilon: 1e-6, dataType: dataType)
        self.selfAttention = MultiHeadAttention(dModel: dModel, headsNum: headsNum, dropoutRate: dropoutRate, dataType: dataType)
        self.encoderAttention = MultiHeadAttention(dModel: dModel, headsNum: headsNum, dropoutRate: dropoutRate, dataType: dataType)
        self.positionWiseFeedForward = PositionwiseFeedforward(dModel: dModel, dFF: dFF, dropoutRate: dropoutRate)
        self.dropout = Dropout(rate: dropoutRate, dataType: dataType)
    }

    func forward(trg: MLXArray, trgMask: MLXArray, src: MLXArray, srcMask: MLXArray, training: Bool) -> (MLXArray, MLXArray) {
        var _trg : MLXArray
        (_trg, _) = self.selfAttention.forward(query: trg, key: trg, value: trg, mask: trgMask, training: training)
        var trgvar = trg
        trgvar = self.selfAttentionNorm.forward(X: trgvar + self.dropout.forward(X: _trg, training: training))
        
        
        var attention : MLXArray
        (_trg, attention) = self.encoderAttention.forward(query: trgvar, key: src, value: src, mask: srcMask, training: training)
        trgvar = self.encAttnLayerNorm.forward(X: trg + self.dropout.forward(X: _trg, training: training))
        
        return (trg, attention)
    }

    func backward(error: MLXArray) -> (MLXArray,MLXArray) {
        var errorvar = self.ffLayerNorm.backward(error: error)

        var _error = self.positionWiseFeedForward.backward(error: self.dropout.backward(errorvar))
        
        errorvar = self.encAttnLayerNorm.backward(error: errorvar + _error)

        var encError1: MLXArray
        var encError2: MLXArray
        (_error, encError1, encError2) = self.encoderAttention.backward(error: self.dropout.backward(errorvar))
        errorvar = self.selfAttentionNorm.backward(error: errorvar + _error)

        var _error2: MLXArray
        var _error3: MLXArray
        (_error, _error2, _error3) = self.selfAttention.backward(error: self.dropout.backward(errorvar))
        
        return (_error + _error2 + _error3 + error, encError1 + encError2)
    }

    func setOptimizer(_ optimizer: Optimizer) {
        selfAttentionNorm.setOptimizer(optimizer: optimizer)
        encAttnLayerNorm.setOptimizer(optimizer: optimizer)
        ffLayerNorm.setOptimizer(optimizer: optimizer)
        selfAttention.setOptimizer(optimizer: optimizer)
        encoderAttention.setOptimizer(optimizer: optimizer)
        positionWiseFeedForward.setOptimizer(optimizer: optimizer)
    }

    func updateWeights(_ layerNum: Int) -> Int {
        var layerNum = layerNum
        layerNum = selfAttentionNorm.updateWeights(layerNum: layerNum)
        layerNum = encAttnLayerNorm.updateWeights(layerNum: layerNum)
        layerNum = ffLayerNorm.updateWeights(layerNum: layerNum)
        layerNum = selfAttention.updateWeights(layerNum: layerNum)
        layerNum = encoderAttention.updateWeights(layerNum: layerNum)
        layerNum = positionWiseFeedForward.updateWeights(startingLayerNum: layerNum)
        return layerNum
    }
}
