import Foundation
import Accelerate
import MLX

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
        
        print ("entered encoder_layer forward")

        //print(src.shape)
        //print(src.shape)
        //print(src.shape)

        var (_src, _) = self.selfAttention.forward(query: src, key: src, value: src, mask: srcMask, training: training)
        
        var srcvar = self.selfAttentionNorm.forward(X: src + self.dropout.forward(X: _src, training: training))
        
        _src = self.positionWiseFeedForward.forward(X: src, training: training)
        
        srcvar = self.ffLayerNorm.forward(X: src + self.dropout.forward(X: _src, training: training))
        
        print ("exited encoder_layer forward")

        return srcvar
    }

    func backward(error: MLXArray) -> MLXArray {
        
        print("entered encoder_layer backward")

        var errorvar = self.ffLayerNorm.backward(error: error)
        
        var _error = self.positionWiseFeedForward.backward(error: self.dropout.backward(errorvar))
        
        errorvar = self.selfAttentionNorm.backward(error:errorvar + _error)
        
        var _error2, _error3 : MLXArray
        
        (_error, _error2, _error3) = self.selfAttention.backward(error: self.dropout.backward(errorvar))
        
        print("exited encoder_layer backward")

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
