import Foundation
import Accelerate

class DecoderLayer {
    var selfAttentionNorm: LayerNormalization
    var encAttnLayerNorm: LayerNormalization
    var ffLayerNorm: LayerNormalization
    var selfAttention: MultiHeadAttention
    var encoderAttention: MultiHeadAttention
    var positionWiseFeedForward: PositionwiseFeedforward
    var dropout: Dropout

    init(dModel: Int, headsNum: Int, dFF: Int, dropoutRate: Float) {
        self.selfAttentionNorm = LayerNormalization(normalizedShape: [dModel], epsilon: 1e-6)
        self.encAttnLayerNorm = LayerNormalization(normalizedShape: [dModel], epsilon: 1e-6)
        self.ffLayerNorm = LayerNormalization(normalizedShape: [dModel], epsilon: 1e-6)
        self.selfAttention = MultiHeadAttention(dModel: dModel, headsNum: headsNum, dropoutRate: dropoutRate)
        self.encoderAttention = MultiHeadAttention(dModel: dModel, headsNum: headsNum, dropoutRate: dropoutRate)
        self.positionWiseFeedForward = PositionwiseFeedforward(dModel: dModel, dFF: dFF, dropoutRate: dropoutRate)
        self.dropout = Dropout(rate: dropoutRate)
    }

    func forward(trg: [Float], trgMask: [[Float]], src: [Float], srcMask: [[Float]], training: Bool) -> ([Float], [Float]) {
        var trg = trg
        let (selfAttnOutput, _) = self.selfAttention.forward(query: trg, key: trg, value: trg, mask: trgMask, training: training)
        let droppedSelfAttnOutput = dropout.forward(selfAttnOutput, training: training)
        trg = selfAttentionNorm.forward([trg + droppedSelfAttnOutput])[0]

        let (encoderAttnOutput, attention) = self.encoderAttention.forward(query: trg, key: src, value: src, mask: srcMask, training: training)
        let droppedEncoderAttnOutput = dropout.forward(encoderAttnOutput, training: training)
        trg = encAttnLayerNorm.forward([trg + droppedEncoderAttnOutput])[0]

        let feedForwardOutput = positionWiseFeedForward.forward(trg, training: training)
        let droppedFeedForwardOutput = dropout.forward(feedForwardOutput, training: training)
        trg = ffLayerNorm.forward([trg + droppedFeedForwardOutput])[0]

        return (trg, attention)
    }

    func backward(_ error: [Float]) -> ([Float], [Float]) {
        var error = ffLayerNorm.backward([error])[0]
        var feedForwardError = positionWiseFeedForward.backward(error)
        let combinedError = addArrays(error, feedForwardError)
        error = encAttnLayerNorm.backward([combinedError])[0]
        let (encoderAttentionError, _, _) = encoderAttention.backward(dropout.backward(error))
        error = selfAttentionNorm.backward([error + encoderAttentionError])[0]
        var errorAfterDropout = dropout.backward(error)
        let (selfAttentionError, _, _) = selfAttention.backward(errorAfterDropout)
        let totalError = addArrays(selfAttentionError, feedForwardError)
        return (totalError, encoderAttentionError)
    }

    func setOptimizer(_ optimizer: Optimizer) {
        selfAttentionNorm.setOptimizer(optimizer: optimizer)
        encAttnLayerNorm.setOptimizer(optimizer: optimizer)
        ffLayerNorm.setOptimizer(optimizer: optimizer)
        selfAttention.setOptimizer(optimizer)
        encoderAttention.setOptimizer(optimizer)
        positionWiseFeedForward.setOptimizer(optimizer: optimizer)
    }

    func updateWeights(_ layerNum: Int) -> Int {
        var layerNum = layerNum
        layerNum = selfAttentionNorm.updateWeights(layerNum: layerNum)
        layerNum = encAttnLayerNorm.updateWeights(layerNum: layerNum)
        layerNum = ffLayerNorm.updateWeights(layerNum: layerNum)
        layerNum = selfAttention.updateWeights(layerNum)
        layerNum = encoderAttention.updateWeights(layerNum)
        layerNum = positionWiseFeedForward.updateWeights(startingLayerNum: layerNum)
        return layerNum
    }

    // Helper function to add two arrays element-wise
    func addArrays(_ arr1: [Float], _ arr2: [Float]) -> [Float] {
        guard arr1.count == arr2.count else { return [] }
        var result = [Float](repeating: 0.0, count: arr1.count)
        for i in 0..<arr1.count {
            result[i] = arr1[i] + arr2[i]
        }
        return result
    }
}
