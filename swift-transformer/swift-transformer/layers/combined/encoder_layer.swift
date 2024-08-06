import Foundation
import Accelerate

//needed
class EncoderLayer {
    var selfAttentionNorm: LayerNormalization
    var ffLayerNorm: LayerNormalization
    var selfAttention: MultiHeadAttention
    var positionWiseFeedForward: PositionwiseFeedforward
    var dropout: Dropout

    init(dModel: Int, headsNum: Int, dFF: Int, dropoutRate: Float, dataType: [Float]) {
        self.selfAttentionNorm = LayerNormalization(normalizedShape: [dModel], epsilon: 1e-6, dataType: dataType)
        self.ffLayerNorm = LayerNormalization(normalizedShape: [dModel], epsilon: 1e-6, dataType: dataType)
        self.selfAttention = MultiHeadAttention(dModel: dModel, headsNum: headsNum, dropoutRate: dropoutRate, dataType: dataType)
        self.positionWiseFeedForward = PositionwiseFeedforward(dModel: dModel, dFF: dFF, dropoutRate: dropoutRate)
        self.dropout = Dropout(rate: dropoutRate, dataType: dataType)
    }

    func forward(_ src: [[[Float]]], srcMask: [[[Float]]], training: Bool) -> [[[Float]]] {
        let (attentionOutput, _) = selfAttention.forward(query: src, key: src, value: src, mask: srcMask, training: training)
        var src = selfAttentionNorm.forward(src + attentionOutput)
        
        let feedForwardOutput = positionWiseFeedForward.forward(src, training: training)
        src = self.ffLayerNorm.forward(src + feedForwardOutput)

        return src
    }

    func backward(_ error: [[[Float]]]) -> [[[Float]]] {
        var errorNorm = ffLayerNorm.backward(error)
        let feedForwardError = positionWiseFeedForward.backward(dropout.backward(errorNorm))
        errorNorm = addArrays(errorNorm, feedForwardError)

        let (attentionError, _, _) = selfAttention.backward(error: dropout.backward(errorNorm))
        return addArrays(attentionError, errorNorm)
    }

    func setOptimizer(_ optimizer: Optimizer) {
        selfAttentionNorm.setOptimizer(optimizer: optimizer)
        ffLayerNorm.setOptimizer(optimizer: optimizer)
        selfAttention.setOptimizer(optimizer: optimizer)
        positionWiseFeedForward.setOptimizer(optimizer: optimizer)
    }

    func updateWeights(_ layerNum: Int) -> Int {
        var layerNum = layerNum
        layerNum = selfAttentionNorm.updateWeights(layerNum: layerNum)
        layerNum = ffLayerNorm.updateWeights(layerNum: layerNum)
        layerNum = selfAttention.updateWeights(layerNum: layerNum)
        layerNum = positionWiseFeedForward.updateWeights(startingLayerNum: layerNum)
        return layerNum
    }

    // Helper function to add two arrays element-wise
    func addArrays(_ arr1: [[[Float]]], _ arr2: [[[Float]]]) -> [[[Float]]] {
        guard arr1.count == arr2.count, arr1[0].count == arr2[0].count, arr1[0][0].count == arr2[0][0].count else { return [] }
        var result = arr1
        for i in 0..<arr1.count {
            for j in 0..<arr1[0].count {
                for k in 0..<arr1[0][0].count {
                    result[i][j][k] = arr1[i][j][k] + arr2[i][j][k]
                }
            }
        }
        return result
    }
}
