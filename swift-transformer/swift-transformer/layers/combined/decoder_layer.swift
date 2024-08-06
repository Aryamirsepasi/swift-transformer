import Foundation
import Accelerate

//needed 
class DecoderLayer {
    var selfAttentionNorm: LayerNormalization
    var encAttnLayerNorm: LayerNormalization
    var ffLayerNorm: LayerNormalization
    var selfAttention: MultiHeadAttention
    var encoderAttention: MultiHeadAttention
    var positionWiseFeedForward: PositionwiseFeedforward
    var dropout: Dropout

    init(dModel: Int, headsNum: Int, dFF: Int, dropoutRate: Float, dataType: [Float]) {
        self.selfAttentionNorm = LayerNormalization(normalizedShape: [dModel], epsilon: 1e-6, dataType: dataType)
        self.encAttnLayerNorm = LayerNormalization(normalizedShape: [dModel], epsilon: 1e-6, dataType: dataType)
        self.ffLayerNorm = LayerNormalization(normalizedShape: [dModel], epsilon: 1e-6, dataType: dataType)
        self.selfAttention = MultiHeadAttention(dModel: dModel, headsNum: headsNum, dropoutRate: dropoutRate, dataType: dataType)
        self.encoderAttention = MultiHeadAttention(dModel: dModel, headsNum: headsNum, dropoutRate: dropoutRate, dataType: dataType)
        self.positionWiseFeedForward = PositionwiseFeedforward(dModel: dModel, dFF: dFF, dropoutRate: dropoutRate)
        self.dropout = Dropout(rate: dropoutRate, dataType: dataType)
    }

    func forward(trg: [[[Float]]], trgMask: [[[Float]]], src: [[[Float]]], srcMask: [[[Float]]], training: Bool) -> ([[[Float]]], [[[Float]]]) {
        var trg = trg
        
        let (selfAttnOutput, _) = self.selfAttention.forward(query: trg, key: trg, value: trg, mask: trgMask, training: training)
        let droppedSelfAttnOutput = dropout.forward(selfAttnOutput, training: training)
        trg = selfAttentionNorm.forward(trg + droppedSelfAttnOutput)

        let (encoderAttnOutput, attention) = self.encoderAttention.forward(query: trg, key: src, value: src, mask: srcMask, training: training)
        let droppedEncoderAttnOutput = dropout.forward(encoderAttnOutput, training: training)
        trg = encAttnLayerNorm.forward(trg + droppedEncoderAttnOutput)

        let feedForwardOutput = positionWiseFeedForward.forward(trg, training: training)
        let droppedFeedForwardOutput = dropout.forward(feedForwardOutput, training: training)
        trg = ffLayerNorm.forward(trg + droppedFeedForwardOutput)

        return (trg, attention)
    }

    func backward(_ error: [[[Float]]]) -> ([[[Float]]], [[[Float]]]) {
        var error = self.ffLayerNorm.backward(error)

        var _error = self.positionWiseFeedForward.backward(self.dropout.backward(error))
        error = self.encAttnLayerNorm.backward(error + _error)

        var encError1: [[[Float]]]
        var encError2: [[[Float]]]
        (_error, encError1, encError2) = self.encoderAttention.backward(error: self.dropout.backward(error))
        error = self.selfAttentionNorm.backward(error + _error)

        var _error2: [[[Float]]]
        var _error3: [[[Float]]]
        (_error, _error2, _error3) = self.selfAttention.backward(error: self.dropout.backward(error))
        
        return (addArrays(_error, _error2, _error3, error), addArrays(encError1, encError2))
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

// Helper function to add multiple 3D arrays element-wise
func addArrays(_ arrays: [[[Float]]]...) -> [[[Float]]] {
    guard let firstArray = arrays.first else { return [] }
    let resultShape = (firstArray.count, firstArray[0].count, firstArray[0][0].count)
    
    var result = [[[Float]]](repeating: [[Float]](repeating: [Float](repeating: 0.0, count: resultShape.2), count: resultShape.1), count: resultShape.0)
    
    for array in arrays {
        for i in 0..<resultShape.0 {
            for j in 0..<resultShape.1 {
                for k in 0..<resultShape.2 {
                    result[i][j][k] += array[i][j][k]
                }
            }
        }
    }
    return result
}
