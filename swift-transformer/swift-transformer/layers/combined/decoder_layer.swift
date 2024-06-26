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

    init(dModel: Int, headsNum: Int, dFF: Int, dropoutRate: Float, dataType: [Float]) {
        self.selfAttentionNorm = LayerNormalization(normalizedShape: [dModel], epsilon: 1e-6, dataType: dataType)
        self.encAttnLayerNorm = LayerNormalization(normalizedShape: [dModel], epsilon: 1e-6, dataType: dataType)
        self.ffLayerNorm = LayerNormalization(normalizedShape: [dModel], epsilon: 1e-6, dataType: dataType)
        self.selfAttention = MultiHeadAttention(dModel: dModel, headsNum: headsNum, dropoutRate: dropoutRate, dataType: dataType)
        self.encoderAttention = MultiHeadAttention(dModel: dModel, headsNum: headsNum, dropoutRate: dropoutRate, dataType: dataType)
        self.positionWiseFeedForward = PositionwiseFeedforward(dModel: dModel, dFF: dFF, dropoutRate: dropoutRate)
        self.dropout = Dropout(rate: dropoutRate, dataType: dataType)
    }

    func forward(trg: [[Float]], trgMask: [[Float]], src: [[Float]], srcMask: [[Float]], training: Bool) -> ([[Float]], [[Float]]) {
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

    func backward(_ error: [[Float]]) -> ([[Float]], [[Float]]) {
        var error = error

        // FF Layer Norm Backward
        error = ffLayerNorm.backward(error)

        // Positionwise Feedforward Backward
        let feedForwardError = positionWiseFeedForward.backward(dropout.backward(error))
        error = error + feedForwardError

        // Encoder Attention Layer Norm Backward
        error = encAttnLayerNorm.backward(error)

        //NEEDS CORRECTION. THE ERRORS RESULT FROM BACKWARD MUST HAVE 3 Variables
        // Encoder Attention Backward
        //let (encoderAttnError, encError1, encError2) = encoderAttention.backward(dropout.backward(error))
        let encoderAttnError = encoderAttention.backward(error: dropout.backward(error))
        let encError1 = encoderAttention.backward(error: dropout.backward(error))
        let encError2 = encoderAttention.backward(error: dropout.backward(error))
        
        error = error + encoderAttnError

        // Self Attention Layer Norm Backward
        error = selfAttentionNorm.backward(error)
        
        //NEEDS CORRECTION. THE ERRORS RESULT FROM BACKWARD MUST HAVE 3 Variables
        // Self Attention Backward
        //let (selfAttnError, selfError1, selfError2) = selfAttention.backward(dropout.backward(error))
        let selfAttnError = selfAttention.backward(error: dropout.backward(error))
        let selfError1 = selfAttention.backward(error: dropout.backward(error))
        let selfError2 = selfAttention.backward(error: dropout.backward(error))

        // Combine all errors
        let combinedError = selfAttnError + selfError1 + selfError2 + error
        let encoderCombinedError = encError1 + encError2

        return (combinedError, encoderCombinedError)
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

// Helper function to add two 2D arrays element-wise
func addArrays(_ arr1: [[Float]], _ arr2: [[Float]]) -> [[Float]] {
    guard arr1.count == arr2.count, arr1[0].count == arr2[0].count else { return [] }
    var result = arr1
    for i in 0..<arr1.count {
        for j in 0..<arr1[0].count {
            result[i][j] = arr1[i][j] + arr2[i][j]
        }
    }
    return result
}
