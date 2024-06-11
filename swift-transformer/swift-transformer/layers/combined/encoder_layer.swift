import Foundation
import Accelerate

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

    func forward(_ src: [[Float]], srcMask: [[Float]], training: Bool) -> [[Float]] {
        var srcFlat = src.flatMap { $0 }
        var srcMaskFlat = srcMask.flatMap { $0 }

        let (attentionOutputFlat, _) = selfAttention.forward(query: srcFlat, key: srcFlat, value: srcFlat, mask: srcMaskFlat, training: training)
        let attentionOutput = reshape(attentionOutputFlat, rows: src.count, cols: src[0].count)
        
        let droppedAttentionOutput = dropout.forward(attentionOutputFlat, shape: (attentionOutputFlat.count, 1), training: training)
        let droppedAttentionOutputReshaped = reshape(droppedAttentionOutput, rows: src.count, cols: src[0].count)
        src = selfAttentionNorm.forward(addArrays(src, droppedAttentionOutputReshaped))

        let feedForwardOutput = positionWiseFeedForward.forward(src, training: training)
        let feedForwardOutputFlat = feedForwardOutput.flatMap { $0 }
        let droppedFeedForwardOutput = dropout.forward(feedForwardOutputFlat, shape: (feedForwardOutputFlat.count, 1), training: training)
        let droppedFeedForwardOutputReshaped = reshape(droppedFeedForwardOutput, rows: src.count, cols: src[0].count)
        src = ffLayerNorm.forward(addArrays(src, droppedFeedForwardOutputReshaped))

        return src
    }

    func backward(_ error: [[Float]]) -> [[Float]] {
        var errorFlat = error.flatMap { $0 }

        var errorNorm = ffLayerNorm.backward(error)
        let feedForwardError = positionWiseFeedForward.backward(dropout.backward(errorNorm.flatMap { $0 }))
        errorNorm = addArrays(errorNorm, feedForwardError)

        var attentionErrorFlat, qErrorFlat, kErrorFlat, vErrorFlat: [Float]
        (attentionErrorFlat, qErrorFlat, kErrorFlat, vErrorFlat) = selfAttention.backward(dropout.backward(errorNorm.flatMap { $0 }))
        
        let attentionError = reshape(attentionErrorFlat, rows: error.count, cols: error[0].count)

        return addArrays(attentionError, errorNorm)
    }

    func setOptimizer(_ optimizer: Optimizer) {
        selfAttentionNorm.setOptimizer(optimizer: optimizer)
        ffLayerNorm.setOptimizer(optimizer: optimizer)
        selfAttention.setOptimizer(optimizer)
        positionWiseFeedForward.setOptimizer(optimizer: optimizer)
    }

    func updateWeights(_ layerNum: Int) -> Int {
        var layerNum = layerNum
        layerNum = selfAttentionNorm.updateWeights(layerNum: layerNum)
        layerNum = ffLayerNorm.updateWeights(layerNum: layerNum)
        layerNum = selfAttention.updateWeights(layerNum)
        layerNum = positionWiseFeedForward.updateWeights(startingLayerNum: layerNum)
        return layerNum
    }

    // Helper function to add two arrays element-wise
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

    // Helper function to reshape 1D array to 2D array
    func reshape(_ array: [Float], rows: Int, cols: Int) -> [[Float]] {
        var reshapedArray = [[Float]]()
        for i in 0..<rows {
            let start = i * cols
            let end = start + cols
            let row = Array(array[start..<end])
            reshapedArray.append(row)
        }
        return reshapedArray
    }
}
