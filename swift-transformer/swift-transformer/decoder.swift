import Foundation
import Accelerate

class Decoder {
    var tokenEmbedding: Embedding
    var positionEmbedding: PositionalEncoding
    var layers: [DecoderLayer]
    var fcOut: Dense
    var dropout: Dropout
    var scale: Float
    var activation: Identity

    init(trgVocabSize: Int, headsNum: Int, layersNum: Int, dModel: Int, dFF: Int, dropoutRate: Float, maxLen: Int = 5000, dataType: [Float]) {
        self.tokenEmbedding = Embedding(inputDim: trgVocabSize, outputDim: dModel, dataType: dataType)
        self.positionEmbedding = PositionalEncoding(maxLen: maxLen, dModel: dModel, dropoutRate: dropoutRate, dataType: dataType)
        self.layers = []
        for _ in 0..<layersNum {
            self.layers.append(DecoderLayer(dModel: dModel, headsNum: headsNum, dFF: dFF, dropoutRate: dropoutRate, dataType: dataType))
        }
        self.fcOut = Dense(unitsNum: trgVocabSize, inputsNum: dModel, useBias: true, dataType: dataType)
        self.dropout = Dropout(rate: dropoutRate, dataType: dataType)
        self.scale = sqrt(Float(dModel))
        self.activation = Identity()
    }

    func forward(trg: [[Float]], trgMask: [[[Float]]], src: [[[Float]]], srcMask: [[[Float]]], training: Bool) -> ([[[Float]]], [[[Float]]]) {
        var trg = trg
        let batchSize = trg.count
        let trgSeqLen = trg[0].count

        // Adjust input dimensions for token embedding and position embedding
        var embeddedTrg = tokenEmbedding.forward(input: trg)
        embeddedTrg = embeddedTrg.map { $0.map { $0.map { $0 * self.scale } } }
        embeddedTrg = positionEmbedding.forward(x: embeddedTrg)
        embeddedTrg = dropout.forward(embeddedTrg, training: training)

        var attention: [[[Float]]] = []

        for layer in layers {
            let (layerOutput, layerAttention) = layer.forward(trg: embeddedTrg, trgMask: trgMask, src: src, srcMask: srcMask, training: training)
            embeddedTrg = layerOutput
            attention = layerAttention
        }

        let output = fcOut.forward(embeddedTrg)
        let activatedOutput = activation.forward(x: output)

        return (activatedOutput, attention)
    }

    func backward(error: [[[Float]]]) {
        var error = activation.backward(grad: error)
        error = fcOut.backward(error)

        for layer in layers.reversed() {
            let (layerError, encError) = layer.backward(error)
            error = layerError + encError
        }

        error = dropout.backward(error)
        error = positionEmbedding.backward(error: error)
        error = error.map { $0.map { $0.map { $0 * self.scale } } }
        error = tokenEmbedding.backward(error: error) ?? []
    }

    func setOptimizer(_ optimizer: Optimizer) {
        tokenEmbedding.setOptimizer(optimizer: optimizer)
        for layer in layers {
            layer.setOptimizer(optimizer)
        }
        fcOut.setOptimizer(optimizer: optimizer)
    }

    func updateWeights() {
        var layerNum = 1
        layerNum = tokenEmbedding.updateWeights(layerNum: layerNum)
        for layer in layers {
            layerNum = layer.updateWeights(layerNum)
        }
        fcOut.updateWeights(layerNum: layerNum)
    }
}
