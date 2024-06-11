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

    init(trgVocabSize: Int, headsNum: Int, layersNum: Int, dModel: Int, dFF: Int, dropoutRate: Float, maxLen: Int = 5000) {
        self.tokenEmbedding = Embedding(inputDim: trgVocabSize, outputDim: dModel)
        self.positionEmbedding = PositionalEncoding(maxLen: maxLen, dModel: dModel, dropoutRate: dropoutRate)
        self.layers = []
        for _ in 0..<layersNum {
            self.layers.append(DecoderLayer(dModel: dModel, headsNum: headsNum, dFF: dFF, dropoutRate: dropoutRate))
        }
        self.fcOut = Dense(unitsNum: trgVocabSize, inputsNum: dModel, useBias: true)
        self.dropout = Dropout(rate: dropoutRate)
        self.scale = sqrt(Float(dModel))
        self.activation = Identity()
    }

    func forward(trg: [[Int]], trgMask: [[[Float]]], src: [[Float]], srcMask: [[[Float]]], training: Bool) -> ([[Float]], [[Float]]) {
        var trgSeq = tokenEmbedding.forward(input: trg[0]).flatMap { $0 }
        trgSeq = trgSeq.map { $0 * self.scale }
        var trgSeqBatch = [trgSeq]
        positionEmbedding.forward(x: &trgSeqBatch)
        trgSeq = trgSeqBatch.first!

        var allOutputs: [Float] = []
        var allAttentions: [[Float]] = []

        for layer in layers {
            let (layerOutput, layerAttention) = layer.forward(trg: trgSeq, trgMask: trgMask[0], src: src.flatMap { $0 }, srcMask: srcMask[0], training: training)
            trgSeq = layerOutput
            allOutputs.append(contentsOf: layerOutput)
            allAttentions.append(layerAttention)
        }

        let output = fcOut.forward(trgSeq)
        let activatedOutput = activation.forward(x: output)

        return ([activatedOutput], allAttentions)
    }

    func backward(error: [[Float]]) {
        for singleError in error {
            var error = self.activation.backward(grad: singleError)
            error = self.fcOut.backward(error)

            for layer in layers.reversed() {
                let (layerError, _) = layer.backward(error)
                error = layerError
            }

            error = self.dropout.backward(error)
            error = positionEmbedding.backward(error: [error]).first!
            tokenEmbedding.backward(error: [error])
        }
    }

    func setOptimizer(_ optimizer: Optimizer) {
        tokenEmbedding.setOptimizer(optimizer: optimizer)
        layers.forEach { $0.setOptimizer(optimizer) }
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
