//
//  encoder.swift
//  swift-transformer
//
//  Created by Arya Mirsepasi on 29.04.24.
//

import Foundation
import Matft

struct EncoderLayer {
    var selfAttention: MultiHeadAttention
    var feedForward: PositionwiseFeedforward
    var normLayer1: LayerNorm
    var normLayer2: LayerNorm
    var dropout: Dropout

    init(dModel: Int, headsNum: Int, dFF: Int, dropoutRate: Float) {
        self.selfAttention = MultiHeadAttention(dModel: dModel, headsNum: headsNum, dropoutRate: dropoutRate)
        self.feedForward = PositionwiseFeedforward(dModel: dModel, dFF: dFF, dropoutRate: dropoutRate)
        self.normLayer1 = LayerNorm(featureSize: dModel, epsilon: 0.000001)
        self.normLayer2 = LayerNorm(featureSize: dModel, epsilon: 0.000001)
        self.dropout = Dropout(rate: dropoutRate)
    }

    func forward(_ src: MfArray, srcMask: MfArray, training: Bool) -> MfArray {
        let (attentionOutput, _) = selfAttention.forward(src, key: src, value: src, mask: srcMask, training: training)
        var srcNorm = normLayer1.forward(src + dropout.forward(attentionOutput, training: training))
        srcNorm = normLayer2.forward(srcNorm + dropout.forward(feedForward.forward(srcNorm, training: training), training: training))
        return srcNorm
    }
}

struct Encoder {
    var tokenEmbedding: Embedding
    var positionEmbedding: PositionalEncoding
    var layers: [EncoderLayer]
    var dropout: Dropout
    let scale: Float

    init(srcVocabSize: Int, headsNum: Int, layersNum: Int, dModel: Int, dFF: Int, dropout: Float, maxLen: Int = 5000) {
        self.tokenEmbedding = Embedding(vocabSize: srcVocabSize, embeddingDim: dModel)
        self.positionEmbedding = PositionalEncoding(maxLen: maxLen, dModel: dModel)
        self.scale = sqrt(Float(dModel))
        self.layers = (0..<layersNum).map { _ in EncoderLayer(dModel: dModel, headsNum: headsNum, dFF: dFF, dropoutRate: dropout) }
        self.dropout = Dropout(rate: dropout)
    }

    func forward(src: MfArray, srcMask: MfArray, training: Bool) -> MfArray {
        var src = tokenEmbedding.forward(src) * scale
        src = positionEmbedding.forward(src)
        src = dropout.forward(src, training: training)

        for layer in layers {
            src = layer.forward(src, srcMask: srcMask, training: training)
        }
        return src
    }
}
