import Accelerate

class Encoder {
    var tokenEmbedding: Embedding
    var positionEmbedding: PositionalEncoding
    var layers: [EncoderLayer]
    var dropout: Dropout
    var scale: Float

    init(srcVocabSize: Int, headsNum: Int, layersNum: Int, dModel: Int, dFF: Int, dropoutRate: Float, maxLen: Int = 5000) {
        self.tokenEmbedding = Embedding(inputDim: srcVocabSize, outputDim: dModel)
        self.positionEmbedding = PositionalEncoding(maxLen: maxLen, dModel: dModel, dropoutRate: dropoutRate)
        self.layers = []
        for _ in 0..<layersNum {
            self.layers.append(EncoderLayer(dModel: dModel, headsNum: headsNum, dFF: dFF, dropoutRate: dropoutRate))
        }
        self.dropout = Dropout(rate: dropoutRate)
        self.scale = sqrt(Float(dModel))
    }

    func forward(src: [Int], srcMask: [Float], training: Bool) -> [[Float]] {
        var src = tokenEmbedding.forward(input: src).map { $0.map { $0 * self.scale } }
        positionEmbedding.forward(x: &src)
        src = src.map { self.dropout.forward($0, training: training) }

        for layer in layers {
            src = src.map { layer.forward($0, srcMask: [srcMask], training: training) }
        }

        return src
    }

    func backward(error: [[Float]]) {
        var error = error
        for layer in layers.reversed() {
            error = error.map { layer.backward($0) }
        }
        
        error = error.map { self.dropout.backward($0) }
        error = positionEmbedding.backward(error: error).map { $0.map { $0 * self.scale } }
        tokenEmbedding.backward(error: error)
    }

    func setOptimizer(_ optimizer: Optimizer) {
        tokenEmbedding.setOptimizer(optimizer: optimizer)
        layers.forEach { $0.setOptimizer(optimizer) }
    }

    func updateWeights() {
        var layerNum = 1
        layerNum = tokenEmbedding.updateWeights(layerNum: layerNum)
        layers.forEach {
            layerNum = $0.updateWeights(layerNum)
        }
    }
}
