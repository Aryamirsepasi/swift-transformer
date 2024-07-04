import Accelerate

class Encoder {
    var tokenEmbedding: Embedding
    var positionEmbedding: PositionalEncoding
    var layers: [EncoderLayer]
    var dropout: Dropout
    var scale: Float

    init(srcVocabSize: Int, headsNum: Int, layersNum: Int, dModel: Int, dFF: Int, dropoutRate: Float, maxLen: Int = 5000, dataType: [Float]) {
        self.tokenEmbedding = Embedding(inputDim: srcVocabSize, outputDim: dModel, dataType: dataType)
        self.positionEmbedding = PositionalEncoding(maxLen: maxLen, dModel: dModel, dropoutRate: dropoutRate, dataType: dataType)
        self.layers = []
        for _ in 0..<layersNum {
            self.layers.append(EncoderLayer(dModel: dModel, headsNum: headsNum, dFF: dFF, dropoutRate: dropoutRate, dataType: dataType))
        }
        self.dropout = Dropout(rate: dropoutRate, dataType: dataType)
        self.scale = sqrt(Float(dModel))
    }

    func forward(src: [[Float]], srcMask: [[[Float]]], training: Bool) -> [[[Float]]] {
        var src = tokenEmbedding.forward(input: src)
        src = src.map { $0.map { $0 * self.scale } }
        src = positionEmbedding.forward(x: src)
        src = dropout.forward(src, training: training)

        for layer in layers {
            src = layer.forward(src: src, srcMask: srcMask, training: training)
        }

        return src
    }

    func backward(error: [[[Float]]]) -> [[[Float]]] {
        var error = error
        for layer in layers.reversed() {
            error = layer.backward(error: error)
        }
        
        error = dropout.backward(error: error)
        error = positionEmbedding.backward(error: error).map { $0.map { $0 * self.scale } }
        return tokenEmbedding.backward(error: error) ?? []
    }

    func setOptimizer(_ optimizer: Optimizer) {
        tokenEmbedding.setOptimizer(optimizer: optimizer)
        for layer in layers {
            layer.setOptimizer(optimizer)
        }
    }

    func updateWeights() {
        var layerNum = 1
        layerNum = tokenEmbedding.updateWeights(layerNum: layerNum)
        for layer in layers {
            layerNum = layer.updateWeights(layerNum)
        }
    }
}


