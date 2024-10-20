import Accelerate
import MLX

class Encoder {
    var tokenEmbedding: Embedding
    var positionEmbedding: PositionalEncoding
    var layers: [EncoderLayer]
    var dropout: Dropout
    var scale: Float
    
    init(srcVocabSize: Int, headsNum: Int, layersNum: Int, dModel: Int, dFF: Int, dropoutRate: Float, maxLen: Int = 5000, dataType: DType = DType.float32) {
        
        self.tokenEmbedding = Embedding(inputDim: srcVocabSize, outputDim: dModel, dataType: dataType)
        
        self.positionEmbedding = PositionalEncoding(maxLen: maxLen, dModel: dModel, dropoutRate: dropoutRate, dataType: dataType)
        
        self.layers = []
        for _ in 0..<layersNum {
            self.layers.append(EncoderLayer(dModel: dModel, headsNum: headsNum, dFF: dFF, dropoutRate: dropoutRate, dataType: dataType))
        }
        self.dropout = Dropout(rate: dropoutRate, dataType: dataType)
        self.scale = sqrt(Float(dModel))
        
    }
    
    func forward(src: MLXArray, srcMask: MLXArray, training: Bool) -> MLXArray {
        
        var srcvar = self.tokenEmbedding.forward(X: src) * self.scale
        srcvar = self.positionEmbedding.forward(x: srcvar)
        srcvar = self.dropout.forward(X: srcvar, training: training)
        
        for layer in layers {
            srcvar = layer.forward(src: srcvar, srcMask: srcMask, training: training)
        }
        
        return srcvar
    }
    
    func backward(error: MLXArray) -> MLXArray {
        
        var errorvar = error
        
        for layer in layers.reversed() {
            errorvar = layer.backward(error: error)
        }
        
        errorvar = dropout.backward(errorvar)
        errorvar = positionEmbedding.backward(error: errorvar) * self.scale
        
        return tokenEmbedding.backward(error: errorvar)
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
            layerNum = layer.updateWeights(layerNum: layerNum)
        }
        
    }
}


