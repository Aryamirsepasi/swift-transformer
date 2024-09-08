import Foundation
import Accelerate
import MLX
//needed
class Decoder {
    var tokenEmbedding: Embedding
    var positionEmbedding: PositionalEncoding
    var layers: [DecoderLayer]
    var fcOut: Dense
    var dropout: Dropout
    var scale: Float
    var activation: Identity
    var encoderError: MLXArray

    init(trgVocabSize: Int, headsNum: Int, layersNum: Int, dModel: Int, dFF: Int, dropoutRate: Float, maxLen: Int = 5000, dataType: DType = DType.float32) {
        
        self.tokenEmbedding = Embedding(inputDim: trgVocabSize, outputDim: dModel, dataType: dataType)
        self.positionEmbedding = PositionalEncoding(maxLen: maxLen, dModel: dModel, dropoutRate: dropoutRate, dataType: dataType)
        
        self.layers = []
        
        for _ in 0..<layersNum {
            self.layers.append(DecoderLayer(dModel: dModel, headsNum: headsNum, dFF: dFF, dropoutRate: dropoutRate, dataType: dataType))
        }
        self.fcOut = Dense(unitsNum: trgVocabSize, inputsNum: dModel, useBias: true, dataType: dataType)
        self.dropout = Dropout(rate: dropoutRate, dataType: dataType)
        self.scale = sqrt(Float(dModel))
        
        self.encoderError = []
        
        self.activation = Identity()
        
    }

    func forward(trg: MLXArray, trgMask: MLXArray, src: MLXArray, srcMask: MLXArray, training: Bool) -> (MLXArray, MLXArray) {
        
        var trgvar = trg
        
        trgvar = self.tokenEmbedding.forward(X: trg) * self.scale
        trgvar = self.positionEmbedding.forward(x: trgvar)
        trgvar = self.dropout.forward(X: trgvar, training: training)
        var attention : MLXArray = []
        
        for layer in self.layers{
            (trgvar, attention) = layer.forward(trg: trgvar, trgMask: trgMask, src: src, srcMask: srcMask, training: training)
        }
        
        var output = self.fcOut.forward(X: trgvar)
        
        var activatedOutput = self.activation.forward(x: output)
        
        return (activatedOutput, attention)
    }

    func backward(error: MLXArray) -> MLXArray {
        
        print("entered decoder backward")

        var errorvar = activation.backward(grad: error)
        errorvar = fcOut.backward(error: errorvar)

        self.encoderError = zeros(errorvar.shape)
        
        for layer in self.layers.reversed() {
            let (layerError, encError) = layer.backward(error: errorvar)
            self.encoderError += encError
        }

        errorvar = self.dropout.backward(errorvar)
        
        errorvar = positionEmbedding.backward(error: errorvar) * self.scale
        errorvar = tokenEmbedding.backward(error: errorvar)
        
        print("exited decoder backward")

        return errorvar
    }

    func setOptimizer(optimizer: Optimizer) {
        tokenEmbedding.setOptimizer(optimizer: optimizer)
        for layer in layers {
            layer.setOptimizer(optimizer)
        }
        fcOut.setOptimizer(optimizer: optimizer)
    }

    func updateWeights() {
        var layerNum = 1
        layerNum = tokenEmbedding.updateWeights(layerNum: layerNum)
        for layer in self.layers {
            layerNum = layer.updateWeights(layerNum)
        }
        self.fcOut.updateWeights(layerNum: layerNum)
    }
}
