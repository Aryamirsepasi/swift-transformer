import Foundation
import Matft
import Metal
import Accelerate

func createZeros(shape: [Int]) -> MfArray {
    let totalSize = shape.reduce(1, *)
    let zeros = Array(repeating: Float(0), count: totalSize)
    return MfArray(zeros).reshape(shape)
}

import Foundation

extension Array where Element == Float {
    static func randomUniform(size: Int, low: Float, high: Float) -> [Float] {
        return (0..<size).map { _ in Float.random(in: low...high) }
    }
}

struct Dense {
    var weights: MfArray
        var biases: MfArray?
        var gradWeights: MfArray?
        var gradBiases: MfArray?
        let useBias: Bool
        var learningRate: Float = 0.01

        init(inputSize: Int, outputSize: Int, useBias: Bool = true, learningRate: Float = 0.01) {
            let stdv = 1.0 / sqrt(Double(inputSize))
            let randomWeights = Array<Float>.randomUniform(size: inputSize * outputSize, low: -Float(stdv), high: Float(stdv))
            self.weights = MfArray(randomWeights).reshape([inputSize, outputSize])
            self.biases = useBias ? Matft.nums(0.0, shape: [outputSize], mftype: .Float) : nil
            self.useBias = useBias
            self.learningRate = learningRate
        }

    func forward(_ input: MfArray) -> MfArray {
        let output = Matft.matmul(input, weights)
        return useBias && biases != nil ? Matft.add(output, biases!) : output
    }

    mutating func backward(input: MfArray, outputError: MfArray) {
        self.gradWeights = Matft.matmul(Matft.transpose(input), outputError)
        if useBias {
            self.gradBiases = Matft.stats.sum(outputError, axis: 0)
        }
        let inputError = Matft.matmul(outputError, Matft.transpose(weights))
        
    }

    mutating func updateWeights() {
        self.weights = Matft.sub(self.weights, Matft.mul(gradWeights!, self.learningRate))
        if useBias && self.gradBiases != nil {
            self.biases = Matft.sub(self.biases!, Matft.mul(gradBiases!, self.learningRate))
        }
    }
}

struct Embedding {
    var embeddings: MfArray

    init(vocabSize: Int, embeddingDim: Int) {
        self.embeddings = Matft.random.rand(shape: [vocabSize, embeddingDim])
    }

    func forward(_ indices: MfArray) -> MfArray {
        return Matft.take(embeddings, indices: indices, axis: 0)
    }
}

struct Dropout {
    var rate: Float

    func forward(_ input: MfArray, training: Bool) -> MfArray {
        guard training else { return input }
        let mask = Matft.random.rand(shape: input.shape) > rate
        return input * mask
    }
}

func variance(_ input: MfArray, axis: Int) -> MfArray {
    let mean = Matft.stats.mean(input, axis: axis)
    let squaredInput = Matft.math.power(bases: input, exponents: 2)
    let meanOfSquares = Matft.stats.mean(squaredInput, axis: axis)
    return meanOfSquares - Matft.math.power(bases: mean, exponents: 2)
}

struct LayerNorm {
    var gamma: MfArray
    var beta: MfArray
    let epsilon: Float

    init(featureSize: Int, epsilon: Float = 0.001) {
        self.gamma = Matft.nums(1.0, shape: [featureSize], mftype: .Float)  // Same as np.ones in Python
        self.beta = Matft.nums(0.0, shape: [featureSize], mftype: .Float)   // Same as np.zeros in Python
        self.epsilon = epsilon
    }

    func forward(_ input: MfArray) -> MfArray {
        let mean = Matft.stats.mean(input, axis: -1)
        let varInput = variance(input, axis: -1)
        let stddev = Matft.math.sqrt(varInput + epsilon)
        let normed = (input - mean) / stddev
        return gamma * normed + beta
    }
}

struct PositionwiseFeedforward {
    var fc1: Dense
    var fc2: Dense
    var dropout: Dropout

    init(dModel: Int, dFF: Int, dropoutRate: Float) {
        self.fc1 = Dense(inputSize: dModel, outputSize: dFF)
        self.fc2 = Dense(inputSize: dFF, outputSize: dModel)
        self.dropout = Dropout(rate: dropoutRate)
    }

    func forward(_ input: MfArray, training: Bool) -> MfArray {
        var x = fc1.forward(input)
        x = relu(x)
        x = dropout.forward(x, training: training)
        return fc2.forward(x)
    }
}


struct MultiHeadAttention {
    var dModel: Int
    var headsNum: Int
    var scale: Float
    var kLinear: Dense
    var qLinear: Dense
    var vLinear: Dense
    var oLinear: Dense
    var dropout: Dropout

    init(dModel: Int, headsNum: Int, dropoutRate: Float) {
        self.dModel = dModel
        self.headsNum = headsNum
        self.scale = sqrt(Float(dModel) / Float(headsNum))
        self.kLinear = Dense(inputSize: dModel, outputSize: dModel)
        self.qLinear = Dense(inputSize: dModel, outputSize: dModel)
        self.vLinear = Dense(inputSize: dModel, outputSize: dModel)
        self.oLinear = Dense(inputSize: dModel, outputSize: dModel)
        self.dropout = Dropout(rate: dropoutRate)
    }

    func forward(_ query: MfArray, key: MfArray, value: MfArray, mask: MfArray, training: Bool) -> (MfArray, MfArray) {
        let K = kLinear.forward(key)
        let Q = qLinear.forward(query)
        let V = vLinear.forward(value)

        let batch_size = query.shape[0]
        let Q_split = Q.reshape([batch_size, -1, headsNum, dModel / headsNum]).transpose(axes: [0, 2, 1, 3]) * scale
        let K_split = K.reshape([batch_size, -1, headsNum, dModel / headsNum]).transpose(axes: [0, 2, 1, 3])
        let V_split = V.reshape([batch_size, -1, headsNum, dModel / headsNum]).transpose(axes: [0, 2, 1, 3])

        var energy = Matft.matmul(Q_split, K_split.transpose(axes: [0, 1, 3, 2]))

        // Apply the mask using broadcast_to
        if !mask.isEmpty {
            let expandedMask = Matft.broadcast_to(mask, shape: energy.shape)
            let negInf = Matft.nums(Float.greatestFiniteMagnitude * -1, shape: energy.shape, mftype: .Float)
            energy = Matft.add(energy, Matft.mul(negInf, Matft.sub(1, expandedMask)))
        }

        let attention = softmax(energy, axis: -1)
        let attentionDropped = dropout.forward(attention, training: training)

        let output = Matft.matmul(attentionDropped, V_split)
        let concatOutput = output.transpose(axes: [0, 2, 1, 3]).reshape([batch_size, -1, headsNum * (dModel / headsNum)])

        return (oLinear.forward(concatOutput), attention)
    }

    // Helper function for softmax
    func softmax(_ input: MfArray, axis: Int) -> MfArray {
        let expInput = Matft.math.exp(input - Matft.stats.max(input, axis: axis, keepDims: true))
        return expInput / Matft.stats.sum(expInput, axis: axis, keepDims: true)
    }
}




struct DecoderLayer {
    var selfAttention: MultiHeadAttention
    var feedForward: PositionwiseFeedforward
    var normLayer: LayerNorm

    init(dModel: Int, heads: Int, dFF: Int, dropout: Float) {
        self.selfAttention = MultiHeadAttention(dModel: dModel, headsNum: heads, dropoutRate: dropout)
        self.feedForward = PositionwiseFeedforward(dModel: dModel, dFF: dFF, dropoutRate: dropout)
        self.normLayer = LayerNorm(featureSize: dModel)
    }

    func forward(_ trg: MfArray, trgMask: MfArray, src: MfArray, srcMask: MfArray, training: Bool) -> (MfArray, MfArray) {
        // Include the training parameter in the forward method call
        let (attnOutput, _) = selfAttention.forward(trg, key: trg, value: trg, mask: trgMask, training: training)
        let output = feedForward.forward(attnOutput, training: training) // Make sure the feedForward forward also accepts a training bool
        return (normLayer.forward(output), attnOutput)
    }
}

struct PositionalEncoding {
    var pe: MfArray

        init(maxLen: Int, dModel: Int) {
            let positions = Matft.arange(start: 0, to: maxLen, by: 1).reshape([maxLen, 1])
            // Generate the divisor term as a Swift array and then convert it back to MfArray if necessary
            let expArray = (0..<dModel/2).map { exp(-Double($0 * 2) / Double(dModel) * log(10000.0)) }
            let divTerm = MfArray(expArray)

            let peSin = Matft.math.sin(Matft.broadcast_to(positions * divTerm, shape: [maxLen, dModel]))
            let peCos = Matft.math.cos(Matft.broadcast_to(positions * divTerm, shape: [maxLen, dModel]))

            self.pe = Matft.vstack([peSin, peCos])
        }

    func forward(_ x: MfArray) -> MfArray {
        // Assuming x.shape[0] is the dimension size for the first and third dimensions
        return x + pe[0..<x.shape[0], 0..<x.shape[1], 0..<x.shape[2]]

    }
}

// Helper functions
func relu(_ input: MfArray) -> MfArray {
    return Matft.stats.max(input, axis: 0)
}

struct Decoder {
    var tokenEmbedding: Embedding
    var positionEmbedding: PositionalEncoding
    var layers: [DecoderLayer]
    var fcOut: Dense
    var dropout: Dropout
    let scale: Float

    init(trgVocabSize: Int, headsNum: Int, layersNum: Int, dModel: Int, dFF: Int, dropout: Float, maxLen: Int = 5000) {
        self.tokenEmbedding = Embedding(vocabSize: trgVocabSize, embeddingDim: dModel)
        self.positionEmbedding = PositionalEncoding(maxLen: maxLen, dModel: dModel)
        self.scale = sqrt(Float(dModel))

        self.layers = []
        for _ in 0..<layersNum {
            self.layers.append(DecoderLayer(dModel: dModel, heads: headsNum, dFF: dFF, dropout: dropout))
        }

        self.fcOut = Dense(inputSize: dModel, outputSize: trgVocabSize)
        self.dropout = Dropout(rate: dropout)
    }

    func forward(trg: MfArray, trgMask: MfArray, src: MfArray, srcMask: MfArray, training: Bool) -> (MfArray, MfArray) {
            var trgEmbed = tokenEmbedding.forward(trg)
            trgEmbed = trgEmbed * scale
            trgEmbed = positionEmbedding.forward(trgEmbed)
            trgEmbed = dropout.forward(trgEmbed, training: training)

            var attention: MfArray?
            for layer in layers {
                (trgEmbed, attention) = layer.forward(trgEmbed, trgMask: trgMask, src: src, srcMask: srcMask, training: training)
            }

            let output = fcOut.forward(trgEmbed)
            return (output, attention!)
        }
}
