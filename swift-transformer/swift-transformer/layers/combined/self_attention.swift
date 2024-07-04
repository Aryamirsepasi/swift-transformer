import Foundation
import Accelerate

class MultiHeadAttention {
    var dModel: Int
    var headsNum: Int
    var dataType: [Float]
    
    var dK: Int
    var dQ: Int
    var dV: Int
    var scale: Float
    
    var KLinear: Dense
    var QLinear: Dense
    var VLinear: Dense
    var OLinear: Dense
    
    var activation: Softmax
    var dropout: Dropout
    
    var splitK: [[[Float]]] = []
    var splitQ: [[[Float]]] = []
    var splitV: [[[Float]]] = []
    var dropoutAttention: [[[Float]]] = []
    var maskArray: [[[Float]]] = []
    
    init(dModel: Int = 512, headsNum: Int = 8, dropoutRate: Float = 0.1, dataType: [Float] = []) {
        self.dModel = dModel
        self.headsNum = headsNum
        self.dataType = dataType
        
        self.dK = dModel / headsNum
        self.dQ = dModel / headsNum
        self.dV = dModel / headsNum
        self.scale = sqrt(Float(dK))
        
        self.KLinear = Dense(unitsNum: dK * headsNum, inputsNum: dModel, useBias: false, dataType: dataType)
        self.QLinear = Dense(unitsNum: dQ * headsNum, inputsNum: dModel, useBias: false, dataType: dataType)
        self.VLinear = Dense(unitsNum: dV * headsNum, inputsNum: dModel, useBias: false, dataType: dataType)
        self.OLinear = Dense(unitsNum: dV * headsNum, inputsNum: dModel, useBias: true, dataType: dataType)
        
        self.activation = Softmax()
        self.dropout = Dropout(rate: dropoutRate, dataType: dataType)
    }
    
    func splitHeadsForward(_ x: [[[Float]]]) -> [[[Float]]] {
        let batchSize = x.count
        let seqLen = x[0].count
        let reshaped = x.flatMap { $0.flatMap { $0 } }
        
        var result = [[[Float]]](repeating: [[Float]](repeating: [Float](repeating: 0.0, count: dK), count: seqLen), count: batchSize * headsNum)
        
        for b in 0..<batchSize {
            for h in 0..<headsNum {
                for s in 0..<seqLen {
                    for d in 0..<dK {
                        result[b * headsNum + h][s][d] = reshaped[b * (seqLen * dModel) + s * dModel + h * dK + d]
                    }
                }
            }
        }
        
        return result
    }
    
    func splitHeadsBackward(_ x: [[[Float]]]) -> [[[Float]]] {
        let batchSize = x.count / headsNum
        let seqLen = x[0].count
        let reshaped = x.flatMap { $0.flatMap { $0 } }
        
        var result = [[[Float]]](repeating: [[Float]](repeating: [Float](repeating: 0.0, count: dModel), count: seqLen), count: batchSize)
        
        for b in 0..<batchSize {
            for h in 0..<headsNum {
                for s in 0..<seqLen {
                    for d in 0..<dK {
                        result[b][s][h * dK + d] = reshaped[b * (headsNum * seqLen * dK) + h * (seqLen * dK) + s * dK + d]
                    }
                }
            }
        }
        
        return result
    }
    
    func groupHeadsForward(_ x: [[[Float]]]) -> [[[Float]]] {
        let batchSize = x.count / headsNum
        let seqLen = x[0].count
        let reshaped = x.flatMap { $0.flatMap { $0 } }
        
        var result = [[[Float]]](repeating: [[Float]](repeating: [Float](repeating: 0.0, count: dModel), count: seqLen), count: batchSize)
        
        for b in 0..<batchSize {
            for h in 0..<headsNum {
                for s in 0..<seqLen {
                    for d in 0..<dK {
                        result[b][s][h * dK + d] = reshaped[b * (headsNum * seqLen * dK) + h * (seqLen * dK) + s * dK + d]
                    }
                }
            }
        }
        
        return result
    }
    
    func groupHeadsBackward(_ x: [[[Float]]]) -> [[[Float]]] {
        let batchSize = x.count
        let seqLen = x[0].count
        let reshaped = x.flatMap { $0.flatMap { $0 } }
        
        var result = [[[Float]]](repeating: [[Float]](repeating: [Float](repeating: 0.0, count: dK), count: seqLen), count: batchSize * headsNum)
        
        for b in 0..<batchSize {
            for h in 0..<headsNum {
                for s in 0..<seqLen {
                    for d in 0..<dK {
                        result[b * headsNum + h][s][d] = reshaped[b * (seqLen * dModel) + s * dModel + h * dK + d]
                    }
                }
            }
        }
        
        return result
    }
    
    func forward(query: [[[Float]]], key: [[[Float]]], value: [[[Float]]], mask: [[[Float]]], training: Bool = true) -> ([[[Float]]], [[[Float]]]) {
        let K = KLinear.forward(key, training: training)
        let Q = QLinear.forward(query, training: training)
        let V = VLinear.forward(value, training: training)
        
        splitK = splitHeadsForward(K)
        splitQ = splitHeadsForward(Q)
        splitV = splitHeadsForward(V)
        
        var energy = [[[Float]]](repeating: [[Float]](repeating: [Float](repeating: 0.0, count: key[0].count), count: query[0].count), count: query.count)
        let transposedSplitK = splitK.transposed()
        
        for i in 0..<splitQ.count {
            for j in 0..<transposedSplitK.count {
                for k in 0..<splitQ[i].count {
                    energy[i][j][k] = zip(splitQ[i][k], transposedSplitK[j][k]).map(*).reduce(0, +) / scale
                }
            }
        }
        
        maskArray = mask
        if !maskArray.isEmpty {
            let maskArrayFloat = maskArray.map { $0.map { $0.map { $0 == 0.0 ? -Float.greatestFiniteMagnitude : 0.0 } } }
            for i in 0..<energy.count {
                for j in 0..<energy[i].count {
                    for k in 0..<energy[i][j].count {
                        energy[i][j][k] += maskArrayFloat[i][j][k]
                    }
                }
            }
        }
        
        let flattenedEnergy = energy.flatMap { $0.flatMap { $0 } }
        let attention = activation.forward(x: reshape(flattenedEnergy, newShape: shape(energy)) as! [[[Float]]])
        
        let dropoutAttention2D = dropout.forward(attention, training: training)
        let dropoutAttentionFlat = dropoutAttention2D.flatMap { $0 }
        dropoutAttention = reshape(dropoutAttentionFlat, newShape: [dropoutAttention2D.count, dropoutAttention2D[0].count, headsNum, dK]) as! [[[Float]]]
        
        var output = [[[Float]]]()
        for row in dropoutAttention {
            var outputRow = [[Float]]()
            for col in splitV {
                let dotProduct = zip(row, col).map { zip($0, $1).map(*).reduce(0, +) }
                outputRow.append(dotProduct)
            }
            output.append(outputRow)
        }
        
        let concatOutput = groupHeadsForward(output)
        let O = OLinear.forward(concatOutput, training: training)
        
        return (O, attention)
    }
    
    func backward(error: [[[Float]]]) -> (error: [[[Float]]], encError1: [[[Float]]], encError2: [[[Float]]]) {
        var error = OLinear.backward(error)
        
        error = groupHeadsBackward(error).flatMap { $0 }
        
        var VError = [[[Float]]]()
        
        var transposedSplitV = splitV.transposed()
        
        error = dropout.backward(error)
        error = activation.backward(grad: error.flatMap { $0 })
        
        var QError = [[[Float]]]()
        
        var KError = [[[Float]]]()
        
        let VErrorFinal = splitHeadsBackward(VError.flatMap { $0 })
        let QErrorFinal = splitHeadsBackward(QError.flatMap { $0 })
        let KErrorFinal = splitHeadsBackward(KError.flatMap { $0 })
        
        let VErrorOutput = VLinear.backward(VErrorFinal)
        let QErrorOutput = QLinear.backward(QErrorFinal)
        let KErrorOutput = KLinear.backward(KErrorFinal)
        
        return (QErrorOutput, VErrorOutput, KErrorOutput)
    }
    
    func setOptimizer(optimizer: Optimizer) {
        KLinear.setOptimizer(optimizer: optimizer)
        QLinear.setOptimizer(optimizer: optimizer)
        VLinear.setOptimizer(optimizer: optimizer)
        OLinear.setOptimizer(optimizer: optimizer)
    }
    
    func updateWeights(layerNum: Int) -> Int {
        var layerNum = KLinear.updateWeights(layerNum: layerNum)
        layerNum = QLinear.updateWeights(layerNum: layerNum)
        layerNum = VLinear.updateWeights(layerNum: layerNum)
        layerNum = OLinear.updateWeights(layerNum: layerNum)
        return layerNum
    }
}

// Extensions and helper functions

extension Array where Element == [Float] {
    func transposed() -> [[Float]] {
        var result = [[Float]](repeating: [Float](repeating: 0.0, count: self.count), count: self[0].count)
        for i in 0..<self.count {
            for j in 0..<self[i].count {
                result[j][i] = self[i][j]
            }
        }
        return result
    }
}

extension Array where Element == [[Float]] {
    func transposed() -> [[[Float]]] {
        let outerCount = self.count
        let innerCount = self[0].count
        let innerMostCount = self[0][0].count
        
        var result = [[[Float]]](repeating: [[Float]](repeating: [Float](repeating: 0.0, count: outerCount), count: innerMostCount), count: innerCount)
        
        for i in 0..<outerCount {
            for j in 0..<innerCount {
                for k in 0..<innerMostCount {
                    result[j][k][i] = self[i][j][k]
                }
            }
        }
        
        return result
    }
}

extension Array where Element == [[[Float]]] {
    func transposed() -> [[[[Float]]]] {
        let outerCount = self.count
        guard let firstTensor = self.first else { return [] }
        let matrixCount = firstTensor.count
        guard let firstMatrix = firstTensor.first else { return [] }
        let rowCount = firstMatrix.count
        guard let firstRow = firstMatrix.first else { return [] }
        let columnCount = firstRow.count
        
        var result = [[[[Float]]]](repeating: [[[Float]]](repeating: [[Float]](repeating: [Float](repeating: 0.0, count: outerCount), count: columnCount), count: rowCount), count: matrixCount)
        
        for i in 0..<outerCount {
            for j in 0..<matrixCount {
                for k in 0..<rowCount {
                    for l in 0..<columnCount {
                        result[j][k][l][i] = self[i][j][k][l]
                    }
                }
            }
        }
        
        return result
    }
}

func convert(_ array: [Float], to shape: [Int]) -> [[[Float]]] {
    var reshapedArray: [[[Float]]] = []
    var start = 0
    for count in shape {
        var subArray: [[Float]] = []
        for _ in 0..<count {
            let end = start + count
            subArray.append(Array(array[start..<end]))
            start = end
        }
        reshapedArray.append(subArray)
    }
    return reshapedArray
}
