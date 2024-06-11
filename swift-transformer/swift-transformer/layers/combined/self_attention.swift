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
    var maskArray: [[Bool]] = []
    
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
    
    func splitHeadsForward(_ x: [[Float]]) -> [[[Float]]] {
        let batchSize = x.count
        let reshaped = reshape(x.flatMap { $0 }, newShape: [batchSize, -1, headsNum, dK]) as! [[[Float]]]
        return reshaped.transposed()
    }
    
    func splitHeadsBackward(_ x: [[[Float]]]) -> [[Float]] {
        let batchSize = x.count
        let transposed = x.transposed()
        return reshape(transposed.flatMap { $0 }, newShape: [batchSize, -1, headsNum * dK]) as! [[Float]]
    }
    
    func groupHeadsForward(_ x: [[[Float]]]) -> [[Float]] {
        let batchSize = x.count
        let transposed = x.transposed()
        return reshape(transposed.flatMap { $0 }, newShape: [batchSize, -1, headsNum * dK]) as! [[Float]]
    }
    
    func groupHeadsBackward(_ x: [[Float]]) -> [[[Float]]] {
        let batchSize = x.count
        let reshaped = reshape(x.flatMap { $0 }, newShape: [batchSize, -1, headsNum, dK]) as! [[[Float]]]
        return reshaped.transposed()
    }
    
    func forward(query: [[Float]], key: [[Float]], value: [[Float]], mask: [[Bool]], training: Bool = true) -> ([[Float]], [[Float]]) {
        let keyLen = key[0].count
        let queryLen = query[0].count
        let valueLen = value[0].count
        
        let K = KLinear.forward(key, training: training)
        let Q = QLinear.forward(query, training: training)
        let V = VLinear.forward(value, training: training)
        
        splitK = splitHeadsForward(K)
        splitQ = splitHeadsForward(Q)
        splitV = splitHeadsForward(V)
        
        var energy = splitQ.flatMap { $0 }.map { row in
            splitK.transposed().map { col in
                zip(row, col).map(*).reduce(0, +) / scale
            }
        }
        
        maskArray = mask
        if !maskArray.isEmpty {
            let maskArrayFloat = maskArray.map { $0.map { $0 ? 0.0 : -Float.greatestFiniteMagnitude } }
            energy = zip(energy, maskArrayFloat).map { zip($0, $1).map { $0 + $1 } }
        }
        
        let attention = activation.forward(x: energy.flatMap { $0 })
        let attention2D = convert(attention, to: energy.map { $0.count })
        
        let dropoutAttention2D = dropout.forward(attention2D, training: training)
        dropoutAttention = reshape(dropoutAttention2D.flatMap { $0 }, newShape: [dropoutAttention2D.count, -1, headsNum, dK]) as! [[[Float]]]
        
        let output = dropoutAttention.flatMap { $0 }.map { row in
            splitV.flatMap { $0 }.map { col in
                zip(row, col).map(*).reduce(0, +)
            }
        }
        
        let concatOutput = groupHeadsForward(output)
        let O = OLinear.forward(concatOutput, training: training)
        
        return (O, attention2D)
    }
    
    func backward(error: [[Float]]) -> ([[Float]], [[Float]], [[Float]]) {
        var error = OLinear.backward(error)
        
        error = groupHeadsBackward(error).flatMap { $0 }
        let VError = dropoutAttention.transposed().flatMap { $0 }.map { row in
            error.map { col in
                zip(row, col).map(*).reduce(0, +)
            }
        }
        error = error.flatMap { $0 }.map { row in
            splitV.transposed().map { col in
                zip(row, col).map(*).reduce(0, +)
            }
        }
        error = dropout.backward(error)
        error = activation.backward(grad: error.flatMap { $0 })
        
        if !maskArray.isEmpty {
            let maskArrayFlat = maskArray.flatMap { $0 }
            error = zip(error.flatMap { $0 }, maskArrayFlat).map { $1 == false ? 0.0 : $0 }
        }
        
        error = error.map { $0 / scale }
        
        let QError = error.flatMap { $0 }.map { row in
            splitK.flatMap { $0 }.map { col in
                zip(row, col).map(*).reduce(0, +)
            }
        }
        var KError = splitQ.transposed().flatMap { $0 }.map { row in
            error.map { col in
                zip(row, col).map(*).reduce(0, +)
            }
        }
        KError = KError.transposed()
        
        let VErrorFinal = splitHeadsBackward(VError)
        let QErrorFinal = splitHeadsBackward(QError)
        let KErrorFinal = splitHeadsBackward(KError)
        
        let VErrorOutput = VLinear.backward(VErrorFinal)
        let QErrorOutput = QLinear.backward(QErrorFinal)
        let KErrorOutput = KLinear.backward(KErrorFinal)
        
        return (QErrorOutput, KErrorOutput, VErrorOutput)
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

func convert(_ array: [Float], to shape: [Int]) -> [[Float]] {
    var reshapedArray: [[Float]] = []
    var start = 0
    for count in shape {
        let end = start + count
        reshapedArray.append(Array(array[start..<end]))
        start = end
    }
    return reshapedArray
}
