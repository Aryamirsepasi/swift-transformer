import Foundation
import Accelerate

class MultiHeadAttention {
    var dModel: Int
    var headsNum: Int
    var scale: Float
    var kLinear, qLinear, vLinear, oLinear: Dense
    var activation: Softmax
    var dropout: Dropout
    var dataType: [Float]

    init(dModel: Int = 512, headsNum: Int = 8, dropoutRate: Float = 0.1, dataType: [Float]) {
        self.dModel = dModel
        self.headsNum = headsNum
        let dK = dModel / headsNum
        self.scale = Float(sqrt(Double(dK)))
        self.dataType = dataType

        self.kLinear = Dense(unitsNum: dK * headsNum, inputsNum: dModel, useBias: false, dataType: dataType)
        self.qLinear = Dense(unitsNum: dK * headsNum, inputsNum: dModel, useBias: false, dataType: dataType)
        self.vLinear = Dense(unitsNum: dK * headsNum, inputsNum: dModel, useBias: false, dataType: dataType)
        self.oLinear = Dense(unitsNum: dModel, inputsNum: dK * headsNum, useBias: true, dataType: dataType)

        self.activation = Softmax()
        self.dropout = Dropout(rate: dropoutRate, dataType: dataType)
    }

    func splitHeadsForward(_ x: [Float], batchCount: Int) -> [Float] {
        let dK = dModel / headsNum
        let batchSize = x.count / (batchCount * dModel)
        var reshaped = Array(repeating: Float(0), count: batchCount * headsNum * batchSize * dK)
        
        for b in 0..<batchCount {
            for i in 0..<batchSize {
                for h in 0..<headsNum {
                    let srcIndex = b * dModel * batchSize + i * dModel + h * dK
                    let dstIndex = b * headsNum * batchSize * dK + h * batchSize * dK + i * dK
                    reshaped[dstIndex..<(dstIndex + dK)] = x[srcIndex..<(srcIndex + dK)]
                }
            }
        }
        
        return reshaped
    }

    func splitHeadsBackward(_ x: [Float], batchCount: Int) -> [Float] {
        return groupHeadsForward(x, batchCount: batchCount)
    }

    func groupHeadsForward(_ x: [Float], batchCount: Int) -> [Float] {
        let dK = dModel / headsNum
        let batchSize = x.count / (batchCount * headsNum * dK)
        var grouped = Array(repeating: Float(0), count: batchCount * dModel * batchSize)
        
        for b in 0..<batchCount {
            for i in 0..<batchSize {
                for h in 0..<headsNum {
                    let srcIndex = b * headsNum * batchSize * dK + h * batchSize * dK + i * dK
                    let dstIndex = b * dModel * batchSize + i * dModel + h * dK
                    grouped[dstIndex..<(dstIndex + dK)] = x[srcIndex..<(srcIndex + dK)]
                }
            }
        }
        
        return grouped
    }

    func groupHeadsBackward(_ x: [Float], batchCount: Int) -> [Float] {
        return splitHeadsForward(x, batchCount: batchCount)
    }

    func forward(query: [Float], key: [Float], value: [Float], mask: [Float]?, training: Bool = true) -> ([Float], [Float]) {
        let batchCount = 1 // Simplification for example
        let dK = dModel / headsNum

        let k = kLinear.forward([key])
        let q = qLinear.forward([query])
        let v = vLinear.forward([value])

        let reshapedK = splitHeadsForward(k.flatMap { $0 }, batchCount: batchCount)
        let reshapedQ = splitHeadsForward(q.flatMap { $0 }, batchCount: batchCount)
        let reshapedV = splitHeadsForward(v.flatMap { $0 }, batchCount: batchCount)

        var energy = [Float](repeating: 0.0, count: reshapedQ.count * reshapedK.count / dModel)
        vDSP_mmul(reshapedQ, 1, reshapedK, 1, &energy, 1, vDSP_Length(headsNum), vDSP_Length(reshapedQ.count / dModel), vDSP_Length(dK))

        var scaledEnergy = [Float](repeating: scale, count: energy.count)
        vDSP_vsdiv(energy, 1, &scaledEnergy, &energy, 1, vDSP_Length(energy.count))

        if let mask = mask {
            for i in 0..<energy.count {
                energy[i] = mask[i] == 0 ? -Float.greatestFiniteMagnitude : energy[i]
            }
        }

        let attention = activation.forward(x: energy)
        let dropoutAttention = dropout.forward(attention, shape: (attention.count, 1), training: training)
        var output = [Float](repeating: 0.0, count: dropoutAttention.count * reshapedV.count / dModel)
        vDSP_mmul(dropoutAttention, 1, reshapedV, 1, &output, 1, vDSP_Length(dropoutAttention.count / dModel), vDSP_Length(reshapedV.count / dModel), vDSP_Length(1))

        let groupedOutput = groupHeadsForward(output, batchCount: batchCount)
        let finalOutput = oLinear.forward([groupedOutput]).flatMap { $0 }

        return (finalOutput, attention)
    }

    func backward(_ error: [Float]) -> ([Float], [Float], [Float]) {
        var adjustedError = oLinear.backward([error]).flatMap { $0 }
        adjustedError = groupHeadsBackward(adjustedError, batchCount: 1)

        var vError = [Float](repeating: 0.0, count: adjustedError.count)
        var qError = [Float](repeating: 0.0, count: adjustedError.count)
        let kError = [Float](repeating: 0.0, count: adjustedError.count)

        // Assuming batch count 1 for simplicity
        vDSP_mtrans(adjustedError, 1, &vError, 1, vDSP_Length(1), vDSP_Length(adjustedError.count))
        vDSP_mtrans(adjustedError, 1, &qError, 1, vDSP_Length(1), vDSP_Length(adjustedError.count))

        let vErrorReshaped = splitHeadsBackward(vError, batchCount: 1)
        let qErrorReshaped = splitHeadsBackward(qError, batchCount: 1)
        let kErrorReshaped = splitHeadsBackward(vError, batchCount: 1) // Placeholder for actual backward pass

        let vBack = vLinear.backward([vErrorReshaped]).flatMap { $0 }
        let qBack = qLinear.backward([qErrorReshaped]).flatMap { $0 }
        let kBack = kLinear.backward([kErrorReshaped]).flatMap { $0 }

        return (qBack, kBack, vBack)
    }

    func setOptimizer(_ optimizer: Optimizer) {
        kLinear.setOptimizer(optimizer: optimizer)
        qLinear.setOptimizer(optimizer: optimizer)
        vLinear.setOptimizer(optimizer: optimizer)
        oLinear.setOptimizer(optimizer: optimizer)
    }

    func updateWeights(_ layerNum: Int) -> Int {
        var layerNum = layerNum
        layerNum = kLinear.updateWeights(layerNum: layerNum)
        layerNum = qLinear.updateWeights(layerNum: layerNum)
        layerNum = vLinear.updateWeights(layerNum: layerNum)
        layerNum = oLinear.updateWeights(layerNum: layerNum)
        return layerNum
    }
}

