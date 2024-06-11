import Foundation
import Accelerate

class PositionwiseFeedforward {
    var fc1: Dense
    var activation: Activation
    var dropout: Dropout
    var fc2: Dense

    init(dModel: Int = 512, dFF: Int = 2048, dropoutRate: Float = 0.1) {
        self.fc1 = Dense(unitsNum: dFF, inputsNum: dModel, useBias: true)
        self.activation = ReLU()
        self.fc2 = Dense(unitsNum: dModel, inputsNum: dFF, useBias: true)
        self.dropout = Dropout(rate: dropoutRate)
    }

    func forward(_ input: [[Float]], training: Bool = true) -> [[Float]] {
        var x = fc1.forward(input, training: training)
        x = activation.forward(x: x.flatMap { $0 }).chunked(into: fc1.unitsNum)
        x = dropout.forward(x.flatMap { $0 }, shape: (x.count, x[0].count), training: training).chunked(into: fc1.unitsNum)
        x = fc2.forward(x, training: training)
        return x
    }

    func backward(_ error: [[Float]]) -> [[Float]] {
        var err = fc2.backward(error)
        err = dropout.backward(err.flatMap { $0 }).chunked(into: fc1.unitsNum)
        err = activation.backward(grad: err.flatMap { $0 }).chunked(into: fc1.inputsNum!)
        err = fc1.backward(err)
        return err
    }

    func setOptimizer(optimizer: Optimizer) {
        fc1.setOptimizer(optimizer: optimizer)
        fc2.setOptimizer(optimizer: optimizer)
    }

    func updateWeights(startingLayerNum: Int) -> Int {
        var layerNum = startingLayerNum
        layerNum = fc1.updateWeights(layerNum: layerNum)
        layerNum = fc2.updateWeights(layerNum: layerNum)
        return layerNum
    }
}

extension Array {
    func chunked(into size: Int) -> [[Element]] {
        stride(from: 0, to: count, by: size).map {
            Array(self[$0..<Swift.min($0 + size, count)])
        }
    }
}
