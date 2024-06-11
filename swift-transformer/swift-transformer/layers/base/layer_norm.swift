import Foundation
import Accelerate

class LayerNormalization {
    var normalizedShape: [Int]?
    var normalizedAxis: [Int]?
    var epsilon: Float = 0.001
    var gamma: [Float]?
    var beta: [Float]?
    var mean: [Float]?
    var vari: [Float]?
    var optimizer: Optimizer?
    var dataType: [Float]
    var featureSize: Int?
    var inputData: [[Float]]?
    var outputData: [[Float]]?
    var gradGamma: [Float]?
    var gradBeta: [Float]?
    var vg: [Float]?
    var mg: [Float]?
    var vgHat: [Float]?
    var mgHat: [Float]?
    var vb: [Float]?
    var mb: [Float]?
    var vbHat: [Float]?
    var mbHat: [Float]?
    var stddevInv: [Float]?
    var XCentered: [[Float]]?
    var XHatT: [[Float]]?
    var XHat: [[Float]]?

    init(normalizedShape: [Int]? = nil, epsilon: Float = 0.001, dataType: [Float]) {
        self.normalizedShape = normalizedShape
        self.epsilon = epsilon
        self.dataType = dataType
        build()
    }

    func setOptimizer(optimizer: Optimizer) {
        self.optimizer = optimizer
    }

    func build() {
        if let normalizedShape = normalizedShape {
            let size = normalizedShape.reduce(1, *)
            self.gamma = ones((1, size)).flatMap { $0 }
            self.beta = zeros((1, size)).flatMap { $0 }
            self.vg = zerosLike(self.gamma!)
            self.mg = zerosLike(self.gamma!)
            self.vgHat = zerosLike(self.gamma!)
            self.mgHat = zerosLike(self.gamma!)
            self.vb = zerosLike(self.beta!)
            self.mb = zerosLike(self.beta!)
            self.vbHat = zerosLike(self.beta!)
            self.mbHat = zerosLike(self.beta!)
        }
    }

    func forward(_ input: [[Float]]) -> [[Float]] {
        inputData = input
        let rowCount = input.count
        let columnCount = input[0].count

        if normalizedShape == nil {
            normalizedShape = [columnCount]
            build()
        }

        normalizedAxis = Array(0..<max(0, inputData!.count - gamma!.count))
        featureSize = gamma!.count

        let xT = inputData!.transpose()

        mean = xT.map { swift_transformer.mean($0) }
        vari = xT.map { variance($0) }

        XCentered = xT.enumerated().map { index, value in
            value.enumerated().map { j, v in v - mean![j] }
        }

        stddevInv = vari!.map { 1.0 / sqrt($0 + epsilon) }

        XHatT = XCentered!.enumerated().map { index, value in
            value.enumerated().map { j, v in v * stddevInv![j] }
        }

        XHat = XHatT!.transpose()

        outputData = XHat!.enumerated().map { index, row in
            zip(row, gamma!).map { $0 * $1 }
        }.enumerated().map { index, row in
            zip(row, beta!).map { $0 + $1 }
        }

        return outputData!
    }

    func backward(_ error: [[Float]]) -> [[Float]] {
        let rowCount = error.count
        let columnCount = error[0].count
        let errorT = error.transpose()

        let xHatT = XHat!.transpose()

        gradGamma = [Float](repeating: 0.0, count: columnCount)
        gradBeta = [Float](repeating: 0.0, count: columnCount)

        for i in 0..<columnCount {
            gradGamma![i] = dot(errorT[i], xHatT[i])
            gradBeta![i] = sum(errorT[i])
        }

        var outputError = errorT.enumerated().map { (i, row) -> [Float] in
            let gammaValue = gamma![i]
            let stdInv = stddevInv![i]
            let xCenter = XCentered![i]

            return row.enumerated().map { (j, errValue) in
                gammaValue * stdInv * (
                    Float(rowCount) * errValue
                    - gradBeta![i]
                    - xCenter[j] * stdInv * gradGamma![i]
                ) / Float(rowCount)
            }
        }.transpose()

        return outputError
    }

    func updateWeights(layerNum: Int) -> Int {
        guard let optimizer = self.optimizer else {
            fatalError("Optimizer is nil")
        }

        guard let gradGamma = gradGamma, let gradBeta = gradBeta else {
            fatalError("Gradients are nil")
        }
        guard var gamma = gamma, var beta = beta else {
            fatalError("Gamma or beta is nil")
        }
        guard var vg = vg, var mg = mg, var vgHat = vgHat, var mgHat = mgHat else {
            fatalError("Optimized gamma parameters are nil")
        }
        guard var vb = vb, var mb = mb, var vbHat = vbHat, var mbHat = mbHat else {
            fatalError("Optimized beta parameters are nil")
        }

        var tempt = layerNum
        (gamma, vg, mg, vgHat, mgHat, tempt) = optimizer.update(gradient: gradGamma, weights: &gamma, v: &vg, m: &mg, vHat: &vgHat, mHat: &mgHat, t: layerNum)
        (beta, vb, mb, vbHat, mbHat, tempt) = optimizer.update(gradient: gradBeta, weights: &beta, v: &vb, m: &mb, vHat: &vbHat, mHat: &mbHat, t: layerNum)
        self.gamma = gamma
        self.beta = beta
        self.vg = vg
        self.mg = mg
        self.vgHat = vgHat
        self.mgHat = mgHat
        self.vb = vb
        self.mb = mb
        self.vbHat = vbHat
        self.mbHat = mbHat
        return layerNum + 1
    }

    func getGrads() -> ([Float], [Float]) {
        guard let gradGamma = gradGamma, let gradBeta = gradBeta else {
            fatalError("Gradients are nil")
        }
        return (gradGamma, gradBeta)
    }

    func setGrads(_ grads: ([Float], [Float])) {
        gradGamma = grads.0
        gradBeta = grads.1
    }
}
