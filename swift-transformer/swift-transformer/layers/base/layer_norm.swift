import Foundation
import Accelerate

//needed
class LayerNormalization {
    var normalizedShape: [Int]?
    var epsilon: Float
    var gamma: [Float]
    var beta: [Float]
    var mean: [Float]?
    var variance: [Float]?
    var optimizer: Optimizer?
    var dataType: [Float]
    var featureSize: Int?
    var inputData: [[[Float]]]?
    var outputData: [[[Float]]]?
    var gradGamma: [Float]?
    var gradBeta: [Float]?
    var vg, mg, vgHat, mgHat: [Float]
    var vb, mb, vbHat, mbHat: [Float]
    var stddevInv: [Float]?
    var XCentered: [[[Float]]]?
    var XHatT: [[[Float]]]?
    var XHat: [[[Float]]]?

    init(normalizedShape: [Int]? = nil, epsilon: Float = 0.001, dataType: [Float]) {
        self.normalizedShape = normalizedShape
        self.epsilon = epsilon
        self.dataType = dataType
        self.gamma = []
        self.beta = []
        self.vg = []
        self.mg = []
        self.vgHat = []
        self.mgHat = []
        self.vb = []
        self.mb = []
        self.vbHat = []
        self.mbHat = []
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
            self.vg = zerosLike(self.gamma)
            self.mg = zerosLike(self.gamma)
            self.vgHat = zerosLike(self.gamma)
            self.mgHat = zerosLike(self.gamma)
            self.vb = zerosLike(self.beta)
            self.mb = zerosLike(self.beta)
            self.vbHat = zerosLike(self.beta)
            self.mbHat = zerosLike(self.beta)
        }
    }

    func forward(_ input: [[[Float]]]) -> [[[Float]]] {
        inputData = input
        let batchSize = input.count
        let seqLen = input[0].count
        let featureSize = input[0][0].count

        if normalizedShape == nil {
            normalizedShape = [featureSize]
            build()
        }

        mean = Array(repeating: 0, count: featureSize)
        variance = Array(repeating: 0, count: featureSize)

        for b in 0..<batchSize {
            for s in 0..<seqLen {
                for f in 0..<featureSize {
                    mean![f] += input[b][s][f]
                    variance![f] += input[b][s][f] * input[b][s][f]
                }
            }
        }

        let totalCount = Float(batchSize * seqLen)
        for f in 0..<featureSize {
            mean![f] /= totalCount
            variance![f] = variance![f] / totalCount - mean![f] * mean![f]
        }

        stddevInv = variance!.map { 1.0 / sqrt($0 + epsilon) }
        
        XCentered = input.map { batch in
            batch.map { seq in
                zip(seq, mean!).map { $0 - $1 }
            }
        }

        XHat = XCentered!.map { batch in
            batch.map { seq in
                zip(seq, stddevInv!).map { $0 * $1 }
            }
        }

        outputData = XHat!.map { batch in
            batch.map { seq in
                zip(seq, gamma).map { $0 * $1 }
            }.map { seq in
                zip(seq, beta).map { $0 + $1 }
            }
        }

        return outputData!
    }

    func backward(_ error: [[[Float]]]) -> [[[Float]]] {
        let batchSize = error.count
        let seqLen = error[0].count
        let featureSize = error[0][0].count

        gradGamma = Array(repeating: 0, count: featureSize)
        gradBeta = Array(repeating: 0, count: featureSize)

        for b in 0..<batchSize {
            for s in 0..<seqLen {
                for f in 0..<featureSize {
                    gradGamma![f] += error[b][s][f] * XHat![b][s][f]
                    gradBeta![f] += error[b][s][f]
                }
            }
        }

        var outputError = error

        for b in 0..<batchSize {
            for s in 0..<seqLen {
                for f in 0..<featureSize {
                    let gammaValue = gamma[f]
                    let stdInv = stddevInv![f]
                    let xCenter = XCentered![b][s][f]
                    outputError[b][s][f] = gammaValue * stdInv * (
                        Float(batchSize * seqLen) * error[b][s][f]
                        - gradBeta![f]
                        - xCenter * stdInv * gradGamma![f]
                    ) / Float(batchSize * seqLen)
                }
            }
        }

        return outputError
    }

    func updateWeights(layerNum: Int) -> Int {
        guard let optimizer = self.optimizer else {
            fatalError("Optimizer is nil")
        }

        guard let gradGamma = gradGamma, let gradBeta = gradBeta else {
            fatalError("Gradients are nil")
        }

        var layerNum = layerNum
        (gamma, vg, mg, vgHat, mgHat, layerNum) = optimizer.update(gradient: gradGamma, weights: &gamma, v: &vg, m: &mg, vHat: &vgHat, mHat: &mgHat, t: layerNum)
        (beta, vb, mb, vbHat, mbHat, layerNum) = optimizer.update(gradient: gradBeta, weights: &beta, v: &vb, m: &mb, vHat: &vbHat, mHat: &mbHat, t: layerNum)

        return layerNum
    }

    func getGrads() -> ([Float], [Float]) {
        return (gradGamma!, gradBeta!)
    }

    func setGrads(_ grads: ([Float], [Float])) {
        gradGamma = grads.0
        gradBeta = grads.1
    }
}
