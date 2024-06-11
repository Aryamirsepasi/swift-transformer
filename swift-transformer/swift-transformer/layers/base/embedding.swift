import Foundation
import Accelerate

class Embedding {
    var inputDim: Int
    var outputDim: Int
    var weights: [Float]
    var optimizer: Optimizer?
    var dataType: [Float]
    
    var v: [Float]
    var m: [Float]
    var vHat: [Float]
    var mHat: [Float]
    
    var inputLabels: [[Float]]?
    var gradWeights: [Float]?

    init(inputDim: Int, outputDim: Int, dataType: [Float]) {
        self.inputDim = inputDim
        self.outputDim = outputDim
        self.dataType = dataType
        self.weights = [Float](repeating: 0.0, count: inputDim * outputDim)
        self.v = [Float](repeating: 0.0, count: inputDim * outputDim)
        self.m = [Float](repeating: 0.0, count: inputDim * outputDim)
        self.vHat = [Float](repeating: 0.0, count: inputDim * outputDim)
        self.mHat = [Float](repeating: 0.0, count: inputDim * outputDim)
        
        build()
    }
    
    func setOptimizer(optimizer: Optimizer) {
        self.optimizer = optimizer
    }
    
    private func build() {
        let scale = sqrt(1.0 / Float(inputDim))
        let distribution = NormalDistribution(mean: 0, standardDeviation: scale)
        weights = (0..<weights.count).map { _ in distribution.next() }
    }
    
    private func prepareLabels(batchLabels: [Int]) -> [[Float]] {
        let batchCount = batchLabels.count
        var preparedBatchLabels = [Float](repeating: 0.0, count: batchCount * inputDim)
        
        for (index, label) in batchLabels.enumerated() {
            guard label >= 0 && label < inputDim else {
                fatalError("Label index \(label) out of bounds for inputDim \(inputDim)")
            }
            preparedBatchLabels[index * inputDim + label] = 1
        }
        
        return stride(from: 0, to: preparedBatchLabels.count, by: inputDim).map {
            Array(preparedBatchLabels[$0..<$0 + inputDim])
        }
    }

    func forward(input: [Int]) -> [[Float]] {
        guard !input.isEmpty else { return [] }
        
        inputLabels = prepareLabels(batchLabels: input)
        var output = [[Float]](repeating: [Float](repeating: 0.0, count: outputDim), count: input.count)
        
        for (i, inputVector) in inputLabels!.enumerated() {
            var result = [Float](repeating: 0.0, count: outputDim)
            vDSP_mmul(inputVector, 1, weights, 1, &result, 1, 1, vDSP_Length(outputDim), vDSP_Length(inputDim))
            output[i] = result
        }
        
        return output
    }
    
    func backward(error: [[Float]]) -> [Float]? {
        guard let inputLabels = inputLabels else { return nil }
        
        let batchCount = error.count
        gradWeights = [Float](repeating: 0.0, count: weights.count)
        var errorFlatten = error.flatMap { $0 }
        
        for i in 0..<batchCount {
            var tempGradWeights = [Float](repeating: 0.0, count: weights.count)
            let inputVector = inputLabels[i]
            let errorVector = Array(errorFlatten[(i * outputDim)..<(i * outputDim + outputDim)])
            vDSP_mmul(inputVector, 1, errorVector, 1, &tempGradWeights, 1, vDSP_Length(inputDim), vDSP_Length(outputDim), 1)
            vDSP_vadd(gradWeights!, 1, tempGradWeights, 1, &gradWeights!, 1, vDSP_Length(weights.count))
        }
        
        return error.flatMap { $0 }
    }

    func updateWeights(layerNum: Int) -> Int {
        if let optimizer = optimizer, let gradWeights = gradWeights {
            var templayerNum = layerNum
            (weights, v, m, vHat, mHat, templayerNum) = optimizer.update(gradient: gradWeights, weights: &weights, v: &v, m: &m, vHat: &vHat, mHat: &mHat, t: layerNum)
        }
        return layerNum + 1
    }
    
    func getGrads() -> [Float]? {
        return gradWeights
    }

    func setGrads(_ grads: [Float]) {
        gradWeights = grads
    }
}

// Helper function to generate normal distribution
struct NormalDistribution {
    var mean: Float
    var standardDeviation: Float
    
    func next() -> Float {
        let u1 = Float.random(in: 0..<1)
        let u2 = Float.random(in: 0..<1)
        let r = sqrt(-2 * log(u1))
        let theta = 2 * Float.pi * u2
        return r * sin(theta) * standardDeviation + mean
    }
}
