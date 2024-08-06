import Foundation
import Accelerate

class Embedding {
    var inputDim: Int
    var outputDim: Int
    var weights: [Float]
    var optimizer: Optimizer?
    var v: [Float]
    var m: [Float]
    var vHat: [Float]
    var mHat: [Float]
    var inputLabels: [[[Float]]]? 
    var gradWeights: [Float]?

    init(inputDim: Int, outputDim: Int, dataType: [Float]) {
        self.inputDim = inputDim
        self.outputDim = outputDim
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
    
    private func prepareLabels(batchLabels: [[Float]]) -> [[[Float]]] {
        let batchCount = batchLabels.count
        let inputLength = batchLabels[0].count
        
        var preparedBatchLabels = [[[Float]]]()
        
        for batch in batchLabels {
            var batchPrepared = [[Float]](repeating: [Float](repeating: 0.0, count: inputDim), count: inputLength)
            for (index, label) in batch.enumerated() {
                let intLabel = Int(label)
                guard intLabel >= 0 && intLabel < inputDim else {
                    fatalError("Label index \(intLabel) out of bounds for inputDim \(inputDim)")
                }
                batchPrepared[index][intLabel] = 1.0
            }
            preparedBatchLabels.append(batchPrepared)
        }
        
        return preparedBatchLabels
    }

    func forward(input: [[Float]]) -> [[[Float]]] {
        guard !input.isEmpty else { return [] }
        
        // Ensure all input sequences are of the same length
        let inputLength = input[0].count
        for array in input {
            if array.count != inputLength {
                fatalError("Input sequences must be of the same length")
            }
        }

        inputLabels = prepareLabels(batchLabels: input)
        
        var output = [[[Float]]](repeating: [[Float]](repeating: [Float](repeating: 0.0, count: outputDim), count: inputLength), count: input.count)
        
        for (i, batch) in inputLabels!.enumerated() {
            for (j, inputVector) in batch.enumerated() {
                var result = [Float](repeating: 0.0, count: outputDim)
                vDSP_mmul(inputVector, 1, weights, 1, &result, 1, 1, vDSP_Length(outputDim), vDSP_Length(inputDim))
                output[i][j] = result
            }
        }
        
        return output
    }
    
    func backward(error: [[[Float]]]) -> [[[Float]]]? {
        guard let inputLabels = inputLabels else { return nil }
        
        let batchCount = inputLabels.count
        let inputLength = inputLabels[0].count
        gradWeights = [Float](repeating: 0.0, count: weights.count)
        
        for i in 0..<batchCount {
            for j in 0..<inputLength {
                var tempGradWeights = [Float](repeating: 0.0, count: weights.count)
                let inputVector = inputLabels[i][j]
                let errorVector = error[i][j]
                vDSP_mmul(inputVector, 1, errorVector, 1, &tempGradWeights, 1, vDSP_Length(inputDim), vDSP_Length(outputDim), 1)
                vDSP_vadd(gradWeights!, 1, tempGradWeights, 1, &gradWeights!, 1, vDSP_Length(weights.count))
            }
        }
        
        return error
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
