import Accelerate
import MLX
import MLXRandom
class Reshape {
    var shape: [Int]
    var inputShape: [Int]
    var prevShape: [Int]
    var outputShape: [Int]

    init(shape: [Int]) {
        self.shape = shape
        self.inputShape = []
        self.outputShape = []
        self.prevShape = []
    }

    func build() {
        self.outputShape = self.shape
    }

    func forwardProp(X: MLXArray, batchSize: Int) -> MLXArray {
        self.prevShape = X.shape
        var temp : [Int] = []
        temp.append(self.prevShape[0])
        for i in 0..<shape.count{
            temp.append(shape[i])
        }
        return X.reshaped(temp)
    }

    func backwardProp(error: MLXArray, batchSize: Int) -> MLXArray {
        
        return error.reshaped(self.prevShape)
    }
}
