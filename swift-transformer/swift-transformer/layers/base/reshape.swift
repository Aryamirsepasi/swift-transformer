import Accelerate
import MLX
import MLXRandom
class Reshape {
    var shape: [Int]
    var inputShape: [Int]
    var prevShape: [Int]
    var outputShape: [Int]

    init(shape: [Int]) {
        
        print("entered reshape init")

        self.shape = shape
        self.inputShape = []
        self.outputShape = []
        self.prevShape = []
        
        print("exited reshape init")

    }

    func build() {
        
        print("entered reshape build")

        self.outputShape = self.shape
        
        print("exited reshape build")

    }

    func forwardProp(X: MLXArray, batchSize: Int) -> MLXArray {
        
        print("entered reshape forwardProp")

        self.prevShape = X.shape
        var temp : [Int] = []
        temp.append(self.prevShape[0])
        for i in 0..<shape.count{
            temp.append(shape[i])
        }
        
        print("exited reshape forwardProp")

        return X.reshaped(temp, stream: .gpu)
    }

    func backwardProp(error: MLXArray, batchSize: Int) -> MLXArray {
        
        print("entered reshape backwardProp")

        return error.reshaped(self.prevShape)
    }
}
