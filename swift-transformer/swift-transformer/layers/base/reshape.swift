import Accelerate

// Utility function to calculate the shape of a flat array
func calculateShape<T>(_ array: [T], newShape: [Int]) -> [Int] {
    var shape = newShape
    let totalElements = array.count
    let inferredDimIndex = shape.firstIndex(of: -1)
    if let index = inferredDimIndex {
        let inferredDim = totalElements / shape.filter { $0 != -1 }.reduce(1, *)
        shape[index] = inferredDim
    }
    return shape
}

// Utility function to reshape a flat array
func reshape<T>(_ array: [T], newShape: [Int]) -> [T] {
    let calculatedShape = calculateShape(array, newShape: newShape)
    let totalElements = calculatedShape.reduce(1, *)
    assert(array.count == totalElements, "Total elements must match in reshape operation")
    return array // No actual reshaping needed as this is a flat array operation
}

// Reshape class using the above utility functions
class Reshape {
    var shape: [Int]
    var inputShape: [Int]?
    var prevShape: [Int]?
    var outputShape: [Int]?

    init(shape: [Int]) {
        self.shape = shape
    }

    func build() {
        
        self.outputShape = self.shape
    }

    func forwardProp(X: [Float], batchSize: Int) -> [Float] {
        self.prevShape = [batchSize] + self.shape
        let reshapedArray = reshape(X, newShape: [batchSize] + self.shape)
        return reshapedArray
    }

    func backwardProp(error: [Float], batchSize: Int) -> [Float] {
        guard let prevShape = self.prevShape else {
            fatalError("forwardProp must be called before backwardProp")
        }

        let reshapedError = reshape(error, newShape: prevShape)
        return reshapedError
    }
}
