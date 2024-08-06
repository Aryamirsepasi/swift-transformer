import Accelerate
//not needed (only the numpy variant

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

// Utility function to reshape a flat array into nested arrays
func reshape<T>(_ array: [T], newShape: [Int]) -> Any {
    let calculatedShape = calculateShape(array, newShape: newShape)
    let totalElements = calculatedShape.reduce(1, *)
    assert(array.count == totalElements, "Total elements must match in reshape operation")
    
    func reshapeHelper<T>(_ array: [T], shape: [Int]) -> Any {
        guard !shape.isEmpty else { return array }
        let step = array.count / shape[0]
        return stride(from: 0, to: array.count, by: step).map { Array(array[$0..<$0+step]) }
    }
    
    return reshapeHelper(array, shape: calculatedShape)
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

    func forwardProp(X: [Float], batchSize: Int) -> [[Float]] {
        self.prevShape = [batchSize] + self.shape
        let reshapedArray = reshape(X, newShape: [batchSize] + self.shape) as! [[Float]]
        return reshapedArray
    }

    func backwardProp(error: [Float], batchSize: Int) -> [[Float]] {
        guard let prevShape = self.prevShape else {
            fatalError("forwardProp must be called before backwardProp")
        }

        let reshapedError = reshape(error, newShape: prevShape) as! [[Float]]
        return reshapedError
    }
}
