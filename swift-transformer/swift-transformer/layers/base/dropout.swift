import Foundation
import Accelerate

//needed 
class Dropout {
    var rate: Float
    var inputShape: [Int]
    var outputShape: [Int]
    var mask: [Float]
    var dataType: [Float]
    
    init(rate: Float = 0.1, dataType: [Float] = []) {
        self.rate = rate
        self.inputShape = []
        self.outputShape = []
        self.mask = []
        self.dataType = dataType
    }
    
    func build() {
        self.outputShape = self.inputShape
    }
    
    func forward(_ X: [[[Float]]], training: Bool = true) -> [[[Float]]] {
        if !training {
            return X
        }
        
        let inputShape = shape(X)
        let maskAny = binomial(n: 1, p: 1 - self.rate, size: inputShape)
        
        // Flattening and reshaping mask to match the input shape
        let flatMask: [Float]
        if let maskArray = maskAny as? [Int] {
            flatMask = maskArray.map { Float($0) }
        } else if let maskArray3D = maskAny as? [[[Int]]] {
            flatMask = maskArray3D.flatMap { $0.flatMap { $0.map { Float($0) } } }
        } else {
            fatalError("Invalid mask type")
        }
        
        let flatX = X.flatMap { $0.flatMap { $0 } }
        var output = [Float](repeating: 0.0, count: flatX.count)
        vDSP_vmul(flatX, 1, flatMask, 1, &output, 1, vDSP_Length(flatX.count))
        
        return convert(output, to: inputShape) as! [[[Float]]]
    }
    
    func backward(_ error: [[[Float]]]) -> [[[Float]]] {
        let flatError = error.flatMap { $0.flatMap { $0 } }
        var outputError = [Float](repeating: 0.0, count: flatError.count)
        vDSP_vmul(flatError, 1, mask.flatMap { $0 }, 1, &outputError, 1, vDSP_Length(flatError.count))
        
        return convert(outputError, to: shape(error)) as! [[[Float]]]
    }
    
    func convert<T>(_ array: [T], to shape: [Int]) -> Any {
        var result: Any = array
        for s in shape.reversed() {
            if let flatResult = result as? [T] {
                result = reshape(flatResult, newShape: [s]) as Any
            }
        }
        return result
    }
}
