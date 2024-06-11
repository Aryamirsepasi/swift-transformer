import Foundation
import Accelerate

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
    
    func forward(_ X: [[Float]], training: Bool = true) -> [[Float]] {
        self.inputShape = shape(X)
        
        if !training {
            return X
        }
        
        let maskAny = binomial(n: 1, p: 1 - self.rate, size: self.inputShape)
        
        // Cast the mask to the appropriate type
        if let maskArray = maskAny as? [Int] {
            self.mask = maskArray.map { Float($0) }
        } else if let maskArray2D = maskAny as? [[Int]] {
            self.mask = maskArray2D.flatMap { $0.map { Float($0) } }
        } else {
            fatalError("Invalid mask type")
        }
        
        let flatX = X.flatMap { $0 }
        var output = [Float](repeating: 0.0, count: flatX.count)
        vDSP_vmul(flatX, 1, mask, 1, &output, 1, vDSP_Length(flatX.count))
        
        return convert(output, to: self.inputShape)
    }
    
    func backward(_ error: [[Float]]) -> [[Float]] {
        let flatError = error.flatMap { $0 }
        var outputError = [Float](repeating: 0.0, count: flatError.count)
        vDSP_vmul(flatError, 1, mask, 1, &outputError, 1, vDSP_Length(flatError.count))
        
        return convert(outputError, to: self.inputShape)
    }
    
    func convert(_ array: [Float], to shape: [Int]) -> [[Float]] {
        var reshapedArray: [[Float]] = []
        var start = 0
        for count in shape {
            let end = start + count
            reshapedArray.append(Array(array[start..<end]))
            start = end
        }
        return reshapedArray
    }
}
