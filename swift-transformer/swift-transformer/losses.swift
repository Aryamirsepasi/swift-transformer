import Foundation
import Accelerate
import MLX

protocol LossFunction {
    func loss(y: MLXArray, t: MLXArray) -> MLXArray
    func derivative(y: MLXArray, t: MLXArray) -> MLXArray
}


class CrossEntropy: LossFunction { //needed CrossEntropyLoss
    var ignore_index: Int
    let log_softmax = LogSoftmax()
    
    init(ignore_index: Int = 0) {
        self.ignore_index = ignore_index
    }
    
    func loss(y: MLXArray, t: MLXArray) -> MLXArray {
        
        print("entered CrossEntropy loss")
        var log_softmax = self.log_softmax.forward(x: y)
        let tInt = t.asType(Int.self)
        let numSamples = tInt.count
        let numClasses = log_softmax.shape[1]
        
        var nll_loss = MLX.zeros([numSamples], stream: .gpu)
        
        /*for i in 0..<numSamples{
            let labelIndex = tInt[i]
            if labelIndex.item(Int.self) == self.ignore_index || labelIndex.item(Int.self) < 0 || labelIndex.item(Int.self) >= numClasses {
                nll_loss[i] = MLXArray(0.0)
            } else {
                nll_loss[i] = -log_softmax[i, labelIndex.item(Int.self)]
            }
        }*/
        nll_loss = -log_softmax[MLXArray(0..<t.count),t]
        
        var loss_output = MLX.where(t .== self.ignore_index, 0, nll_loss, stream: .gpu)
        
        print("exited CrossEntropy loss")
        
        return loss_output
    }
    
    func derivative(y: MLXArray, t: MLXArray) -> MLXArray {
        print("entered CrossEntropy derivative")
        
        var softmax = self.log_softmax.forward(x: y)
        
        
        let tInt = t.asType(Int.self)
            let numSamples = tInt.count
            let numClasses = softmax.shape[1]
        
        var grad = softmax
        /*for i in 0..<numSamples{
            let labelIndex = tInt[i]
            if labelIndex.item(Int.self) == self.ignore_index || labelIndex.item(Int.self) < 0 || labelIndex.item(Int.self) >= numClasses {
                grad[i, 0...] = MLXArray(0.0)
            } else {
                grad[i, labelIndex] -= 1
            }
        }*/
        grad[MLXArray(0..<t.count), t] -= 1
        grad /= t.count
        var res = MLX.where(t.reshaped([-1,1]) .== self.ignore_index, 0, grad, stream: .gpu)
        
        print("exited CrossEntropy derivative")
        
        return res
        
    }
}

let lossFunctions: [String: LossFunction] = [
    "cross_entropy": CrossEntropy()
]
