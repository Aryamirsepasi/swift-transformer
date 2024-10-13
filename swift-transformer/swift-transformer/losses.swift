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
        
        //print("entered CrossEntropy loss")
        var log_softmax = self.log_softmax.forward(x: y)
        var nll_loss = MLX.zeros([t.count])
        for i in 0..<t.count{
            nll_loss[i] = -log_softmax[i, t[i]]
        }
        var loss_output = MLX.where(t .== self.ignore_index, 0, nll_loss)
        
        //print("exited CrossEntropy loss")
        
        return loss_output
    }
    
    func derivative(y: MLXArray, t: MLXArray) -> MLXArray {
        //print("entered CrossEntropy derivative")
        
        var softmax = self.log_softmax.forward(x: y)
        var grad = softmax
        for i in 0..<t.count{
            grad[i, t[i]] -= 1
        }
        grad /= t.count
        var res = MLX.where(t.reshaped([-1,1]) .== self.ignore_index, 0, grad)
        
        //print("exited CrossEntropy derivative")
        
        return res
        
    }
}

let lossFunctions: [String: LossFunction] = [
    "cross_entropy": CrossEntropy()
]
