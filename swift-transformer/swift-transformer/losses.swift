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
            
            let batch_size = y.shape[0]
            let err = 1 / batch_size
            
            var nll_loss_der = MLX.zeros(like: y)
            
            // Create a range of indices similar to np.arange
            let indices = MLXArray(0..<t.count)
            
            // Iterate over each batch
            for i in 0..<batch_size {
                let targetIndex = t[i].item(Int.self)
                if targetIndex != ignore_index {
                    // Find the target index in the y array and assign the error
                    nll_loss_der[i, targetIndex] = MLXArray(-err)
                }
            }
            
            let output_err = self.log_softmax.backward(grad: nll_loss_der)
            
            //print("exited CrossEntropy derivative")
            return MLX.where(t.reshaped([-1, 1]) .== self.ignore_index, 0, output_err)
        }
}

let lossFunctions: [String: LossFunction] = [
    "cross_entropy": CrossEntropy()
]
