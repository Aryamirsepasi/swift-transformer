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
        
        var log_softmax = self.log_softmax.forward(x: y)
        var nll_loss = MLX.zeros([t.count])
        for i in 0..<t.count{
            nll_loss[i] = -log_softmax[i, t[i]]
        }
        var loss_output = MLX.where(t .== self.ignore_index, 0, nll_loss)
        return loss_output
    }
    
    func derivative(y: MLXArray, t: MLXArray) -> MLXArray {
        var batch_size = y.shape[0]
        var err = 1 / batch_size
                
        var nll_loss_der = MLX.zeros(like: y)
        
        /*
         has 2 errors:
         - Cannot convert value of type 'MLXArray' to expected condition type 'Bool'
         - Cannot assign value of type 'Int' to subscript of type 'MLXArray'
        for i in 0..<t.count{
            if (ti .!= self.ignore_index){
                nll_loss_der[i, t[i]] = -err
            }
        }
         */
        
        // Custom logic to replace np.isin and np.where
            for i in 0..<batch_size {
                if t[i].item(Int.self) != ignore_index {
                    let targetIndex = t[i].item(Int.self)
                    if y[i, targetIndex].item(Float.self) == y[i, targetIndex].item(Float.self) {
                        // Equivalent of np.isin logic
                        // Not sure if correct:
                        nll_loss_der[i, targetIndex] = MLXArray(-err)
                    }
                }
            }
        
        var output_err = self.log_softmax.backward(grad: nll_loss_der)
        
        return MLX.where(t.reshaped(-1, 1) .== self.ignore_index, 0, output_err)
    }
}

let lossFunctions: [String: LossFunction] = [
    "cross_entropy": CrossEntropy()
]
