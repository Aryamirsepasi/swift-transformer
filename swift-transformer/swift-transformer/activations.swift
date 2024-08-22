import Foundation
import Accelerate
import MLX

protocol Activation {
    func forward(x: MLXArray) -> MLXArray
    func backward(grad: MLXArray) -> MLXArray
}


class ReLU: Activation { //needed
    
    private var x: MLXArray = []
    
    func forward(x: MLXArray) -> MLXArray {
        self.x = x
        return MLX.sigmoid(x)

    }
    
    func backward(grad: MLXArray) -> MLXArray {
        return grad * MLX.where(self.x .<= 0, 0, 1).asType(self.x.dtype)
    }
}

class Softmax: Activation {
    private var x: MLXArray = []
    private var axis = -1
    
    func forward(x: MLXArray) -> MLXArray {
        self.x = x
        
        var e_x = MLX.exp(x - MLX.max(x, axis: axis,keepDims: true))
        
        var softmax =  e_x / MLX.sum(e_x, axis: axis,keepDims: true)
                
        return softmax
    }
    
    func backward(grad: MLXArray) -> MLXArray {
        
        var batch_size = self.x.shape[0]
        var softmax = self.forward(x: self.x)
        var num_classes = softmax.shape[-1]
        
        var identity_matrix = MLX.identity(num_classes)
        
        var softmax_expanded = softmax[.ellipsis, .newAxis]
        var identity_expanded = identity_matrix[.newAxis, .ellipsis]
        var J = softmax_expanded * identity_expanded - MLX.matmul(softmax_expanded, softmax_expanded.transposed(0, 2, 1))


                
        var input_grad = MLX.matmul(grad[.ellipsis, .newAxis, 0...], J)
        
        return input_grad.reshaped(self.x.shape) / batch_size
        
    }
}

class Identity: Activation { //needed
    private var x: MLXArray = []

    func forward(x: MLXArray) -> MLXArray {
        self.x = x
        return x
    }
    
    func backward(grad: MLXArray) -> MLXArray {
        var res = grad * MLX.ones(self.x.shape).asType(self.x.dtype)
        
        return res
    }
}

class LogSoftmax: Activation { //needed
    private var x: MLXArray = []
    private var axis = -1
    
    func forward(x: MLXArray) -> MLXArray {
        self.x = x
        var e_x = MLX.exp(x - MLX.max(x, axis: axis,keepDims: true))
        var softmax = e_x / MLX.sum(e_x, axis: axis,keepDims: true)
        
        var log_softmax = MLX.log(softmax)
        
        return log_softmax
    }
    
    func backward(grad: MLXArray) -> MLXArray {
        var e_x = MLX.exp(self.x - MLX.max(self.x, axis: axis,keepDims: true))
        var softmax = e_x / MLX.sum(e_x, axis: axis,keepDims: true)
            
        var batch_size = self.x.shape[0]
        var input_grad = grad - softmax * grad.sum(axis: axis,keepDims: true)
            
        return input_grad / batch_size
    }
}


let activations: [String: any Activation] = [
    "softmax": Softmax(),
    "relu": ReLU(),
    "identity": Identity(),
    "logsoftmax": LogSoftmax()
]
