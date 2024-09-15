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
        
        //print ("entered activation ReLU forward")

        self.x = x
        
        //print ("exited activation ReLU forward")

        return MLX.sigmoid(x)

    }
    
    func backward(grad: MLXArray) -> MLXArray {
        
        //print ("entered activation ReLU backward")

        return grad * MLX.where(self.x .<= 0, 0, 1).asType(self.x.dtype)
    }
}

class Softmax: Activation {
    private var x: MLXArray = []
    private var axis = -1
    
    func forward(x: MLXArray) -> MLXArray {
        
        //print ("entered activation Softmax forward")

        self.x = x
        
        var e_x = MLX.exp(x - MLX.max(x, axis: axis,keepDims: true))
        
        var softmax =  e_x / MLX.sum(e_x, axis: axis,keepDims: true)
                
        //print ("exited activation Softmax forward")

        return softmax
    }
    
    func backward(grad: MLXArray) -> MLXArray {
        //print("entered activation Softmax backward")

        let batch_size = self.x.shape[0]
        let softmax = self.forward(x: self.x)
        //print("softmax shape: ", softmax.shape)
        let num_classes = softmax.shape[softmax.ndim - 1]
        let num_batches = softmax.shape[0]

        // Create the identity matrix for num_classes
        let identity_matrix = MLX.identity(num_classes)

        // Reshape softmax for easier multiplication
        let softmax_expanded = softmax[.ellipsis, .newAxis]
        
        //  Calculate the diagonal part of the Jacobian
        let jacobian_diag = softmax_expanded * identity_matrix[.newAxis, .ellipsis]

        // Calculate the outer product part of the Jacobian
        let softmax_outer = MLX.matmul(softmax_expanded,softmax_expanded.transposed(0, 1, 2, 4, 3))

        // Combine to get the Jacobian matrix J
        let J = jacobian_diag - softmax_outer
        
        // Multiply the gradient with the Jacobian
        let grad_expanded = grad[.ellipsis, .newAxis, 0...]
        let input_grad = MLX.matmul(grad_expanded, J) // Shape: [batch_size, seq_len, num_classes, num_classes]

        //print("exited activation Softmax backward")

        //print((input_grad.reshaped(self.x.shape) / batch_size).shape)
        // Reshape and scale the gradient
        return input_grad.reshaped(self.x.shape) / batch_size
    }



}

class Identity: Activation { //needed
    private var x: MLXArray = []

    func forward(x: MLXArray) -> MLXArray {
        //print ("entered activation Identity forward")

        self.x = x
        
        //print ("exited activation Identity forward")

        return x
    }
    
    func backward(grad: MLXArray) -> MLXArray {
        
        //print ("entered activation Identity backward")

        var res = grad * MLX.ones(self.x.shape).asType(self.x.dtype)
        
        //print ("exited activation Identity backward")

        return res
    }
}

class LogSoftmax: Activation { //needed
    private var x: MLXArray = []
    private var axis = -1
    
    func forward(x: MLXArray) -> MLXArray {
        
        //print ("entered activation LogSoftmax forward")

        self.x = x
        var e_x = MLX.exp(x - MLX.max(x, axis: axis,keepDims: true))
        var softmax = e_x / MLX.sum(e_x, axis: axis,keepDims: true)
        
        var log_softmax = MLX.log(softmax)
        
        //print ("exited activation LogSoftmax forward")

        return log_softmax
    }
    
    func backward(grad: MLXArray) -> MLXArray {
        //print ("entered activation LogSoftmax backward")

        var e_x = MLX.exp(self.x - MLX.max(self.x, axis: axis,keepDims: true))
        var softmax = e_x / MLX.sum(e_x, axis: axis,keepDims: true)
            
        var batch_size = self.x.shape[0]
        var input_grad = grad - softmax * grad.sum(axis: axis,keepDims: true)
            
        //print ("exited activation LogSoftmax backward")

        return input_grad / batch_size
    }
}


let activations: [String: any Activation] = [
    "softmax": Softmax(),
    "relu": ReLU(),
    "identity": Identity(),
    "logsoftmax": LogSoftmax()
]
