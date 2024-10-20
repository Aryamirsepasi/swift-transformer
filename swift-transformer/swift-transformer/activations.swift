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
        
        return MLX.sigmoid(x, stream: .gpu)

    }
    
    func backward(grad: MLXArray) -> MLXArray {
        
        return grad * MLX.where(self.x .<= 0, 0, 1, stream: .gpu).asType(self.x.dtype)
    }
}

class Softmax: Activation {
    private var x: MLXArray = []
    private var axis = -1
    
    func forward(x: MLXArray) -> MLXArray {
        
        self.x = x
        
        //var e_x = MLX.exp(x - MLX.max(x, axis: axis,keepDims: true, stream: .gpu), stream: .gpu)
        
        //var softmax =  e_x / MLX.sum(e_x, axis: axis,keepDims: true, stream: .gpu)
        let softmax = MLX.softmax(x - MLX.max(x, axis: axis,keepDims: true, stream: .gpu), axis: axis)
                        

        return softmax
    }
    
    func backward(grad: MLXArray) -> MLXArray {

        let batch_size = self.x.shape[0]
        let softmax = self.forward(x: self.x)

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
        let input_grad = MLX.matmul(grad_expanded, J, stream: .gpu) // Shape: [batch_size, seq_len, num_classes, num_classes]

        // Reshape and scale the gradient
        return input_grad.reshaped(self.x.shape, stream: .gpu) / batch_size
    }



}

class Identity: Activation { //needed
    private var x: MLXArray = []

    func forward(x: MLXArray) -> MLXArray {
        
        self.x = x
        
        return x
    }
    
    func backward(grad: MLXArray) -> MLXArray {
        
        let res = grad * MLX.ones(self.x.shape, stream: .gpu).asType(self.x.dtype)
        
        return res
    }
}

class LogSoftmax: Activation { //needed
    private var x: MLXArray = []
    private var axis = -1
    
    func forward(x: MLXArray) -> MLXArray {
        
        self.x = x
        let e_x = MLX.exp(x - MLX.max(x, axis: self.axis, keepDims: true, stream: .gpu), stream: .gpu)
        let softmax = e_x / MLX.sum(e_x, axis: self.axis, keepDims: true, stream: .gpu)
        
        let log_softmax = MLX.log(softmax)
                
        return log_softmax
    }
    
    func backward(grad: MLXArray) -> MLXArray {
        
        let e_x = MLX.exp(self.x - MLX.max(self.x, axis: axis,keepDims: true, stream: .gpu), stream: .gpu)
        let softmax = e_x / MLX.sum(e_x, axis: axis,keepDims: true, stream: .gpu)
            
        let batch_size = self.x.shape[0]
        let input_grad = grad - softmax * grad.sum(axis: axis,keepDims: true, stream: .gpu)
            
        return input_grad / batch_size
    }
}


let activations: [String: any Activation] = [
    "softmax": Softmax(),
    "relu": ReLU(),
    "identity": Identity(),
    "logsoftmax": LogSoftmax()
]
