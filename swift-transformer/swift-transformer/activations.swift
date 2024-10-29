import Foundation
import Accelerate
import MLX

protocol Activation {
    func forward(x: MLXArray) -> MLXArray
    func backward(grad: MLXArray) -> MLXArray
}


class ReLU: Activation {
    private var x: MLXArray = []
    
    func forward(x: MLXArray) -> MLXArray {
        autoreleasepool {
            self.x = x
            return MLX.maximum(x, 0, stream: .gpu)
        }
    }
    
    func backward(grad: MLXArray) -> MLXArray {
        autoreleasepool {
            let result = grad * MLX.where(self.x .<= 0, 0, 1, stream: .gpu).asType(self.x.dtype)
            self.x = [] // Clear reference after use
            return result
        }
    }
}

class Softmax: Activation {
    private var x: MLXArray = []
    private var softmaxOutput: MLXArray = [] // Cache the softmax output
    private var axis = -1
    
    func forward(x: MLXArray) -> MLXArray {
        autoreleasepool {
            self.x = x
            // Use MLX's optimized softmax implementation
            self.softmaxOutput = MLX.softmax(x - MLX.max(x, axis: axis, keepDims: true, stream: .gpu), axis: axis)
            return self.softmaxOutput
        }
    }
    
    /*func backward(grad: MLXArray) -> MLXArray {
     
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
     }*/
    func backward(grad: MLXArray) -> MLXArray {
        autoreleasepool {
            let batchSize = self.x.shape[0]
            
            // Reuse cached softmax output
            let softmax = self.softmaxOutput
            
            // Optimize memory usage by releasing cached values when done
            defer {
                self.x = []
                self.softmaxOutput = []
            }
            
            // Compute gradient more efficiently
            return autoreleasepool {
                let scaledGrad = grad / Float(batchSize)
                let softmaxGrad = softmax * (scaledGrad - MLX.sum(scaledGrad * softmax, axis: axis, keepDims: true, stream: .gpu))
                return softmaxGrad
            }
        }
    }
    
    
    
}

class Identity: Activation {
    private var x: MLXArray = []
    
    func forward(x: MLXArray) -> MLXArray {
        autoreleasepool {
            self.x = x
            return x
        }
    }
    
    func backward(grad: MLXArray) -> MLXArray {
        autoreleasepool {
            let result = grad
            self.x = [] // Clear reference
            return result
        }
    }
}

class LogSoftmax: Activation { //needed
    private var x: MLXArray = []
    private var axis = -1
    private var softmaxOutput: MLXArray = []
    
    
    /*func forward(x: MLXArray) -> MLXArray {
     
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
     }*/
    
    func forward(x: MLXArray) -> MLXArray {
        autoreleasepool {
            self.x = x
            
            // Compute and cache softmax
            let maxX = MLX.max(x, axis: axis, keepDims: true, stream: .gpu)
            let expX = MLX.exp(x - maxX, stream: .gpu)
            let sumExpX = MLX.sum(expX, axis: axis, keepDims: true, stream: .gpu)
            self.softmaxOutput = expX / sumExpX
            
            return MLX.log(self.softmaxOutput)
        }
    }
    
    func backward(grad: MLXArray) -> MLXArray {
        autoreleasepool {
            let batchSize = Float(self.x.shape[0])
            
            // Use cached softmax output
            let result = autoreleasepool {
                let gradSum = grad.sum(axis: axis, keepDims: true, stream: .gpu)
                return (grad - self.softmaxOutput * gradSum) / batchSize
            }
            
            // Clear cached values
            self.x = []
            self.softmaxOutput = []
            
            return result
        }
    }
}


let activations: [String: any Activation] = [
    "softmax": Softmax(),
    "relu": ReLU(),
    "identity": Identity(),
    "logsoftmax": LogSoftmax()
]
