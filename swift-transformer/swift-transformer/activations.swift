import Foundation
import Accelerate

protocol Activation {
    func forward(x: [Float]) -> [Float]
    func backward(grad: [Float]) -> [Float]
}

class Sigmoid: Activation {
    private var output: [Float]?
    
    func forward(x: [Float]) -> [Float] {
        var result = [Float](repeating: 0.0, count: x.count)
        var negX = [Float](repeating: 0.0, count: x.count)
        vDSP_vneg(x, 1, &negX, 1, vDSP_Length(x.count))
        vvexpf(&result, negX, [Int32(x.count)])
        var one = Float(1.0)
        vDSP_vsadd(result, 1, &one, &result, 1, vDSP_Length(x.count))
        vDSP_svdiv(&one, result, 1, &result, 1, vDSP_Length(x.count))
        output = result
        return result
    }

    func backward(grad: [Float]) -> [Float] {
        guard let output = self.output else { return [] }
        var result = [Float](repeating: 0.0, count: grad.count)
        var oneMinusOutput = [Float](repeating: 0.0, count: grad.count)
        var one = Float(1.0)
        vDSP_vsadd(output, 1, &one, &oneMinusOutput, 1, vDSP_Length(grad.count))
        vDSP_vsub(oneMinusOutput, 1, output, 1, &result, 1, vDSP_Length(grad.count))
        vDSP_vmul(grad, 1, result, 1, &result, 1, vDSP_Length(grad.count))
        return result
    }
}

class Tanh: Activation {
    private var output: [Float]?
    
    func forward(x: [Float]) -> [Float] {
        var result = [Float](repeating: 0.0, count: x.count)
        vvtanhf(&result, x, [Int32(x.count)])
        output = result
        return result
    }
    
    func backward(grad: [Float]) -> [Float] {
        guard let output = self.output else { return [] }
        var result = [Float](repeating: 0.0, count: grad.count)
        var squaredOutput = [Float](repeating: 0.0, count: grad.count)
        vDSP_vsq(output, 1, &squaredOutput, 1, vDSP_Length(grad.count))
        var one = Float(1.0)
        var onesMinusSquaredOutput = [Float](repeating: 0.0, count: grad.count)
        vDSP_vsadd(squaredOutput, 1, &one, &onesMinusSquaredOutput, 1, vDSP_Length(grad.count))
        vDSP_vsub(onesMinusSquaredOutput, 1, squaredOutput, 1, &result, 1, vDSP_Length(grad.count))
        vDSP_vmul(grad, 1, result, 1, &result, 1, vDSP_Length(grad.count))
        return result
    }
}

class Softmax: Activation {
    private var output: [Float]?
    private var inputShape: [Int]?
    
    func forward(x: [Float]) -> [Float] {
        self.inputShape = calculateShape(x, newShape: [-1, x.count])
        var result = x
        let maxElement = x.max() ?? 0
        var maxArray = [Float](repeating: maxElement, count: x.count)
        vDSP_vsub(x, 1, &maxArray, 1, &result, 1, vDSP_Length(x.count))
        vvexpf(&result, result, [Int32(x.count)])
        var sumExp = result.reduce(0, +)
        vDSP_vsdiv(result, 1, &sumExp, &result, 1, vDSP_Length(x.count))
        output = result
        return result
    }
    
    func backward(grad: [Float]) -> [Float] {
        guard let softmax = self.output, let inputShape = self.inputShape else { return [] }
        let reshapedGrad = reshape(grad, newShape: inputShape)
        var gradSum: Float = 0
        vDSP_dotpr(reshapedGrad, 1, softmax, 1, &gradSum, vDSP_Length(reshapedGrad.count))
        var gradSumArray = [Float](repeating: gradSum, count: reshapedGrad.count)
        var result = [Float](repeating: 0.0, count: reshapedGrad.count)
        vDSP_vsub(reshapedGrad, 1, &gradSumArray, 1, &result, 1, vDSP_Length(reshapedGrad.count))
        vDSP_vmul(result, 1, softmax, 1, &result, 1, vDSP_Length(reshapedGrad.count))
        return reshape(result, newShape: inputShape)
    }
}


class ReLU: Activation {
    private var input: [Float]?
    
    func forward(x: [Float]) -> [Float] {
        var result = x
        var zero = Float(0)
        vDSP_vthres(x, 1, &zero, &result, 1, vDSP_Length(x.count))
        input = x
        return result
    }
    
    func backward(grad: [Float]) -> [Float] {
        guard let input = self.input else { return [] }
        var result = [Float](repeating: 0.0, count: grad.count)
        for i in 0..<input.count {
            result[i] = input[i] > 0 ? grad[i] : 0
        }
        return result
    }
}

class Softplus: Activation {
    private var x: [Float]?
    
    func forward(x: [Float]) -> [Float] {
        self.x = x
        var result = [Float](repeating: 0.0, count: x.count)
        var one = Float(1.0)
        vvlog1pf(&result, x.map { $0 + one }, [Int32(x.count)])
        return result
    }
    
    func backward(grad: [Float]) -> [Float] {
        guard let x = self.x else { return [] }
        var result = [Float](repeating: 0.0, count: x.count)
        var one = Float(1.0)
        var expX = [Float](repeating: 0.0, count: x.count)
        vvexpf(&expX, x, [Int32(x.count)])
        vDSP_vdiv(expX, 1, expX.map { $0 + one }, 1, &result, 1, vDSP_Length(x.count))
        vDSP_vmul(grad, 1, result, 1, &result, 1, vDSP_Length(x.count))
        return result
    }
}

class Softsign: Activation {
    private var x: [Float]?
    
    func forward(x: [Float]) -> [Float] {
        self.x = x
        var result = [Float](repeating: 0.0, count: x.count)
        var onePlusAbsX = [Float](repeating: 0.0, count: x.count)
        vDSP_vabs(x, 1, &onePlusAbsX, 1, vDSP_Length(x.count))
        let one = Float(1.0)
        vDSP_vsadd(onePlusAbsX, 1, [one], &onePlusAbsX, 1, vDSP_Length(x.count))
        vDSP_vdiv(x, 1, onePlusAbsX, 1, &result, 1, vDSP_Length(x.count))
        return result
    }
    
    func backward(grad: [Float]) -> [Float] {
        guard let x = self.x else { return [] }
        var result = [Float](repeating: 0.0, count: x.count)
        var onePlusAbsX = [Float](repeating: 0.0, count: x.count)
        vDSP_vabs(x, 1, &onePlusAbsX, 1, vDSP_Length(x.count))
        let one = Float(1.0)
        vDSP_vsadd(onePlusAbsX, 1, [one], &onePlusAbsX, 1, vDSP_Length(x.count))
        var denominator = [Float](repeating: 0.0, count: x.count)
        vvrecf(&denominator, onePlusAbsX, [Int32(x.count)])
        vvrecf(&denominator, denominator, [Int32(x.count)])
        vDSP_vmul(grad, 1, denominator, 1, &result, 1, vDSP_Length(x.count))
        return result
    }
}

class Swish: Activation {
    private var x: [Float]?
    private var beta: Float
    
    init(beta: Float = 1.0) {
        self.beta = beta
    }
    
    func forward(x: [Float]) -> [Float] {
        self.x = x
        var betaX = [Float](repeating: 0.0, count: x.count)
        var expBetaX = [Float](repeating: 0.0, count: x.count)
        var onePlusExpBetaX = [Float](repeating: 0.0, count: x.count)
        var result = [Float](repeating: 0.0, count: x.count)
        let one = Float(1.0)
        
        vDSP_vsmul(x, 1, &beta, &betaX, 1, vDSP_Length(x.count))
        vvexpf(&expBetaX, betaX, [Int32(x.count)])
        vDSP_vsadd(expBetaX, 1, [one], &onePlusExpBetaX, 1, vDSP_Length(x.count))
        
        var reciprocals = [Float](repeating: 0.0, count: x.count)
        vvrecf(&reciprocals, onePlusExpBetaX, [Int32(x.count)])
        vDSP_vmul(x, 1, reciprocals, 1, &result, 1, vDSP_Length(x.count))
        
        return result
    }
    
    func backward(grad: [Float]) -> [Float] {
        guard let x = self.x else { return [] }
        var sigmoid = [Float](repeating: 0.0, count: x.count)
        let forwardOutput = self.forward(x: x)
        for i in 0..<x.count {
            sigmoid[i] = 1 / (1 + exp(-beta * x[i]))
        }
        var sigmoidDerivative = [Float](repeating: 0.0, count: x.count)
        for i in 0..<x.count {
            sigmoidDerivative[i] = sigmoid[i] * (1 - sigmoid[i])
        }
        var result = [Float](repeating: 0.0, count: x.count)
        for i in 0..<x.count {
            result[i] = grad[i] * (beta * forwardOutput[i] + sigmoid[i] * (1 - beta * forwardOutput[i]))
        }
        return result
    }
}

class Mish: Activation {
    private var x: [Float]?
    
    func forward(x: [Float]) -> [Float] {
        self.x = x
        var softplus = [Float](repeating: 0.0, count: x.count)
        vvlog1pf(&softplus, x.map { 1 + exp($0) }, [Int32(x.count)])
        var tanhSoftplus = [Float](repeating: 0.0, count: x.count)
        vvtanhf(&tanhSoftplus, softplus, [Int32(x.count)])
        var result = [Float](repeating: 0.0, count: x.count)
        vDSP_vmul(x, 1, tanhSoftplus, 1, &result, 1, vDSP_Length(x.count))
        return result
    }
    
    func backward(grad: [Float]) -> [Float] {
        guard let x = self.x else { return [] }
        var softplus = [Float](repeating: 0.0, count: x.count)
        vvlog1pf(&softplus, x.map { 1 + exp($0) }, [Int32(x.count)])
        var tanhSoftplus = [Float](repeating: 0.0, count: x.count)
        vvtanhf(&tanhSoftplus, softplus, [Int32(x.count)])
        var dtanh = [Float](repeating: 0.0, count: x.count)
        var one = Float(1.0)
        vDSP_vsq(tanhSoftplus, 1, &dtanh, 1, vDSP_Length(x.count))
        vDSP_vsmsa(dtanh, 1, &one, &one, &dtanh, 1, vDSP_Length(x.count))
        var delta = [Float](repeating: 0.0, count: x.count)
        for i in 0..<x.count {
            delta[i] = (exp(x[i]) * (4 * (x[i] + 1) + 4 * exp(2 * x[i]) + exp(3 * x[i]) + exp(x[i]) * (4 * x[i] + 6))) / pow((2 * exp(x[i]) + exp(2 * x[i]) + 2), 2)
        }
        var result = [Float](repeating: 0.0, count: x.count)
        vDSP_vmul(grad, 1, delta, 1, &result, 1, vDSP_Length(x.count))
        return result
    }
}

class TanhExp: Activation {
    private var x: [Float]?
    
    func forward(x: [Float]) -> [Float] {
        self.x = x
        var expX = [Float](repeating: 0.0, count: x.count)
        vvexpf(&expX, x, [Int32(x.count)])
        var tanhExpX = [Float](repeating: 0.0, count: x.count)
        vvtanhf(&tanhExpX, expX, [Int32(x.count)])
        var result = [Float](repeating: 0.0, count: x.count)
        vDSP_vmul(x, 1, tanhExpX, 1, &result, 1, vDSP_Length(x.count))
        return result
    }
    
    func backward(grad: [Float]) -> [Float] {
        guard let x = self.x else { return [] }
        var expX = [Float](repeating: 0.0, count: x.count)
        vvexpf(&expX, x, [Int32(x.count)])
        var tanhExpX = [Float](repeating: 0.0, count: x.count)
        vvtanhf(&tanhExpX, expX, [Int32(x.count)])
        var dtanhExpX = [Float](repeating: 0.0, count: x.count)
        for i in 0..<x.count {
            dtanhExpX[i] = tanhExpX[i] - x[i] * expX[i] * (1 - pow(tanhExpX[i], 2))
        }
        var result = [Float](repeating: 0.0, count: x.count)
        vDSP_vmul(grad, 1, dtanhExpX, 1, &result, 1, vDSP_Length(x.count))
        return result
    }
}

class LeakyReLU: Activation {
    private var alpha: Float
    private var x: [Float]?
    
    init(alpha: Float = 0.01) {
        self.alpha = alpha
    }
    
    func forward(x: [Float]) -> [Float] {
        self.x = x
        var result = [Float](repeating: 0.0, count: x.count)
        for i in 0..<x.count {
            result[i] = x[i] > 0 ? x[i] : alpha * x[i]
        }
        return result
    }
    
    func backward(grad: [Float]) -> [Float] {
        guard let x = self.x else { return [] }
        var result = [Float](repeating: 0.0, count: x.count)
        for i in 0..<x.count {
            result[i] = x[i] > 0 ? grad[i] : alpha * grad[i]
        }
        return result
    }
}

class ELU: Activation {
    private var alpha: Float
    private var x: [Float]?
    
    init(alpha: Float = 1.0) {
        self.alpha = alpha
    }
    
    func forward(x: [Float]) -> [Float] {
        self.x = x
        var result = [Float](repeating: 0.0, count: x.count)
        for i in 0..<x.count {
            result[i] = x[i] > 0 ? x[i] : alpha * (exp(x[i]) - 1)
        }
        return result
    }
    
    func backward(grad: [Float]) -> [Float] {
        guard let x = self.x else { return [] }
        var result = [Float](repeating: 0.0, count: x.count)
        for i in 0..<x.count {
            result[i] = x[i] > 0 ? grad[i] : grad[i] * (alpha + forward(x: [x[i]])[0])
        }
        return result
    }
}

class SELU: Activation {
    private let alpha: Float = 1.6732632423543772848170429916717
    private let lambda: Float = 1.0507009873554804934193349852946
    private var x: [Float]?
    
    func forward(x: [Float]) -> [Float] {
        self.x = x
        var result = [Float](repeating: 0.0, count: x.count)
        for i in 0..<x.count {
            result[i] = lambda * (x[i] > 0 ? x[i] : alpha * (exp(x[i]) - 1))
        }
        return result
    }
    
    func backward(grad: [Float]) -> [Float] {
        guard let x = self.x else { return [] }
        var result = [Float](repeating: 0.0, count: x.count)
        for i in 0..<x.count {
            let deriv = x[i] > 0 ? lambda : lambda * alpha * exp(x[i])
            result[i] = grad[i] * deriv
        }
        return result
    }
}

class GELU: Activation {
    private var x: [Float]?
    
    func forward(x: [Float]) -> [Float] {
        self.x = x
        var result = [Float](repeating: 0.0, count: x.count)
        let sqrtTwoOverPi = sqrt(2 / Float.pi)
        for i in 0..<x.count {
            let term = 1 + tanh(sqrtTwoOverPi * (x[i] + 0.044715 * pow(x[i], 3)))
            result[i] = 0.5 * x[i] * term
        }
        return result
    }
    
    func backward(grad: [Float]) -> [Float] {
        guard let x = self.x else { return [] }
        var result = [Float](repeating: 0.0, count: x.count)
        let sqrtTwoOverPi = sqrt(2 / Float.pi)
        for i in 0..<x.count {
            let xCubed = pow(x[i], 3)
            let tanhVal = tanh(sqrtTwoOverPi * (x[i] + 0.044715 * xCubed))
            let sechVal = 1 - pow(tanhVal, 2)
            let term = 0.5 * tanhVal + (0.0356774 * xCubed + 0.797885 * x[i]) * sechVal * sqrtTwoOverPi
            result[i] = grad[i] * (0.5 + 0.5 * term)
        }
        return result
    }
}

class Identity: Activation {
    func forward(x: [Float]) -> [Float] {
        return x
    }
    
    func backward(grad: [Float]) -> [Float] {
        return grad
    }
}

class LogSoftmax: Activation {
    private var x: [Float]?
    private var softmax: [Float]?
    
    func forward(x: [Float]) -> [Float] {
        self.x = x
        var maxVal = x.max() ?? 0
        var shiftedX = x.map { $0 - maxVal }
        var exps = [Float](repeating: 0.0, count: x.count)
        vvexpf(&exps, shiftedX, [Int32(x.count)])
        let sumExps = exps.reduce(0, +)
        self.softmax = exps.map { $0 / sumExps }
        let logSoftmax = self.softmax!.map { log($0) }
        return logSoftmax
    }
    
    func backward(grad: [Float]) -> [Float] {
        guard let softmax = self.softmax, let x = self.x else { return [] }
        let sumGradSoftmax = grad.reduce(0, +)
        return grad.map { $0 - sumGradSoftmax }
    }
}

let activations: [String: Activation] = [
    "sigmoid": Sigmoid(),
    "tanh": Tanh(),
    "softmax": Softmax(),
    "softplus": Softplus(),
    "softsign": Softsign(),
    "swish": Swish(),
    "mish": Mish(),
    "tanh_exp": TanhExp(),
    "relu": ReLU(),
    "leaky_relu": LeakyReLU(),
    "elu": ELU(),
    "selu": SELU(),
    "gelu": GELU(),
    "identity": Identity(),
    "logsoftmax": LogSoftmax()
]
