import Foundation
import Accelerate

protocol Activation {
    func forward(x: [[Float]]) -> [[Float]]
    func backward(grad: [[Float]]) -> [[Float]]
}

class ReLU: Activation {
    private var input: [[Float]]?
    
    func forward(x: [[Float]]) -> [[Float]] {
        input = x
        return x.map { row in
            var result = row
            var zero = Float(0)
            vDSP_vthres(row, 1, &zero, &result, 1, vDSP_Length(row.count))
            return result
        }
    }
    
    func backward(grad: [[Float]]) -> [[Float]] {
        guard let input = self.input else { return [] }
        return zip(input, grad).map { (inputRow, gradRow) in
            var result = [Float](repeating: 0.0, count: gradRow.count)
            for i in 0..<inputRow.count {
                result[i] = inputRow[i] > 0 ? gradRow[i] : 0
            }
            return result
        }
    }
}

class Sigmoid: Activation {
    private var output: [[Float]]?
    
    func forward(x: [[Float]]) -> [[Float]] {
        output = x.map { row in
            var result = [Float](repeating: 0.0, count: row.count)
            var negRow = [Float](repeating: 0.0, count: row.count)
            vDSP_vneg(row, 1, &negRow, 1, vDSP_Length(row.count))
            vvexpf(&result, negRow, [Int32(row.count)])
            var one = Float(1.0)
            vDSP_vsadd(result, 1, &one, &result, 1, vDSP_Length(row.count))
            vDSP_svdiv(&one, result, 1, &result, 1, vDSP_Length(row.count))
            return result
        }
        return output!
    }

    func backward(grad: [[Float]]) -> [[Float]] {
        guard let output = self.output else { return [] }
        return zip(output, grad).map { (outputRow, gradRow) in
            var result = [Float](repeating: 0.0, count: gradRow.count)
            var oneMinusOutput = [Float](repeating: 0.0, count: gradRow.count)
            var one = Float(1.0)
            vDSP_vsadd(outputRow, 1, &one, &oneMinusOutput, 1, vDSP_Length(gradRow.count))
            vDSP_vsub(oneMinusOutput, 1, outputRow, 1, &result, 1, vDSP_Length(gradRow.count))
            vDSP_vmul(gradRow, 1, result, 1, &result, 1, vDSP_Length(gradRow.count))
            return result
        }
    }
}

class Tanh: Activation {
    private var output: [[Float]]?
    
    func forward(x: [[Float]]) -> [[Float]] {
        output = x.map { row in
            var result = [Float](repeating: 0.0, count: row.count)
            vvtanhf(&result, row, [Int32(row.count)])
            return result
        }
        return output!
    }
    
    func backward(grad: [[Float]]) -> [[Float]] {
        guard let output = self.output else { return [] }
        return zip(output, grad).map { (outputRow, gradRow) in
            var result = [Float](repeating: 0.0, count: gradRow.count)
            var squaredOutput = [Float](repeating: 0.0, count: gradRow.count)
            vDSP_vsq(outputRow, 1, &squaredOutput, 1, vDSP_Length(gradRow.count))
            var one = Float(1.0)
            var onesMinusSquaredOutput = [Float](repeating: 0.0, count: gradRow.count)
            vDSP_vsadd(squaredOutput, 1, &one, &onesMinusSquaredOutput, 1, vDSP_Length(gradRow.count))
            vDSP_vsub(onesMinusSquaredOutput, 1, squaredOutput, 1, &result, 1, vDSP_Length(gradRow.count))
            vDSP_vmul(gradRow, 1, result, 1, &result, 1, vDSP_Length(gradRow.count))
            return result
        }
    }
}

class Softmax: Activation {
    private var output: [[Float]]?
    
    func forward(x: [[Float]]) -> [[Float]] {
        output = x.map { row in
            var result = row
            let maxElement = row.max() ?? 0
            var maxArray = [Float](repeating: maxElement, count: row.count)
            vDSP_vsub(row, 1, &maxArray, 1, &result, 1, vDSP_Length(row.count))
            vvexpf(&result, result, [Int32(row.count)])
            var sumExp = result.reduce(0, +)
            vDSP_vsdiv(result, 1, &sumExp, &result, 1, vDSP_Length(row.count))
            return result
        }
        return output!
    }
    
    func backward(grad: [[Float]]) -> [[Float]] {
        guard let softmax = self.output else { return [] }
        return zip(softmax, grad).map { (softmaxRow, gradRow) in
            var gradSum: Float = 0
            vDSP_dotpr(gradRow, 1, softmaxRow, 1, &gradSum, vDSP_Length(gradRow.count))
            var gradSumArray = [Float](repeating: gradSum, count: gradRow.count)
            var result = [Float](repeating: 0.0, count: gradRow.count)
            vDSP_vsub(gradRow, 1, &gradSumArray, 1, &result, 1, vDSP_Length(gradRow.count))
            vDSP_vmul(result, 1, softmaxRow, 1, &result, 1, vDSP_Length(gradRow.count))
            return result
        }
    }
}

// Similarly, update other Activation classes to work with [[Float]]...

class Softplus: Activation {
    private var x: [[Float]]?
    
    func forward(x: [[Float]]) -> [[Float]] {
        self.x = x
        return x.map { row in
            var result = [Float](repeating: 0.0, count: row.count)
            var one = Float(1.0)
            vvlog1pf(&result, row.map { $0 + one }, [Int32(row.count)])
            return result
        }
    }
    
    func backward(grad: [[Float]]) -> [[Float]] {
        guard let x = self.x else { return [] }
        return zip(x, grad).map { (xRow, gradRow) in
            var result = [Float](repeating: 0.0, count: xRow.count)
            var one = Float(1.0)
            var expX = [Float](repeating: 0.0, count: xRow.count)
            vvexpf(&expX, xRow, [Int32(xRow.count)])
            vDSP_vdiv(expX, 1, expX.map { $0 + one }, 1, &result, 1, vDSP_Length(xRow.count))
            vDSP_vmul(gradRow, 1, result, 1, &result, 1, vDSP_Length(xRow.count))
            return result
        }
    }
}

class Softsign: Activation {
    private var x: [[Float]]?
    
    func forward(x: [[Float]]) -> [[Float]] {
        self.x = x
        return x.map { row in
            var result = [Float](repeating: 0.0, count: row.count)
            var onePlusAbsX = [Float](repeating: 0.0, count: row.count)
            vDSP_vabs(row, 1, &onePlusAbsX, 1, vDSP_Length(row.count))
            let one = Float(1.0)
            vDSP_vsadd(onePlusAbsX, 1, [one], &onePlusAbsX, 1, vDSP_Length(row.count))
            vDSP_vdiv(row, 1, onePlusAbsX, 1, &result, 1, vDSP_Length(row.count))
            return result
        }
    }
    
    func backward(grad: [[Float]]) -> [[Float]] {
        guard let x = self.x else { return [] }
        return zip(x, grad).map { (xRow, gradRow) in
            var result = [Float](repeating: 0.0, count: xRow.count)
            var onePlusAbsX = [Float](repeating: 0.0, count: xRow.count)
            vDSP_vabs(xRow, 1, &onePlusAbsX, 1, vDSP_Length(xRow.count))
            let one = Float(1.0)
            vDSP_vsadd(onePlusAbsX, 1, [one], &onePlusAbsX, 1, vDSP_Length(xRow.count))
            var denominator = [Float](repeating: 0.0, count: xRow.count)
            vvrecf(&denominator, onePlusAbsX, [Int32(xRow.count)])
            vvrecf(&denominator, denominator, [Int32(xRow.count)])
            vDSP_vmul(gradRow, 1, denominator, 1, &result, 1, vDSP_Length(xRow.count))
            return result
        }
    }
}

class Swish: Activation {
    private var x: [[Float]]?
    private var beta: Float
    
    init(beta: Float = 1.0) {
        self.beta = beta
    }
    
    func forward(x: [[Float]]) -> [[Float]] {
        self.x = x
        return x.map { row in
            var betaRow = [Float](repeating: 0.0, count: row.count)
            var expBetaRow = [Float](repeating: 0.0, count: row.count)
            var onePlusExpBetaRow = [Float](repeating: 0.0, count: row.count)
            var result = [Float](repeating: 0.0, count: row.count)
            let one = Float(1.0)
            
            vDSP_vsmul(row, 1, &beta, &betaRow, 1, vDSP_Length(row.count))
            vvexpf(&expBetaRow, betaRow, [Int32(row.count)])
            vDSP_vsadd(expBetaRow, 1, [one], &onePlusExpBetaRow, 1, vDSP_Length(row.count))
            
            var reciprocals = [Float](repeating: 0.0, count: row.count)
            vvrecf(&reciprocals, onePlusExpBetaRow, [Int32(row.count)])
            vDSP_vmul(row, 1, reciprocals, 1, &result, 1, vDSP_Length(row.count))
            
            return result
        }
    }
    
    func backward(grad: [[Float]]) -> [[Float]] {
        guard let x = self.x else { return [] }
        return x.enumerated().map { (i, xRow) in
            let forwardOutput = self.forward(x: [xRow])[0]
            var sigmoid = [Float](repeating: 0.0, count: xRow.count)
            for j in 0..<xRow.count {
                sigmoid[j] = 1 / (1 + exp(-beta * xRow[j]))
            }
            var sigmoidDerivative = [Float](repeating: 0.0, count: xRow.count)
            for j in 0..<xRow.count {
                sigmoidDerivative[j] = sigmoid[j] * (1 - sigmoid[j])
            }
            var result = [Float](repeating: 0.0, count: xRow.count)
            for j in 0..<xRow.count {
                result[j] = grad[i][j] * (beta * forwardOutput[j] + sigmoid[j] * (1 - beta * forwardOutput[j]))
            }
            return result
        }
    }
}

class Mish: Activation {
    private var x: [[Float]]?
    
    func forward(x: [[Float]]) -> [[Float]] {
        self.x = x
        return x.map { row in
            var softplus = [Float](repeating: 0.0, count: row.count)
            vvlog1pf(&softplus, row.map { 1 + exp($0) }, [Int32(row.count)])
            var tanhSoftplus = [Float](repeating: 0.0, count: row.count)
            vvtanhf(&tanhSoftplus, softplus, [Int32(row.count)])
            var result = [Float](repeating: 0.0, count: row.count)
            vDSP_vmul(row, 1, tanhSoftplus, 1, &result, 1, vDSP_Length(row.count))
            return result
        }
    }
    
    func backward(grad: [[Float]]) -> [[Float]] {
        guard let x = self.x else { return [] }
        return x.enumerated().map { (i, xRow) in
            var softplus = [Float](repeating: 0.0, count: xRow.count)
            vvlog1pf(&softplus, xRow.map { 1 + exp($0) }, [Int32(xRow.count)])
            var tanhSoftplus = [Float](repeating: 0.0, count: xRow.count)
            vvtanhf(&tanhSoftplus, softplus, [Int32(xRow.count)])
            var dtanh = [Float](repeating: 0.0, count: xRow.count)
            var one = Float(1.0)
            vDSP_vsq(tanhSoftplus, 1, &dtanh, 1, vDSP_Length(xRow.count))
            vDSP_vsmsa(dtanh, 1, &one, &one, &dtanh, 1, vDSP_Length(xRow.count))
            var delta = [Float](repeating: 0.0, count: xRow.count)
            for j in 0..<xRow.count {
                delta[j] = (exp(xRow[j]) * (4 * (xRow[j] + 1) + 4 * exp(2 * xRow[j]) + exp(3 * xRow[j]) + exp(xRow[j]) * (4 * xRow[j] + 6))) / pow((2 * exp(xRow[j]) + exp(2 * xRow[j]) + 2), 2)
            }
            var result = [Float](repeating: 0.0, count: xRow.count)
            vDSP_vmul(grad[i], 1, delta, 1, &result, 1, vDSP_Length(xRow.count))
            return result
        }
    }
}

class TanhExp: Activation {
    private var x: [[Float]]?
    
    func forward(x: [[Float]]) -> [[Float]] {
        self.x = x
        return x.map { row in
            var expX = [Float](repeating: 0.0, count: row.count)
            vvexpf(&expX, row, [Int32(row.count)])
            var tanhExpX = [Float](repeating: 0.0, count: row.count)
            vvtanhf(&tanhExpX, expX, [Int32(row.count)])
            var result = [Float](repeating: 0.0, count: row.count)
            vDSP_vmul(row, 1, tanhExpX, 1, &result, 1, vDSP_Length(row.count))
            return result
        }
    }
    
    func backward(grad: [[Float]]) -> [[Float]] {
        guard let x = self.x else { return [] }
        return zip(x, grad).map { (xRow, gradRow) in
            var expX = [Float](repeating: 0.0, count: xRow.count)
            vvexpf(&expX, xRow, [Int32(xRow.count)])
            var tanhExpX = [Float](repeating: 0.0, count: xRow.count)
            vvtanhf(&tanhExpX, expX, [Int32(xRow.count)])
            var dtanhExpX = [Float](repeating: 0.0, count: xRow.count)
            for j in 0..<xRow.count {
                dtanhExpX[j] = tanhExpX[j] - xRow[j] * expX[j] * (1 - pow(tanhExpX[j], 2))
            }
            var result = [Float](repeating: 0.0, count: xRow.count)
            vDSP_vmul(gradRow, 1, dtanhExpX, 1, &result, 1, vDSP_Length(xRow.count))
            return result
        }
    }
}

class LeakyReLU: Activation {
    private var alpha: Float
    private var x: [[Float]]?
    
    init(alpha: Float = 0.01) {
        self.alpha = alpha
    }
    
    func forward(x: [[Float]]) -> [[Float]] {
        self.x = x
        return x.map { row in
            var result = [Float](repeating: 0.0, count: row.count)
            for j in 0..<row.count {
                result[j] = row[j] > 0 ? row[j] : alpha * row[j]
            }
            return result
        }
    }
    
    func backward(grad: [[Float]]) -> [[Float]] {
        guard let x = self.x else { return [] }
        return zip(x, grad).map { (xRow, gradRow) in
            var result = [Float](repeating: 0.0, count: xRow.count)
            for j in 0..<xRow.count {
                result[j] = xRow[j] > 0 ? gradRow[j] : alpha * gradRow[j]
            }
            return result
        }
    }
}

class ELU: Activation {
    private var alpha: Float
    private var x: [[Float]]?
    
    init(alpha: Float = 1.0) {
        self.alpha = alpha
    }
    
    func forward(x: [[Float]]) -> [[Float]] {
        self.x = x
        return x.map { row in
            var result = [Float](repeating: 0.0, count: row.count)
            for j in 0..<row.count {
                result[j] = row[j] > 0 ? row[j] : alpha * (exp(row[j]) - 1)
            }
            return result
        }
    }
    
    func backward(grad: [[Float]]) -> [[Float]] {
        guard let x = self.x else { return [] }
        return zip(x, grad).map { (xRow, gradRow) in
            var result = [Float](repeating: 0.0, count: xRow.count)
            for j in 0..<xRow.count {
                result[j] = xRow[j] > 0 ? gradRow[j] : gradRow[j] * (alpha + forward(x: [[xRow[j]]])[0][0])
            }
            return result
        }
    }
}

class SELU: Activation {
    private let alpha: Float = 1.6732632423543772848170429916717
    private let lambda: Float = 1.0507009873554804934193349852946
    private var x: [[Float]]?
    
    func forward(x: [[Float]]) -> [[Float]] {
        self.x = x
        return x.map { row in
            var result = [Float](repeating: 0.0, count: row.count)
            for j in 0..<row.count {
                result[j] = lambda * (row[j] > 0 ? row[j] : alpha * (exp(row[j]) - 1))
            }
            return result
        }
    }
    
    func backward(grad: [[Float]]) -> [[Float]] {
        guard let x = self.x else { return [] }
        return zip(x, grad).map { (xRow, gradRow) in
            var result = [Float](repeating: 0.0, count: xRow.count)
            for j in 0..<xRow.count {
                let deriv = xRow[j] > 0 ? lambda : lambda * alpha * exp(xRow[j])
                result[j] = gradRow[j] * deriv
            }
            return result
        }
    }
}

class GELU: Activation {
    private var x: [[Float]]?
    
    func forward(x: [[Float]]) -> [[Float]] {
        self.x = x
        return x.map { row in
            var result = [Float](repeating: 0.0, count: row.count)
            let sqrtTwoOverPi = sqrt(2 / Float.pi)
            for j in 0..<row.count {
                let term = 1 + tanh(sqrtTwoOverPi * (row[j] + 0.044715 * pow(row[j], 3)))
                result[j] = 0.5 * row[j] * term
            }
            return result
        }
    }
    
    func backward(grad: [[Float]]) -> [[Float]] {
        guard let x = self.x else { return [] }
        return x.enumerated().map { (i, xRow) in
            var result = [Float](repeating: 0.0, count: xRow.count)
            let sqrtTwoOverPi = sqrt(2 / Float.pi)
            for j in 0..<xRow.count {
                let xCubed = pow(xRow[j], 3)
                let tanhVal = tanh(sqrtTwoOverPi * (xRow[j] + 0.044715 * xCubed))
                let sechVal = 1 - pow(tanhVal, 2)
                let term = 0.5 * tanhVal + (0.0356774 * xCubed + 0.797885 * xRow[j]) * sechVal * sqrtTwoOverPi
                result[j] = grad[i][j] * (0.5 + 0.5 * term)
            }
            return result
        }
    }
}

class Identity: Activation {
    func forward(x: [[Float]]) -> [[Float]] {
        return x
    }
    
    func backward(grad: [[Float]]) -> [[Float]] {
        return grad
    }
}

class LogSoftmax: Activation {
    private var x: [[Float]]?
    private var softmax: [[Float]]?
    
    func forward(x: [[Float]]) -> [[Float]] {
        self.x = x
        return x.map { row in
            var maxVal = row.max() ?? 0
            var shiftedRow = row.map { $0 - maxVal }
            var exps = [Float](repeating: 0.0, count: row.count)
            vvexpf(&exps, shiftedRow, [Int32(row.count)])
            let sumExps = exps.reduce(0, +)
            self.softmax = [exps.map { $0 / sumExps }]
            return self.softmax!.map { log($0) }[0]
        }
    }
    
    func backward(grad: [[Float]]) -> [[Float]] {
        guard let softmax = self.softmax, let x = self.x else { return [] }
        return zip(softmax, grad).map { (softmaxRow, gradRow) in
            let sumGradSoftmax = gradRow.reduce(0, +)
            return gradRow.map { $0 - sumGradSoftmax }
        }
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
