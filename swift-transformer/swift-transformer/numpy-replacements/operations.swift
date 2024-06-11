import Accelerate
import Foundation

// 1. np.exp
func exp(_ array: [Float]) -> [Float] {
    var results = [Float](repeating: 0.0, count: array.count)
    vvexpf(&results, array, [Int32(array.count)])
    return results
}

// 2. np.tanh
func tanh(_ array: [Float]) -> [Float] {
    var results = [Float](repeating: 0.0, count: array.count)
    vvtanhf(&results, array, [Int32(array.count)])
    return results
}

// 3. np.power
func power(_ array: [Float], exponent: Float) -> [Float] {
    var results = [Float](repeating: 0.0, count: array.count)
    var exp = exponent
    vvpowf(&results, &exp, array, [Int32(array.count)])
    return results
}

// 4. np.max
func max(_ array: [Float]) -> Float {
    return array.max() ?? 0.0
}

// 5. np.sum
func sum(_ array: [Float]) -> Float {
    return array.reduce(0, +)
}

// 6. np.newaxis
extension Array {
    func newaxis() -> [[Element]] {
        return self.map { [$0] }
    }
}

// 7. np.tile
func tile(_ array: [Float], reps: Int) -> [Float] {
    return Array(repeating: array, count: reps).flatMap { $0 }
}

// 8. np.identity
func identity(_ n: Int) -> [[Float]] {
    return (0..<n).map { i in (0..<n).map { j in i == j ? 1.0 : 0.0 } }
}

// 9. np.ones
func ones(_ shape: (Int, Int)) -> [[Float]] {
    return Array(repeating: Array(repeating: 1.0, count: shape.1), count: shape.0)
}

// 10. np.int8
func int8(_ array: [Float]) -> [Int8] {
    return array.map { Int8($0) }
}

// 11. np.zeros
func zeros(_ shape: (Int, Int)) -> [[Float]] {
    return Array(repeating: Array(repeating: 0.0, count: shape.1), count: shape.0)
}

// 12. np.abs
func abs(_ array: [Float]) -> [Float] {
    return array.map { Swift.abs($0) }
}

// 13. np.log
func log(_ array: [Float]) -> [Float] {
    var results = [Float](repeating: 0.0, count: array.count)
    vvlogf(&results, array, [Int32(array.count)])
    return results
}

// 14. np.where
func whereCondition(_ condition: [Bool], _ x: [Float], _ y: [Float]) -> [Float] {
    return zip(condition, zip(x, y)).map { $0 ? $1.0 : $1.1 }
}

// 15. np.ndarray.astype
// Swift does not have a direct method for casting arrays but manual casting can be done

// 16. np.asarray
func asarray(_ array: [Float]) -> [Float] {
    return array
}

// 17. np.sqrt
func sqrt(_ array: [Float]) -> [Float] {
    var results = [Float](repeating: 0.0, count: array.count)
    vvsqrtf(&results, array, [Int32(array.count)])
    return results
}

// 18. np.array_split
func arraySplit(_ array: [Float], sections: Int) -> [[Float]] {
    let chunkSize = (array.count + sections - 1) / sections
    return stride(from: 0, to: array.count, by: chunkSize).map {
        Array(array[$0..<min($0 + chunkSize, array.count)])
    }
}

// 19. np.arange
func arange(start: Float, stop: Float, step: Float) -> [Float] {
    return stride(from: start, to: stop, by: step).map { $0 }
}

// 20. np.triu
func triu(_ matrix: [[Float]]) -> [[Float]] {
    let n = matrix.count
    var result = matrix
    for i in 0..<n {
        for j in 0..<i {
            result[i][j] = 0.0
        }
    }
    return result
}

// 21. np.float32
func float32(_ array: [Float]) -> [Float] {
    return array.map { Float($0) }
}

// 22. np.int32
func int32(_ array: [Float]) -> [Int32] {
    return array.map { Int32($0) }
}

// 23. np.random.randint
func randint(_ low: Int, _ high: Int, _ size: Int) -> [Int] {
    return (0..<size).map { _ in Int.random(in: low..<high) }
}

// 24. np.random.uniform
func uniform(_ low: Float, _ high: Float, _ size: Int) -> [Float] {
    return (0..<size).map { _ in Float.random(in: low..<high) }
}

// 25. np.zeros_like
func zerosLike(_ array: [Float]) -> [Float] {
    return [Float](repeating: 0.0, count: array.count)
}

// 26. np.dot
func dot(_ a: [Float], _ b: [Float]) -> Float {
    return zip(a, b).map(*).reduce(0, +)
}

// 27. np.random.binomial
func binomial(n: Int, p: Float, size: Any) -> Any {
    if let sizeInt = size as? Int {
        return (0..<sizeInt).map { _ in (0..<n).reduce(0) { acc, _ in acc + (Float.random(in: 0...1) < p ? 1 : 0) } }
    } else if let sizeArray = size as? [Int], !sizeArray.isEmpty {
        return generateBinomialArray(n: n, p: p, sizeArray: sizeArray)
    } else {
        fatalError("Invalid size parameter")
    }
}

func generateBinomialArray(n: Int, p: Float, sizeArray: [Int]) -> Any {
    if sizeArray.count == 1 {
        return binomial(n: n, p: p, size: sizeArray[0])
    } else {
        return (0..<sizeArray[0]).map { _ in generateBinomialArray(n: n, p: p, sizeArray: Array(sizeArray.dropFirst())) }
    }
}


// 28. np.random.normal
func normal(mean: Float, stddev: Float, size: Int) -> [Float] {
    return (0..<size).map { _ in Float.random(in: 0...1) }.map { z in
        let u1 = z
        let u2 = Float.random(in: 0...1)
        let r = sqrt(-2.0 * log(u1))
        let theta = 2.0 * Float.pi * u2
        return r * cos(theta) * stddev + mean
    }
}

// 29. np.equal
func equal(_ a: [Float], _ b: [Float]) -> [Bool] {
    return zip(a, b).map(==)
}

// 30. np.matmul
func matmul(_ a: [[Float]], _ b: [[Float]]) -> [[Float]] {
    let rowsA = a.count
    let colsA = a[0].count
    let colsB = b[0].count
    
    var result: [[Float]] = Array(repeating: Array(repeating: 0.0, count: colsB), count: rowsA)
    
    for i in 0..<rowsA {
        for j in 0..<colsB {
            for k in 0..<colsA {
                result[i][j] += a[i][k] * b[k][j]
            }
        }
    }
    
    return result
}

// 31. np.mean
func mean(_ array: [Float]) -> Float {
    return sum(array) / Float(array.count)
}

// 32. np.var
func variance(_ array: [Float]) -> Float {
    let m = mean(array)
    return mean(array.map { pow($0 - m, 2) })
}

// 33. np.expand_dims
func expandDims(_ array: [Float], axis: Int) -> [[Float]] {
    if axis == 0 {
        return [array]
    } else {
        return array.map { [$0] }
    }
}

// 34. np.sin
func sin(_ array: [Float]) -> [Float] {
    var results = [Float](repeating: 0.0, count: array.count)
    vvsinf(&results, array, [Int32(array.count)])
    return results
}

// 35. np.cos
func cos(_ array: [Float]) -> [Float] {
    var results = [Float](repeating: 0.0, count: array.count)
    vvcosf(&results, array, [Int32(array.count)])
    return results
}

// 36. transpose
extension Array where Element == [Float] {
    func transpose() -> [[Float]] {
        var result = [[Float]](repeating: [Float](repeating: 0.0, count: self.count), count: self[0].count)
        for i in 0..<self.count {
            for j in 0..<self[i].count {
                result[j][i] = self[i][j]
            }
        }
        return result
    }
    
        
        /*func mean(axis: Int) -> [Float] {
            var result = [Float](repeating: 0.0, count: self[0].count)
            for i in 0..<self.count {
                for j in 0..<self[i].count {
                    result[j] += self[i][j]
                }
            }
            return result.map { $0 / Float(self.count) }
        }*/
}

// 36. shape
func shape(_ array: Any) -> [Int] {
    if let array = array as? [Any] {
        if let firstElement = array.first {
            return [array.count] + shape(firstElement)
        } else {
            return [array.count]
        }
    } else {
        return []
    }
}
