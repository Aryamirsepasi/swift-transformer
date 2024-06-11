// transformer.swift
import Foundation
import Accelerate

// Data types
let dataType = Float.self
let batchSize = 128

// Special tokens and their indices
let padToken = "<pad>"
let sosToken = "<sos>"
let eosToken = "<eos>"
let unkToken = "<unk>"

let padIndex = 0
let sosIndex = 1
let eosIndex = 2
let unkIndex = 3

// Prepare data
let dataPreparator = DataPreparator(tokens: [padToken, sosToken, eosToken, unkToken], indexes: [padIndex, sosIndex, eosIndex, unkIndex])
let (trainData, testData, valData) = dataPreparator.prepareData(path: "dataset/", batchSize: batchSize, minFreq: 2)
let (source, target) = trainData

// Define Seq2Seq class
class Seq2Seq {
    var encoder: Encoder
    var decoder: Decoder
    var padIdx: Int
    var optimizer: Optimizer
    var lossFunction: LossFunction
    
    init(encoder: Encoder, decoder: Decoder, padIdx: Int) {
        self.encoder = encoder
        self.decoder = decoder
        self.padIdx = padIdx
        self.optimizer = Adam()
        self.lossFunction = CrossEntropy(ignoreIndex: padIdx)
    }
    
    func setOptimizer() {
        encoder.setOptimizer(optimizer)
        decoder.setOptimizer(optimizer)
    }
    
    func compile(optimizer: Optimizer, lossFunction: LossFunction) {
        self.optimizer = optimizer
        self.lossFunction = lossFunction
    }
    
    func load(path: String) {
        let picklePath = "\(path)/encoder.pkl"
        let picklePath2 = "\(path)/decoder.pkl"
        
        do {
            let pickleBinaryData = try Data(contentsOf: URL(fileURLWithPath: picklePath))
            let pickleBinaryData2 = try Data(contentsOf: URL(fileURLWithPath: picklePath2))
            
            self.encoder = try NSKeyedUnarchiver.unarchiveTopLevelObjectWithData(pickleBinaryData) as! Encoder
            self.decoder = try NSKeyedUnarchiver.unarchiveTopLevelObjectWithData(pickleBinaryData2) as! Decoder
        } catch {
            print("Error loading from \(path): \(error)")
        }
        print("Loaded from \"\(path)\"")
    }
    
    func save(path: String) {
        if !FileManager.default.fileExists(atPath: path) {
            do {
                try FileManager.default.createDirectory(atPath: path, withIntermediateDirectories: true, attributes: nil)
            } catch {
                print("Error creating directory at \(path): \(error)")
                return
            }
        }
        
        let picklePath = "\(path)/encoder.pkl"
        let picklePath2 = "\(path)/decoder.pkl"
        
        do {
            let encoderData = try NSKeyedArchiver.archivedData(withRootObject: self.encoder as Any)
            let decoderData = try NSKeyedArchiver.archivedData(withRootObject: self.decoder as Any)
            
            try encoderData.write(to: URL(fileURLWithPath: picklePath))
            try decoderData.write(to: URL(fileURLWithPath: picklePath2))
        } catch {
            print("Error saving to \(path): \(error)")
        }
        
        print("Saved to \"\(path)\"")
    }
    
    func getPadMask(x: [[Int]]) -> [[[Float]]] {
        return x.map { sequence in
            sequence.map { $0 != self.padIdx ? 1.0 : 0.0 }
        }.map { $0.map { [$0] } }
    }
    
    func getSubMask(size: Int) -> [[[Float]]] {
        var mask: [[[Float]]] = Array(repeating: Array(repeating: Array(repeating: 0.0, count: size), count: size), count: 1)
        for i in 0..<size {
            for j in 0..<size {
                mask[0][i][j] = j > i ? 0.0 : 1.0
            }
        }
        return mask
    }
    
    func forward(src: [[Int]], trg: [[Int]], training: Bool) -> ([[[Float]]], [[[Float]]]) {
        let srcMask = getPadMask(x: src)
        let trgMask = trg.map { getPadMask(x: [$0]) }.flatMap { $0 }
        let subMask = getSubMask(size: trg[0].count)
        
        let flatSrc = src.flatMap { $0 }
        let flatSrcMask = srcMask.flatMap { $0.flatMap { $0 } }
        
        let encSrc = encoder.forward(src: flatSrc, srcMask: flatSrcMask, training: training)
        
        var allOutputs: [[[Float]]] = []
        var allAttentions: [[[Float]]] = []
        
        for i in 0..<src.count {
            let (output, attention) = decoder.forward(
                trg: [trg[i]],
                trgMask: [trgMask[i]],
                src: encSrc,
                srcMask: [srcMask[i]],
                training: training
            )
            allOutputs.append(output)
            allAttentions.append(attention)
        }
        
        return (allOutputs, allAttentions)
    }
    
    func backward(error: [[Float]]) {
        var allEncoderErrors: [[Float]] = []
        // TODO: Implement the backward function.
    }
    
    func train(source: [[Int]], target: [[Int]], epoch: Int, epochs: Int) -> Float {
        var lossHistory: [Float] = []
        
        for (sourceBatch, targetBatch) in zip(source, target) {
            let (output, _) = forward(src: [sourceBatch], trg: [targetBatch.dropLast()], training: true)
            let outputFlat = output.flatMap { $0.flatMap { $0 } } // Flatten to [Float]
            let targetFlat = targetBatch.dropFirst().flatMap { Float($0) } // Flatten to [Float]
            
            let loss = lossFunction.loss(y: outputFlat, t: targetFlat)
            lossHistory.append(loss.reduce(0, +) / Float(loss.count))
            let error = lossFunction.derivative(y: outputFlat, t: targetFlat)
            
            backward(error: [error])
            updateWeights()
        }
        
        let epochLoss = lossHistory.reduce(0, +) / Float(lossHistory.count)
        return epochLoss
    }
    
    func evaluate(source: [[Int]], target: [[Int]]) -> Float {
        var lossHistory: [Float] = []
        
        for (sourceBatch, targetBatch) in zip(source, target) {
            let (output, _) = forward(src: [sourceBatch], trg: [targetBatch.dropLast()], training: false)
            let outputFlat = output.flatMap { $0.flatMap { $0 } } // Flatten to [Float]
            let targetFlat = targetBatch.dropFirst().flatMap { Float($0) } // Flatten to [Float]
            
            let loss = lossFunction.loss(y: outputFlat, t: targetFlat)
            lossHistory.append(loss.reduce(0, +) / Float(loss.count))
        }
        
        let epochLoss = lossHistory.reduce(0, +) / Float(lossHistory.count)
        return epochLoss
    }
    
    func fit(trainData: ([[Int]], [[Int]]), valData: ([[Int]], [[Int]]), epochs: Int, saveEveryEpochs: Int, savePath: String?, validationCheck: Bool) -> ([Float], [Float]) {
        setOptimizer()
        
        var bestValLoss = Float.infinity
        var trainLossHistory: [Float] = []
        var valLossHistory: [Float] = []
        
        let (trainSource, trainTarget) = trainData
        let (valSource, valTarget) = valData
        
        for epoch in 0..<epochs {
            let trainLoss = train(source: trainSource, target: trainTarget, epoch: epoch, epochs: epochs)
            trainLossHistory.append(trainLoss)
            
            let valLoss = evaluate(source: valSource, target: valTarget)
            valLossHistory.append(valLoss)
            
            if let savePath = savePath, (epoch + 1) % saveEveryEpochs == 0 {
                if !validationCheck {
                    save(path: "\(savePath)/\(epoch + 1)")
                } else {
                    if valLoss < bestValLoss {
                        bestValLoss = valLoss
                        save(path: "\(savePath)/\(epoch + 1)")
                    } else {
                        print("Current validation loss is higher than previous; Not saved")
                    }
                }
            }
        }
        
        return (trainLossHistory, valLossHistory)
    }
    
    func predict(sentence: [Int], max_length: Int = 50) -> ([Int], [[Float]]) {
        var srcIndices = [sosIndex] + sentence + [eosIndex]
        var trgIndices = [sosIndex]
        
        var attentionWeights = [[Float]]()
        while trgIndices.count < max_length {
            let (output, attention) = forward(src: [srcIndices], trg: [trgIndices], training: false)
            guard let lastOutput = output.last?.last else { continue }
            let predictedIndex = lastOutput.indices.max(by: { lastOutput[$0] < lastOutput[$1] }) ?? eosIndex
            trgIndices.append(predictedIndex)
            attentionWeights.append(attention.last?.last ?? [])

            if predictedIndex == eosIndex { break }
        }
        
        return (trgIndices, attentionWeights)
    }
    
    func updateWeights() {
        encoder.updateWeights()
        decoder.updateWeights()
    }
}
