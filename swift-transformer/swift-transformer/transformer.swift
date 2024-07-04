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
        self.lossFunction = CrossEntropy()
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
        let encoderPath = "\(path)/encoder.pkl"
        let decoderPath = "\(path)/decoder.pkl"
        
        do {
            let encoderData = try Data(contentsOf: URL(fileURLWithPath: encoderPath))
            let decoderData = try Data(contentsOf: URL(fileURLWithPath: decoderPath))
            
            self.encoder = try NSKeyedUnarchiver.unarchiveTopLevelObjectWithData(encoderData) as! Encoder
            self.decoder = try NSKeyedUnarchiver.unarchiveTopLevelObjectWithData(decoderData) as! Decoder
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
        
        let encoderPath = "\(path)/encoder.pkl"
        let decoderPath = "\(path)/decoder.pkl"
        
        do {
            let encoderData = try NSKeyedArchiver.archivedData(withRootObject: self.encoder)
            let decoderData = try NSKeyedArchiver.archivedData(withRootObject: self.decoder)
            
            try encoderData.write(to: URL(fileURLWithPath: encoderPath))
            try decoderData.write(to: URL(fileURLWithPath: decoderPath))
        } catch {
            print("Error saving to \(path): \(error)")
        }
        
        print("Saved to \"\(path)\"")
    }
    
    func getPadMask(x: [[Float]]) -> [[[Float]]] {
        return x.map { sequence in
            sequence.map { Int($0) != padIdx ? 1.0 : 0.0 }
        }.map { [$0] }
    }
    
    func getSubMask(x: [[Float]]) -> [[Float]] {
        let seqLen = x[0].count
        var subsequentMask = [[Float]](repeating: [Float](repeating: 0, count: seqLen), count: seqLen)
        
        for i in 0..<seqLen {
            for j in 0..<seqLen {
                if i < j {
                    subsequentMask[i][j] = 1
                } else {
                    subsequentMask[i][j] = 0
                }
            }
        }
        
        for i in 0..<seqLen {
            for j in 0..<seqLen {
                subsequentMask[i][j] = 1 - subsequentMask[i][j]
            }
        }
        
        return subsequentMask
    }
    
    func forward(src: [[Float]], trg: [[Float]], training: Bool) -> ([[[Float]]], [[[[Float]]]]) {
        
        let src = src
        let trg = trg
        
        let srcMask = getPadMask(x: src)
        var trgMask = getPadMask(x: trg)
        let subMask = getSubMask(x: trg)
        
        let trgMaskCount = trgMask.count
        let trgMaskSeqLen = trgMask[0][0].count
        let subMaskSeqLen = subMask.count
        
        /*for i in 0..<trgMaskCount {
            for j in 0..<min(trgMaskSeqLen, subMaskSeqLen) {
                for k in 0..<min(trgMaskSeqLen, subMask[j].count) {
                    trgMask[i][0][j] *= subMask[j][k]
                }
            }
        }*/
        
        let encSrc = encoder.forward(src: src, srcMask: srcMask, training: training)
        
        var allOutputs: [[[Float]]] = []
        var allAttentions: [[[[Float]]]] = []
        
        for i in 0..<src.count {
            let (output, attention) = decoder.forward(
                trg: [trg[i]],
                trgMask: trgMask,
                src: encSrc,
                srcMask: srcMask,
                training: training
            )
            allOutputs.append(contentsOf: output)
            allAttentions.append(attention)
        }
        
        return (allOutputs, allAttentions)
    }
    
    func backward(error: [[[Float]]]) {
        var decoderError = error
        for layer in decoder.layers.reversed() {
            let (layerError, encError) = layer.backward(decoderError)
            decoderError = layerError + encError
        }
        
        let encoderError = decoderError.map { $0.map { $0.map { $0 * decoder.scale } } }
        encoder.backward(error: encoderError)
    }
    
    func updateWeights() {
        encoder.updateWeights()
        decoder.updateWeights()
    }
    
    func train(source: [[[Float]]], target: [[[Float]]], epoch: Int, epochs: Int) -> Float {
        var lossHistory: [Float] = []
        
        for (sourceBatch, targetBatch) in zip(source, target) {
            let (output, _) = forward(src: sourceBatch, trg: targetBatch.dropLast(), training: true)
            let outputFlat = output.flatMap { $0.flatMap { $0 } }
            let targetFlat = targetBatch.dropFirst().flatMap { $0 }
            
            let loss = lossFunction.loss(y: [outputFlat], t: targetFlat)
            lossHistory.append(loss.reduce(0, +) / Float(loss.count))
            let error = lossFunction.derivative(y: [outputFlat], t: targetFlat)
            
            backward(error: [error])
            updateWeights()
        }
        
        let epochLoss = lossHistory.reduce(0, +) / Float(lossHistory.count)
        return epochLoss
    }
    
    func evaluate(source: [[[Float]]], target: [[[Float]]]) -> Float {
        var lossHistory: [Float] = []
        
        for (sourceBatch, targetBatch) in zip(source, target) {
            let (output, _) = forward(src: sourceBatch, trg: targetBatch.dropLast(), training: false)
            let outputFlat = output.flatMap { $0.flatMap { $0 } }
            let targetFlat = targetBatch.dropFirst().flatMap { $0 }
            
            let loss = lossFunction.loss(y: [outputFlat], t: targetFlat)
            lossHistory.append(loss.reduce(0, +) / Float(loss.count))
        }
        
        let epochLoss = lossHistory.reduce(0, +) / Float(lossHistory.count)
        return epochLoss
    }
    
    func fit(trainData: ([[[Float]]], [[[Float]]]), valData: ([[[Float]]], [[[Float]]]), epochs: Int, saveEveryEpochs: Int, savePath: String?, validationCheck: Bool) -> ([Float], [Float]) {
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
    
    func predict(sentence: [Int], vocabs: ([String: Int], [String: Int]), maxLength: Int = 50) -> ([String], [[Float]]) {
        var srcIndices = [sosIndex] + sentence + [eosIndex]
        var trgIndices = [sosIndex]
        
        var attentionWeights = [[Float]]()
        while trgIndices.count < maxLength {
            let (output, attention) = forward(src: [srcIndices.map { Float($0) }], trg: [trgIndices.map { Float($0) }], training: false)
            guard let lastOutput = output.last?.last else { continue }
            let predictedIndex = (lastOutput.enumerated().max(by: { $0.element < $1.element })?.offset) ?? eosIndex
            trgIndices.append(predictedIndex)
            attentionWeights.append(attention.last?.last?.flatMap { $0 } ?? [])
            
            if predictedIndex == eosIndex { break }
        }
        
        let reversedVocab = Dictionary(uniqueKeysWithValues: vocabs.1.map { ($1, $0) })
        let decodedSentence = trgIndices.dropFirst().compactMap { reversedVocab[$0] }
        
        return (decodedSentence, attentionWeights)
    }
}
