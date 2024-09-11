import Foundation
import Accelerate
import MLX
import Tqdm

//needed

// Data types
let dataType = DType.float32
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

var tokens = [padToken, sosToken, eosToken, unkToken]
var indexes = [padIndex, sosIndex, eosIndex, unkIndex]


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
        decoder.setOptimizer(optimizer: optimizer)
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
    
    func getPadMask(x: MLXArray) -> MLXArray {
        print("entered getPadMask")
        
        return (x .!= self.padIdx).asType(Int.self)[0..., .newAxis, 0...]
    }
    
    func getSubMask(x: MLXArray) -> MLXArray {
        
        print("entered getSubMask")
        
        let seqLen = x.shape[1]
        var subsequentMask = MLX.triu(MLX.ones([seqLen, seqLen]), k: 1).asType(Int.self)
        
        subsequentMask = MLX.logicalNot(subsequentMask)
        
        print("exited getSubMask")
        
        return subsequentMask
    }
    
    /*func forward(src: MLXArray, trg: MLXArray, training: Bool) -> (MLXArray, MLXArray) {
     print("entered forward")
     
     let srcvar = src.asType(dataType)
     let trgvar = trg.asType(dataType)
     
     print(srcvar[0])
     
     //currently wrong result:
     let srcMask = self.getPadMask(x: srcvar)
     
     let trgMask = getPadMask(x: trgvar) & getSubMask(x: trgvar)
     
     
     let encSrc = encoder.forward(src: srcvar, srcMask: srcMask, training: training)
     
     let (out, attention) = self.decoder.forward(trg: trgvar, trgMask: trgMask, src: encSrc, srcMask: srcMask, training: training)
     
     print("exited forward")
     
     return (out, attention)
     }*/
    
    func forward(src: MLXArray, trg: MLXArray, training: Bool) -> (MLXArray, MLXArray) {
        print("entered forward")
        
        let srcvar = src.asType(dataType)
        let trgvar = trg.asType(dataType)
        
        //print(srcvar[0])
        
        // Correct shape for srcMask
        let srcMask = self.getPadMask(x: srcvar)
        
        // Ensure trgMask is broadcastable: (batch_size, 1, seq_len) & (seq_len, seq_len)
        let padMask = getPadMask(x: trgvar)
        let subMask = getSubMask(x: trgvar)
        
        // Adjust trgMask shape: (batch_size, seq_len, seq_len)
        let trgMask = broadcast(padMask, to: [trgvar.shape[0], trgvar.shape[1], trgvar.shape[1]]) & broadcast(subMask, to: [trgvar.shape[0], trgvar.shape[1], trgvar.shape[1]])
        
        let encSrc = encoder.forward(src: srcvar, srcMask: srcMask, training: training)
        
        let (out, attention) = self.decoder.forward(trg: trgvar, trgMask: trgMask, src: encSrc, srcMask: srcMask, training: training)
        
        print("exited forward")
        
        return (out, attention)
    }
    
    
    func backward(error: MLXArray)-> MLXArray {
        print("entered backward")
        
        var errorvar = error
        errorvar = self.decoder.backward(error: errorvar)
        errorvar = self.encoder.backward(error: self.decoder.encoderError)
        
        print("exited backward")
        
        return errorvar
    }
    
    func updateWeights() {
        print("entered updateWeights")
        
        encoder.updateWeights()
        decoder.updateWeights()
        
        print("exited updateWeights")
        
    }
    
    func train(source: [MLXArray], target: [MLXArray], epoch: Int, epochs: Int) -> MLXArray {
        
        print("entered train")
        
        // Start timer
        let startTime = Date()
        
        var lossHistory : [Float] = []
        let totalBatches = source.count
        var epochLoss : MLXArray = []
        
        var zipped = zip(source, target).enumerated()
        
        let tqdmRange = TqdmSequence(sequence: zipped, description: "Training", unit: "batch", color: .cyan)
        
        for (batchNum, (sourceBatch, targetBatch)) in tqdmRange {
            print("Processing batch \(batchNum + 1)")
            
            //print(sourceBatch.shape)
            //print(targetBatch[0..., 0..<(targetBatch.shape[1] - 1)].shape)
            // Perform forward pass
            let (output, attention) = self.forward(src: sourceBatch, trg: targetBatch[0..., 0..<(targetBatch.shape[1] - 1)], training: true)
            //let (output, attention) = self.forward(src: sourceBatch, trg: targetBatch, training: true)
            
            //print(output.shape[0])
            //print(output.shape[1])
            //print(output.shape[2])
            // Reshape the output
            
            let _output = output.reshaped([output.shape[0] * output.shape[1], output.shape[2]])
            
            // Compute the loss and append to history
            var loss = self.lossFunction.loss(y: _output, t: targetBatch[0..., 1...].asType(DType.int32).flattened()).mean()
            
            lossHistory.append(loss.item(Float.self))
            //print("Computed loss for batch \(batchNum + 1): \(loss.item(Float.self))")
            
            /*for i in 0..<loss.count {
             lossHistory[i + loss.count] = loss[i]
             }*/
            
            // Compute the error for backpropagation
            let error = self.lossFunction.derivative(y: _output, t: targetBatch[0..., 1...].asType(DType.int32).flattened())
            
            
            // Perform backward pass and update weights
            self.backward(error: error.reshaped(output.shape))
            
            self.updateWeights()
            
            //print("got here")
            
            let latestLoss = lossHistory[lossHistory.count-1]
            tqdmRange.setDescription(description: "training | loss: \(latestLoss) | perplexity: \(exp(latestLoss)) | epoch \(epoch + 1)/\(epochs)")
            
            
            if batchNum == (totalBatches - 1) {
                epochLoss = MLX.mean(MLXArray(lossHistory))
                //print("Epoch \(epoch + 1) average loss: \(epochLoss.item(Float.self))")
                tqdmRange.setDescription(description: "training | avg loss: \(epochLoss.item(Float.self)) | avg perplexity: \(exp(epochLoss.item(Float.self))) | epoch \(epoch + 1)/\(epochs)")
                
            }
        }
        
        // End timer
        let endTime = Date()
        let timeInterval = endTime.timeIntervalSince(startTime)
        print("Training completed in \(timeInterval) seconds")
        //print(epochLoss)
        
        print("exited train")
        
        return epochLoss
    }
    
    func evaluate(source: [MLXArray], target: [MLXArray]) -> MLXArray {
        print("entered evaluate")
        
        
        // Start timer
        let startTime = Date()
        
        var lossHistory : [Float] = []
        let totalBatches = source.count
        var epochLoss : MLXArray = []
        
        var zipped =  zip(source, target).enumerated()
        
        let tqdmRange = TqdmSequence(sequence: zipped, description: "Testing", unit: "batch", color: .cyan)
                
        for (batchNum, (sourceBatch, targetBatch)) in tqdmRange {
            
            let (output, attention) = self.forward(src: sourceBatch, trg: targetBatch[0..., 0..<(targetBatch.shape[1] - 1)], training: false)
            
            let _output = output.reshaped([output.shape[0] * output.shape[1], output.shape[2]])
            
            let loss = self.lossFunction.loss(y: _output, t: targetBatch[0..., 1...].asType(DType.int32).flattened()).mean()
            
            
            lossHistory.append(loss.item(Float.self))

            
            let latestLoss = lossHistory[lossHistory.count - 1]
            tqdmRange.setDescription(description: "testing | loss: \(latestLoss) | perplexity: \(exp(latestLoss))")
            
            if batchNum == (source.count - 1) {
                
                epochLoss = MLX.mean(MLXArray(lossHistory))
                
                tqdmRange.setDescription(description: "testing | avg loss: \(epochLoss.item(Float.self)) | avg perplexity: \(exp(epochLoss.item(Float.self)))")
                
            }
        }
        
        // End timer
        let endTime = Date()
        let timeInterval = endTime.timeIntervalSince(startTime)
        print("Testing completed in \(timeInterval) seconds")
        
        print("exited evaluate")
        
        return epochLoss
        
    }
    
    func fit(trainData: ([MLXArray], [MLXArray]), valData: ([MLXArray], [MLXArray]), epochs: Int, saveEveryEpochs: Int, savePath: String?, validationCheck: Bool) -> ([MLXArray],[MLXArray]) {
        
        print("entered fit")
        
        setOptimizer()
        
        var bestValLoss = Float.infinity
        var trainLossHistory: [MLXArray] = []
        var valLossHistory: [MLXArray] = []
        
        let (trainSource, trainTarget) = trainData
        let (valSource, valTarget) = valData
        
        for epoch in 0..<epochs {
            
            let trainLoss = self.train(source: trainSource, target: trainTarget, epoch: epoch, epochs: epochs)
            trainLossHistory.append(trainLoss)
            
            let valLoss = self.evaluate(source: valSource, target: valTarget)
            valLossHistory.append(valLoss)
            
            if ((savePath != nil) && ((epoch + 1) % saveEveryEpochs == 0)) {
                if !validationCheck {
                    self.save(path: "\(savePath)/\(epoch + 1)")
                } else {
                    if valLossHistory[valLossHistory.count - 1].item(Float.self) < bestValLoss {
                        bestValLoss = valLossHistory[-1].item(Float.self)
                        self.save(path: "\(savePath)/\(epoch + 1)")
                    } else {
                        print("Current validation loss is higher than previous; Not saved")
                    }
                }
            }
        }
        
        print("exited fit")
        
        return (trainLossHistory, valLossHistory)
    }
    
    func predict(sentence: [Int], vocabs: ([String: Int], [String: Int]), maxLength: Int = 50) -> ([String], MLXArray) {
        
        // not completely correct:
        let srcIndices = [sosIndex] + sentence + [eosIndex]
        
        let src = MLXArray(srcIndices).reshaped([1,-1])
        let srcMask = self.getPadMask(x: src)
        
        let encSrc = self.encoder.forward(src: src, srcMask: srcMask, training: false)
        
        var trgIndices = [sosIndex]
        
        var (output, attention) : (MLXArray, MLXArray) = ([],[])
        
        for _ in 0..<maxLength{
            
            let trg = MLXArray(trgIndices).reshaped([1,-1])
            let trgMask = self.getPadMask(x: trg) & self.getSubMask(x: trg)
            
            (output, attention) = self.decoder.forward(trg: trg, trgMask: trgMask, src: encSrc, srcMask: srcMask, training: false)
            
            let trgIndx = output.argMax(axis: -1)[0..., -1]
            
            trgIndices.append(trgIndx.item(Int.self))
            
            
            if trgIndx.item(Int.self) == eosIndex || trgIndices.count >= maxLength {
                break
            }
            
        }
        
        let reversedVocab = Dictionary(uniqueKeysWithValues: vocabs.1.map { ($1, $0) })
        let decodedSentence = trgIndices.dropFirst().compactMap { reversedVocab[$0] }
        
        return (decodedSentence, attention[0])
    }
}
