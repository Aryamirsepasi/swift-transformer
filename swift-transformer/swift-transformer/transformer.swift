import Foundation
import Accelerate
import MLX
import Tqdm
import MLXNN
import MLXRandom


//needed

// Data types
let dataType = DType.float32
let batchSize = 128

/// Special tokens and their indices
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


/// Main Sequence-to-Sequence model class implementing a transformer architecture for translation
class Seq2Seq {
    var encoder: Encoder
    var decoder: Decoder
    var padIdx: Int
    var optimizer: Optimizer
    var lossFunction: LossFunction
    
    /// Initialize a new Seq2Seq model
    /// - Parameters:
    ///   - encoder: Transformer encoder for processing source sentences
    ///   - decoder: Transformer decoder for generating target translations
    ///   - padIdx: Index used for padding tokens in sequences
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
    
    /// Load model parameters from disk
    /// - Parameter path: Directory path where the model is saved
    func load(path: String) {
        do {
            // Load encoder parameters
            let encoderURL = URL(fileURLWithPath: "\(path)/encoder.safetensors")
            let encoderParams = try MLX.loadArrays(url: encoderURL)
            
            // Load decoder parameters
            let decoderURL = URL(fileURLWithPath: "\(path)/decoder.safetensors")
            let decoderParams = try MLX.loadArrays(url: decoderURL)
            
            // Restore encoder parameters
            self.encoder.tokenEmbedding.w = encoderParams["token_embedding"]!
            self.encoder.positionEmbedding.pe = encoderParams["position_embedding"]!
            
            // Restore encoder layers
            for (i, layer) in self.encoder.layers.enumerated() {
                layer.selfAttention.KLinear.w = encoderParams["layer_\(i)_self_attention_k"]!
                layer.selfAttention.QLinear.w = encoderParams["layer_\(i)_self_attention_q"]!
                layer.selfAttention.VLinear.w = encoderParams["layer_\(i)_self_attention_v"]!
                layer.selfAttention.OLinear.w = encoderParams["layer_\(i)_self_attention_o"]!
                
                if layer.selfAttention.KLinear.useBias {
                    layer.selfAttention.KLinear.b = encoderParams["layer_\(i)_self_attention_k_bias"]!
                    layer.selfAttention.QLinear.b = encoderParams["layer_\(i)_self_attention_q_bias"]!
                    layer.selfAttention.VLinear.b = encoderParams["layer_\(i)_self_attention_v_bias"]!
                    layer.selfAttention.OLinear.b = encoderParams["layer_\(i)_self_attention_o_bias"]!
                }
            }
            
            // Restore decoder parameters
            self.decoder.tokenEmbedding.w = decoderParams["token_embedding"]!
            self.decoder.positionEmbedding.pe = decoderParams["position_embedding"]!
            
            // Restore decoder layers
            for (i, layer) in self.decoder.layers.enumerated() {
                layer.selfAttention.KLinear.w = decoderParams["layer_\(i)_self_attention_k"]!
                layer.selfAttention.QLinear.w = decoderParams["layer_\(i)_self_attention_q"]!
                layer.selfAttention.VLinear.w = decoderParams["layer_\(i)_self_attention_v"]!
                layer.selfAttention.OLinear.w = decoderParams["layer_\(i)_self_attention_o"]!
                
                layer.encoderAttention.KLinear.w = decoderParams["layer_\(i)_enc_attention_k"]!
                layer.encoderAttention.QLinear.w = decoderParams["layer_\(i)_enc_attention_q"]!
                layer.encoderAttention.VLinear.w = decoderParams["layer_\(i)_enc_attention_v"]!
                layer.encoderAttention.OLinear.w = decoderParams["layer_\(i)_enc_attention_o"]!
                
                if layer.selfAttention.KLinear.useBias {
                    layer.selfAttention.KLinear.b = decoderParams["layer_\(i)_self_attention_k_bias"]!
                    layer.selfAttention.QLinear.b = decoderParams["layer_\(i)_self_attention_q_bias"]!
                    layer.selfAttention.VLinear.b = decoderParams["layer_\(i)_self_attention_v_bias"]!
                    layer.selfAttention.OLinear.b = decoderParams["layer_\(i)_self_attention_o_bias"]!
                    
                    layer.encoderAttention.KLinear.b = decoderParams["layer_\(i)_enc_attention_k_bias"]!
                    layer.encoderAttention.QLinear.b = decoderParams["layer_\(i)_enc_attention_q_bias"]!
                    layer.encoderAttention.VLinear.b = decoderParams["layer_\(i)_enc_attention_v_bias"]!
                    layer.encoderAttention.OLinear.b = decoderParams["layer_\(i)_enc_attention_o_bias"]!
                }
            }
            
            print("Successfully loaded model from \"\(path)\"")
        } catch {
            print("Error loading model parameters: \(error)")
        }
    }
    
    /// Save model parameters to disk
    /// - Parameter path: Directory path to save the model
    /// - Throws: FileManager and JSON encoding errors
    func save(path: String) {
        if !FileManager.default.fileExists(atPath: path) {
            do {
                try FileManager.default.createDirectory(atPath: path, withIntermediateDirectories: true, attributes: nil)
            } catch {
                print("Error creating directory at \(path): \(error)")
                return
            }
        }
        
        // Create dictionaries to store parameters
        var encoderParams: [String: MLXArray] = [:]
        var decoderParams: [String: MLXArray] = [:]
        
        // Save encoder parameters
        encoderParams["token_embedding"] = self.encoder.tokenEmbedding.w
        encoderParams["position_embedding"] = self.encoder.positionEmbedding.pe
        
        // Save encoder layers
        for (i, layer) in self.encoder.layers.enumerated() {
            encoderParams["layer_\(i)_self_attention_k"] = layer.selfAttention.KLinear.w
            encoderParams["layer_\(i)_self_attention_q"] = layer.selfAttention.QLinear.w
            encoderParams["layer_\(i)_self_attention_v"] = layer.selfAttention.VLinear.w
            encoderParams["layer_\(i)_self_attention_o"] = layer.selfAttention.OLinear.w
            
            if layer.selfAttention.KLinear.useBias {
                encoderParams["layer_\(i)_self_attention_k_bias"] = layer.selfAttention.KLinear.b
                encoderParams["layer_\(i)_self_attention_q_bias"] = layer.selfAttention.QLinear.b
                encoderParams["layer_\(i)_self_attention_v_bias"] = layer.selfAttention.VLinear.b
                encoderParams["layer_\(i)_self_attention_o_bias"] = layer.selfAttention.OLinear.b
            }
        }
        
        // Save decoder parameters
        decoderParams["token_embedding"] = self.decoder.tokenEmbedding.w
        decoderParams["position_embedding"] = self.decoder.positionEmbedding.pe
        
        // Save decoder layers
        for (i, layer) in self.decoder.layers.enumerated() {
            decoderParams["layer_\(i)_self_attention_k"] = layer.selfAttention.KLinear.w
            decoderParams["layer_\(i)_self_attention_q"] = layer.selfAttention.QLinear.w
            decoderParams["layer_\(i)_self_attention_v"] = layer.selfAttention.VLinear.w
            decoderParams["layer_\(i)_self_attention_o"] = layer.selfAttention.OLinear.w
            
            decoderParams["layer_\(i)_enc_attention_k"] = layer.encoderAttention.KLinear.w
            decoderParams["layer_\(i)_enc_attention_q"] = layer.encoderAttention.QLinear.w
            decoderParams["layer_\(i)_enc_attention_v"] = layer.encoderAttention.VLinear.w
            decoderParams["layer_\(i)_enc_attention_o"] = layer.encoderAttention.OLinear.w
            
            if layer.selfAttention.KLinear.useBias {
                decoderParams["layer_\(i)_self_attention_k_bias"] = layer.selfAttention.KLinear.b
                decoderParams["layer_\(i)_self_attention_q_bias"] = layer.selfAttention.QLinear.b
                decoderParams["layer_\(i)_self_attention_v_bias"] = layer.selfAttention.VLinear.b
                decoderParams["layer_\(i)_self_attention_o_bias"] = layer.selfAttention.OLinear.b
                
                decoderParams["layer_\(i)_enc_attention_k_bias"] = layer.encoderAttention.KLinear.b
                decoderParams["layer_\(i)_enc_attention_q_bias"] = layer.encoderAttention.QLinear.b
                decoderParams["layer_\(i)_enc_attention_v_bias"] = layer.encoderAttention.VLinear.b
                decoderParams["layer_\(i)_enc_attention_o_bias"] = layer.encoderAttention.OLinear.b
            }
        }
        
        do {
            // Save encoder parameters
            let encoderURL = URL(fileURLWithPath: "\(path)/encoder.safetensors")
            try MLX.save(arrays: encoderParams, url: encoderURL)
            
            // Save decoder parameters
            let decoderURL = URL(fileURLWithPath: "\(path)/decoder.safetensors")
            try MLX.save(arrays: decoderParams, url: decoderURL)
            
            print("Successfully saved model to \"\(path)\"")
        } catch {
            print("Error saving model parameters: \(error)")
        }
    }
    
    /// Creates a padding mask to handle variable-length sequences
    /// - Parameter x: Input tensor of token indices
    /// - Returns: A boolean mask tensor where True indicates non-padding positions
    ///
    func getPadMask(x: MLXArray) -> MLXArray {
        
        return (x .!= self.padIdx).asType(Int.self)[0..., .newAxis, 0...]
    }
    
    
    /// Creates a subsequent (causal) mask for decoder self-attention
    /// - Parameter x: Input tensor
    /// - Returns: A triangular mask to prevent attending to future positions
    ///
    func getSubMask(x: MLXArray) -> MLXArray {
        
        let seqLen = x.shape[1]
        
        /// Create upper triangular matrix for masking future positions
        var subsequentMask = MLX.triu(MLX.ones([seqLen, seqLen]), k: 1).asType(Int.self)
        
        subsequentMask = MLX.logicalNot(subsequentMask, stream: .gpu)
        
        return subsequentMask
    }
    
    /// Forward pass through the entire model
    /// - Parameters:
    ///   - src: Source sequence input tensor
    ///   - trg: Target sequence input tensor
    ///   - training: Boolean indicating training mode
    /// - Returns: Tuple of (output logits, attention weights)
    
    func forward(src: MLXArray, trg: MLXArray, training: Bool) -> (MLXArray, MLXArray) {
        return autoreleasepool {
            let srcvar = src.asType(dataType)
            let trgvar = trg.asType(dataType)
            
            // Create source padding mask
            let srcMask = self.getPadMask(x: srcvar)
            
            // Create target mask that combines padding and subsequent mask
            let trgPadMask = getPadMask(x: trgvar)
            let trgSubMask = getSubMask(x: trgvar)
            
            // Combine masks properly
            let trgMask = broadcast(trgPadMask, to: [trgvar.shape[0], trgvar.shape[1], trgvar.shape[1]], stream: .gpu)
            & broadcast(trgSubMask, to: [trgvar.shape[0], trgvar.shape[1], trgvar.shape[1]], stream: .gpu)
            
            // Encoder forward pass
            let encSrc = self.encoder.forward(src: srcvar, srcMask: srcMask, training: training)
            
            // Decoder forward pass
            let (out, attention) = self.decoder.forward(trg: trgvar, trgMask: trgMask, src: encSrc, srcMask: srcMask, training: training)
            
            return (out, attention)
        }
    }
    
    
    func backward(error: MLXArray)-> MLXArray {
        
        var errorvar = error
        errorvar = self.decoder.backward(error: errorvar)
        errorvar = self.encoder.backward(error: self.decoder.encoderError)
        
        
        return errorvar
    }
    
    func updateWeights() {
        
        encoder.updateWeights()
        decoder.updateWeights()
        
    }
    
    func train(source: [MLXArray], target: [MLXArray], epoch: Int, epochs: Int) -> MLXArray {
        
        // Start timer
        let startTime = Date()
        
        var lossHistory : [Float] = []
        let totalBatches = source.count
        var epochLoss : MLXArray = []
        
        let zipped = zip(source, target).enumerated()
        
        let tqdmRange = TqdmSequence(sequence: zipped, description: "Training", unit: "batch", color: .cyan)
        
        for (batchNum, (sourceBatch, targetBatch)) in tqdmRange {
            autoreleasepool {
                
                // Perform forward pass
                let (output, attention) = self.forward(src: sourceBatch, trg: targetBatch[0..., 0..<(targetBatch.shape[1] - 1)], training: true)
                
                let _output = output.reshaped([output.shape[0] * output.shape[1], output.shape[2]], stream: .gpu)
                
                // Compute the loss and append to history
                var loss = self.lossFunction.loss(y: _output, t: targetBatch[0..., 1...].asType(DType.int32).flattened()).mean()
                
                lossHistory.append(loss.item(Float.self))
                
                // Compute the error for backpropagation
                let error = self.lossFunction.derivative(y: _output, t: targetBatch[0..., 1...].asType(DType.int32).flattened())
                
                
                // Perform backward pass and update weights
                self.backward(error: error.reshaped(output.shape))
                
                self.updateWeights()
                
                let latestLoss = lossHistory[lossHistory.count-1]
                tqdmRange.setDescription(description: "training | loss: \(latestLoss) | perplexity: \(exp(latestLoss)) | epoch \(epoch + 1)/\(epochs)")
                
                
                if batchNum == (totalBatches - 1) {
                    epochLoss = MLX.mean(MLXArray(lossHistory))
                    tqdmRange.setDescription(description: "training | avg loss: \(epochLoss.item(Float.self)) | avg perplexity: \(exp(epochLoss.item(Float.self))) | epoch \(epoch + 1)/\(epochs)")
                    
                }
                
                GPU.clearCache()
            }
        }
        
        // End timer
        let endTime = Date()
        let timeInterval = endTime.timeIntervalSince(startTime)
        print("Training completed in \(timeInterval) seconds")
        
        return epochLoss
    }
    
    func evaluate(source: [MLXArray], target: [MLXArray]) -> MLXArray {
        
        // Start timer
        let startTime = Date()
        
        var lossHistory : [Float] = []
        let totalBatches = source.count
        var epochLoss : MLXArray = []
        
        let zipped =  zip(source, target).enumerated()
        
        let tqdmRange = TqdmSequence(sequence: zipped, description: "Testing", unit: "batch", color: .cyan)
        
        
        for (batchNum, (sourceBatch, targetBatch)) in tqdmRange {
            autoreleasepool {
                
                let (output, attention) = self.forward(src: sourceBatch, trg: targetBatch[0..., 0..<(targetBatch.shape[1] - 1)], training: false)
                
                let _output = output.reshaped([output.shape[0] * output.shape[1], output.shape[2]], stream: .gpu)
                
                let loss = self.lossFunction.loss(y: _output, t: targetBatch[0..., 1...].asType(DType.int32).flattened()).mean()
                
                lossHistory.append(loss.item(Float.self))
                
                
                let latestLoss = lossHistory[lossHistory.count - 1]
                tqdmRange.setDescription(description: "testing | loss: \(latestLoss) | perplexity: \(exp(latestLoss))")
                
                if batchNum == (source.count - 1) {
                    
                    epochLoss = MLX.mean(MLXArray(lossHistory))
                    
                    tqdmRange.setDescription(description: "testing | avg loss: \(epochLoss.item(Float.self)) | avg perplexity: \(exp(epochLoss.item(Float.self)))")
                    
                }
                
                GPU.clearCache()
                
            }
        }
        
        // End timer
        let endTime = Date()
        let timeInterval = endTime.timeIntervalSince(startTime)
        print("Testing completed in \(timeInterval) seconds")
        
        return epochLoss
        
    }
    
    func fit(trainData: ([MLXArray], [MLXArray]), valData: ([MLXArray], [MLXArray]), epochs: Int, saveEveryEpochs: Int, savePath: String?, validationCheck: Bool) -> ([MLXArray],[MLXArray]) {
        
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
                        bestValLoss = valLossHistory[valLossHistory.count - 1].item(Float.self)
                        self.save(path: "\(savePath)/\(epoch + 1)")
                    } else {
                        print("Current validation loss is higher than previous; Not saved")
                    }
                }
            }
        }
        
        return (trainLossHistory, valLossHistory)
    }
    
    /// Generate translations for a given input sentence
    /// - Parameters:
    ///   - sentence: Array of input tokens
    ///   - vocabs: Tuple of source and target vocabularies
    ///   - maxLength: Maximum length of generated translation
    /// - Returns: Tuple of (translated tokens, attention weights)
    ///
    func predict(sentence: [String], vocabs: ([String: Int], [String: Int]), maxLength: Int = 50) -> ([String], MLXArray) {
        // Map words to indices, using the source vocabulary
        let srcIndices = sentence.map { word in
            vocabs.0[word] ?? unkIndex
        }
        
        // Add SOS and EOS tokens
        let srcIndicesWithTokens = [sosIndex] + srcIndices + [eosIndex]
        
        // Create an MLXArray from the source indices and reshape it
        let src = MLXArray(srcIndicesWithTokens).reshaped([1, -1])
        let srcMask = self.getPadMask(x: src)
        
        // Pass through the encoder
        let encSrc = self.encoder.forward(src: src.asType(dataType), srcMask: srcMask, training: false)  // Added asType
        
        // Initialize target indices with SOS token
        var trgIndices = [sosIndex]
        var output = MLXArray([])
        var attention = MLXArray([])
        
        for _ in 0..<maxLength {
            // Create target tensor and mask
            let trg = MLXArray(trgIndices).reshaped([1, -1])
            let trgMask = self.getPadMask(x: trg) & self.getSubMask(x: trg)
            
            // Pass through the decoder
            let (out, attn) = self.decoder.forward(trg: trg.asType(dataType), trgMask: trgMask, src: encSrc, srcMask: srcMask, training: false)  // Added asType
            output = out
            attention = attn
            
            // Get prediction for next token - using argmax directly like in Python
            let trgIndex = out.argMax(axis: -1)[0..., -1].item(Int.self)  // Fixed index access
            
            // Break if we hit EOS or max length
            if trgIndex == eosIndex {
                break
            }
            
            trgIndices.append(trgIndex)
            
            if trgIndices.count >= maxLength {
                break
            }
        }
        
        // Create a reversed vocabulary mapping indices back to words
        let reversedVocab = Dictionary(uniqueKeysWithValues: vocabs.1.map { ($1, $0) })
        
        // Decode the sentence by mapping indices to words (and handle unknown tokens)
        let decodedSentence = trgIndices.map { idx in
            if let word = reversedVocab[idx] {
                return word
            }
            return unkToken
        }
        
        // Get first attention layer like in Python
        let attention0 = attention[0]
        
        // Drop the SOS token like in Python implementation
        return (Array(decodedSentence.dropFirst()), attention0)
    }
}
