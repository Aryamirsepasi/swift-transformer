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
                // Self-attention weights
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
                
                // Layer normalization parameters
                if let gamma = encoderParams["layer_\(i)_self_attention_norm_gamma"],
                   let beta = encoderParams["layer_\(i)_self_attention_norm_beta"] {
                    layer.selfAttentionNorm.gamma = gamma
                    layer.selfAttentionNorm.beta = beta
                }
                if let gamma = encoderParams["layer_\(i)_ff_norm_gamma"],
                   let beta = encoderParams["layer_\(i)_ff_norm_beta"] {
                    layer.ffLayerNorm.gamma = gamma
                    layer.ffLayerNorm.beta = beta
                }
                
                // Feed-forward weights
                if let fc1w = encoderParams["layer_\(i)_ff_fc1_w"],
                   let fc1b = encoderParams["layer_\(i)_ff_fc1_b"],
                   let fc2w = encoderParams["layer_\(i)_ff_fc2_w"],
                   let fc2b = encoderParams["layer_\(i)_ff_fc2_b"] {
                    layer.positionWiseFeedForward.fc1.w = fc1w
                    layer.positionWiseFeedForward.fc1.b = fc1b
                    layer.positionWiseFeedForward.fc2.w = fc2w
                    layer.positionWiseFeedForward.fc2.b = fc2b
                }
            }
            
            // Restore decoder parameters
            self.decoder.tokenEmbedding.w = decoderParams["token_embedding"]!
            self.decoder.positionEmbedding.pe = decoderParams["position_embedding"]!
            
            // Restore decoder layers
            for (i, layer) in self.decoder.layers.enumerated() {
                // Self-attention weights
                layer.selfAttention.KLinear.w = decoderParams["layer_\(i)_self_attention_k"]!
                layer.selfAttention.QLinear.w = decoderParams["layer_\(i)_self_attention_q"]!
                layer.selfAttention.VLinear.w = decoderParams["layer_\(i)_self_attention_v"]!
                layer.selfAttention.OLinear.w = decoderParams["layer_\(i)_self_attention_o"]!
                
                // Encoder-attention weights
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
                
                // Layer normalization parameters
                if let gamma = decoderParams["layer_\(i)_self_attention_norm_gamma"],
                   let beta = decoderParams["layer_\(i)_self_attention_norm_beta"] {
                    layer.selfAttentionNorm.gamma = gamma
                    layer.selfAttentionNorm.beta = beta
                }
                if let gamma = decoderParams["layer_\(i)_enc_attention_norm_gamma"],
                   let beta = decoderParams["layer_\(i)_enc_attention_norm_beta"] {
                    layer.encAttnLayerNorm.gamma = gamma
                    layer.encAttnLayerNorm.beta = beta
                }
                if let gamma = decoderParams["layer_\(i)_ff_norm_gamma"],
                   let beta = decoderParams["layer_\(i)_ff_norm_beta"] {
                    layer.ffLayerNorm.gamma = gamma
                    layer.ffLayerNorm.beta = beta
                }
                
                // Feed-forward weights
                if let fc1w = decoderParams["layer_\(i)_ff_fc1_w"],
                   let fc1b = decoderParams["layer_\(i)_ff_fc1_b"],
                   let fc2w = decoderParams["layer_\(i)_ff_fc2_w"],
                   let fc2b = decoderParams["layer_\(i)_ff_fc2_b"] {
                    layer.positionWiseFeedForward.fc1.w = fc1w
                    layer.positionWiseFeedForward.fc1.b = fc1b
                    layer.positionWiseFeedForward.fc2.w = fc2w
                    layer.positionWiseFeedForward.fc2.b = fc2b
                }
            }
            
            logPrint("Successfully loaded model from \"\(path)\"")
        } catch {
            logPrint("Error loading model parameters: \(error)")
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
                logPrint("Error creating directory at \(path): \(error)")
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
            // Self-attention weights
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
            
            // Layer normalization parameters
            encoderParams["layer_\(i)_self_attention_norm_gamma"] = layer.selfAttentionNorm.gamma
            encoderParams["layer_\(i)_self_attention_norm_beta"] = layer.selfAttentionNorm.beta
            encoderParams["layer_\(i)_ff_norm_gamma"] = layer.ffLayerNorm.gamma
            encoderParams["layer_\(i)_ff_norm_beta"] = layer.ffLayerNorm.beta
            
            // Feed-forward weights
            encoderParams["layer_\(i)_ff_fc1_w"] = layer.positionWiseFeedForward.fc1.w
            encoderParams["layer_\(i)_ff_fc1_b"] = layer.positionWiseFeedForward.fc1.b
            encoderParams["layer_\(i)_ff_fc2_w"] = layer.positionWiseFeedForward.fc2.w
            encoderParams["layer_\(i)_ff_fc2_b"] = layer.positionWiseFeedForward.fc2.b
        }
        
        // Save decoder parameters
        decoderParams["token_embedding"] = self.decoder.tokenEmbedding.w
        decoderParams["position_embedding"] = self.decoder.positionEmbedding.pe
        
        // Save decoder layers
        for (i, layer) in self.decoder.layers.enumerated() {
            // Self-attention weights
            decoderParams["layer_\(i)_self_attention_k"] = layer.selfAttention.KLinear.w
            decoderParams["layer_\(i)_self_attention_q"] = layer.selfAttention.QLinear.w
            decoderParams["layer_\(i)_self_attention_v"] = layer.selfAttention.VLinear.w
            decoderParams["layer_\(i)_self_attention_o"] = layer.selfAttention.OLinear.w
            
            // Encoder-attention weights
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
            
            // Layer normalization parameters
            decoderParams["layer_\(i)_self_attention_norm_gamma"] = layer.selfAttentionNorm.gamma
            decoderParams["layer_\(i)_self_attention_norm_beta"] = layer.selfAttentionNorm.beta
            decoderParams["layer_\(i)_enc_attention_norm_gamma"] = layer.encAttnLayerNorm.gamma
            decoderParams["layer_\(i)_enc_attention_norm_beta"] = layer.encAttnLayerNorm.beta
            decoderParams["layer_\(i)_ff_norm_gamma"] = layer.ffLayerNorm.gamma
            decoderParams["layer_\(i)_ff_norm_beta"] = layer.ffLayerNorm.beta
            
            // Feed-forward weights
            decoderParams["layer_\(i)_ff_fc1_w"] = layer.positionWiseFeedForward.fc1.w
            decoderParams["layer_\(i)_ff_fc1_b"] = layer.positionWiseFeedForward.fc1.b
            decoderParams["layer_\(i)_ff_fc2_w"] = layer.positionWiseFeedForward.fc2.w
            decoderParams["layer_\(i)_ff_fc2_b"] = layer.positionWiseFeedForward.fc2.b
        }
        
        do {
            // Evaluate arrays before saving to ensure computation is complete
            MLX.eval(Array(encoderParams.values))
            MLX.eval(Array(decoderParams.values))
            
            // Save encoder parameters
            let encoderURL = URL(fileURLWithPath: "\(path)/encoder.safetensors")
            try MLX.save(arrays: encoderParams, url: encoderURL)
            
            // Save decoder parameters
            let decoderURL = URL(fileURLWithPath: "\(path)/decoder.safetensors")
            try MLX.save(arrays: decoderParams, url: decoderURL)
            
            logPrint("Successfully saved model to \"\(path)\"")
        } catch {
            logPrint("Error saving model parameters: \(error)")
        }
    }
    
    /// Creates a padding mask to handle variable-length sequences
    /// - Parameter x: Input tensor of token indices
    /// - Returns: A boolean mask tensor where True (1) indicates non-padding positions
    /// Shape: [batch, 1, seq_len] for broadcasting in attention
    ///
    func getPadMask(x: MLXArray) -> MLXArray {
        // x has shape [batch, seq_len]
        // mask where pad_idx should be 0 (masked), non-pad should be 1 (not masked)
        let mask = (x .!= self.padIdx).asType(DType.int32)
        // Expand to [batch, 1, seq_len] for attention broadcasting
        return mask.expandedDimensions(axis: 1)
    }
    
    
    /// Creates a subsequent (causal) mask for decoder self-attention
    /// - Parameter x: Input tensor
    /// - Returns: A lower triangular mask to prevent attending to future positions
    /// Shape: [seq_len, seq_len] - will be broadcast over batch and heads
    ///
    func getSubMask(x: MLXArray) -> MLXArray {
        
        let seqLen = x.shape[1]
        
        // Create lower triangular matrix (1s below and on diagonal, 0s above)
        // This allows attending to previous positions and current position
        let subsequentMask = MLX.tril(MLX.ones([seqLen, seqLen]), k: 0).asType(DType.int32)
        
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
            
            // Create source padding mask: [batch, 1, src_len]
            let srcMask = self.getPadMask(x: src)
            
            // Create target masks
            let trgPadMask = getPadMask(x: trg)  // [batch, 1, trg_len]
            let trgSubMask = getSubMask(x: trg)  // [trg_len, trg_len]
            
            // Combine padding and causal masks for target
            // trgPadMask: [batch, 1, trg_len] - broadcasts to [batch, 1, trg_len, trg_len]
            // trgSubMask: [trg_len, trg_len] - broadcasts to [batch, 1, trg_len, trg_len]
            // Result: [batch, trg_len, trg_len]
            let expandedPadMask = trgPadMask.expandedDimensions(axis: 2)  // [batch, 1, 1, trg_len]
            let trgMask = (expandedPadMask * trgSubMask).squeezed(axis: 1)  // [batch, trg_len, trg_len]
            
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
    
    func train(source: [MLXArray], target: [MLXArray], epoch: Int, epochs: Int, onBatchProgress: ((Int, Int, Float) -> Void)? = nil) -> MLXArray {
        
        // Start timer
        let startTime = Date()
        
        var lossHistory : [Float] = []
        let totalBatches = source.count
        var epochLoss : MLXArray = []
        
        let zipped = zip(source, target).enumerated()
        
        let tqdmRange = TqdmSequence(sequence: zipped, description: "Epoch \(epoch + 1)/\(epochs)", unit: "batch", color: .cyan)
        
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
                tqdmRange.setDescription(description: "loss: \(String(format: "%.4f", latestLoss)) | ppl: \(String(format: "%.2f", exp(latestLoss)))")
                
                // Report batch progress to UI (every 10 batches or on last batch)
                if batchNum % 10 == 0 || batchNum == totalBatches - 1 {
                    onBatchProgress?(batchNum + 1, totalBatches, latestLoss)
                    // Also log to UI periodically
                    if batchNum % 50 == 0 {
                        logPrint("[Epoch \(epoch + 1)/\(epochs)] Batch \(batchNum + 1)/\(totalBatches) | Loss: \(String(format: "%.4f", latestLoss)) | Perplexity: \(String(format: "%.2f", exp(latestLoss)))")
                    }
                }
                
                if batchNum == (totalBatches - 1) {
                    epochLoss = MLX.mean(MLXArray(lossHistory))
                    let avgLoss = epochLoss.item(Float.self)
                    tqdmRange.setDescription(description: "avg loss: \(String(format: "%.4f", avgLoss)) | avg ppl: \(String(format: "%.2f", exp(avgLoss)))")
                    logPrint("✓ Epoch \(epoch + 1)/\(epochs) completed | Avg Loss: \(String(format: "%.4f", avgLoss)) | Avg Perplexity: \(String(format: "%.2f", exp(avgLoss)))")
                }
                
                GPU.clearCache()
            }
        }
        
        // End timer
        let endTime = Date()
        let timeInterval = endTime.timeIntervalSince(startTime)
        logPrint("Training completed in \(timeInterval) seconds")
        
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
                tqdmRange.setDescription(description: "loss: \(String(format: "%.4f", latestLoss)) | ppl: \(String(format: "%.2f", exp(latestLoss)))")
                
                if batchNum == (source.count - 1) {
                    
                    epochLoss = MLX.mean(MLXArray(lossHistory))
                    let avgLoss = epochLoss.item(Float.self)
                    tqdmRange.setDescription(description: "avg loss: \(String(format: "%.4f", avgLoss)) | avg ppl: \(String(format: "%.2f", exp(avgLoss)))")
                    logPrint("✓ Validation | Avg Loss: \(String(format: "%.4f", avgLoss)) | Avg Perplexity: \(String(format: "%.2f", exp(avgLoss)))")
                }
                
                GPU.clearCache()
                
            }
        }
        
        // End timer
        let endTime = Date()
        let timeInterval = endTime.timeIntervalSince(startTime)
        logPrint("Testing completed in \(timeInterval) seconds")
        
        return epochLoss
        
    }
    
    func fit(trainData: ([MLXArray], [MLXArray]), valData: ([MLXArray], [MLXArray]), epochs: Int, saveEveryEpochs: Int, savePath: String?, validationCheck: Bool, onEpochStart: ((Int, Int) -> Void)? = nil, onEpochEnd: ((Int, Float, Float) -> Void)? = nil, onBatchProgress: ((Int, Int, Float) -> Void)? = nil) -> ([MLXArray],[MLXArray]) {
        
        setOptimizer()
        
        var bestValLoss = Float.infinity
        var trainLossHistory: [MLXArray] = []
        var valLossHistory: [MLXArray] = []
        
        let (trainSource, trainTarget) = trainData
        let (valSource, valTarget) = valData
        
        for epoch in 0..<epochs {
            // Notify epoch start
            onEpochStart?(epoch + 1, epochs)
            
            let trainLoss = self.train(source: trainSource, target: trainTarget, epoch: epoch, epochs: epochs, onBatchProgress: onBatchProgress)
            trainLossHistory.append(trainLoss)
            
            let valLoss = self.evaluate(source: valSource, target: valTarget)
            valLossHistory.append(valLoss)
            
            // Notify epoch end with losses
            onEpochEnd?(epoch + 1, trainLoss.item(Float.self), valLoss.item(Float.self))
            
            if ((savePath != nil) && ((epoch + 1) % saveEveryEpochs == 0)) {
                if !validationCheck {
                    self.save(path: "\(savePath)/\(epoch + 1)")
                } else {
                    if valLossHistory[valLossHistory.count - 1].item(Float.self) < bestValLoss {
                        bestValLoss = valLossHistory[valLossHistory.count - 1].item(Float.self)
                        self.save(path: "\(savePath)/\(epoch + 1)")
                    } else {
                        logPrint("Current validation loss is higher than previous; Not saved")
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
        
        // Create an MLXArray from the source indices and reshape it [1, seq_len]
        let src = MLXArray(srcIndicesWithTokens).reshaped([1, -1])
        let srcMask = self.getPadMask(x: src)
        
        // Pass through the encoder
        let encSrc = self.encoder.forward(src: src.asType(dataType), srcMask: srcMask, training: false)
        
        // Initialize target indices with SOS token
        var trgIndices = [sosIndex]
        var output = MLXArray([])
        var attention = MLXArray([])
        
        for _ in 0..<maxLength {
            // Create target tensor [1, current_seq_len]
            let trg = MLXArray(trgIndices).reshaped([1, -1])
            
            // Create combined mask for target
            let trgPadMask = self.getPadMask(x: trg)  // [1, 1, trg_len]
            let trgSubMask = self.getSubMask(x: trg)  // [trg_len, trg_len]
            let expandedPadMask = trgPadMask.expandedDimensions(axis: 2)  // [1, 1, 1, trg_len]
            let trgMask = (expandedPadMask * trgSubMask).squeezed(axis: 1)  // [1, trg_len, trg_len]
            
            // Pass through the decoder
            let (out, attn) = self.decoder.forward(trg: trg.asType(dataType), trgMask: trgMask, src: encSrc, srcMask: srcMask, training: false)
            output = out
            attention = attn
            
            // Get prediction for next token - using argmax on the last position
            // out shape: [1, seq_len, vocab_size]
            let lastOutput = out[0, -1]  // Get last position: [vocab_size]
            let trgIndex = lastOutput.argMax().item(Int.self)
            
            // Break if we hit EOS
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
        
        // Get first attention layer
        let attention0 = attention.ndim > 0 ? attention[0] : attention
        
        // Drop the SOS token like in Python implementation
        return (Array(decodedSentence.dropFirst()), attention0)
    }
}
