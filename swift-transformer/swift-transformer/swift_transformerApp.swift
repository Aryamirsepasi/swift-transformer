import SwiftUI
import MLX
import Charts

@main
struct swift_transformerApp: App {
    @StateObject var viewModel = TransformerViewModel()
    
    var body: some Scene {
        WindowGroup {
            ContentView(viewModel: viewModel)
        }
    }
}

class TransformerViewModel: ObservableObject {
    @Published var lossData: [LossDataPoint] = []
    @Published var trainingState: TrainingState = .idle
    @Published var exampleTranslations: [ExampleTranslation] = []
    @Published var currentBatch: Int = 0
    @Published var totalBatches: Int = 0
    @Published var currentLoss: Float = 0
    
    // Model and vocabulary storage for inference
    private var model: Seq2Seq?
    private var vocabs: ([String: Int], [String: Int])?
    
    func setupTransformer() {
        // Run training on background thread
        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            self?.runTraining()
        }
    }
    
    private func updateState(_ state: TrainingState) {
        DispatchQueue.main.async {
            self.trainingState = state
        }
    }
    
    private func runTraining() {
        updateState(.preparingData)
        logPrint("Starting transformer training...")
        
        // Token and index setup
        let dataType = DType.float32
        let padToken = "<pad>"
        let sosToken = "<sos>"
        let eosToken = "<eos>"
        let unkToken = "<unk>"
        
        let batchSize = 128
        
        let padIndex = 0
        let sosIndex = 1
        let eosIndex = 2
        let unkIndex = 3
        
        let tokens = [padToken, sosToken, eosToken, unkToken]
        logPrint("Directory: \(FileManager.default.currentDirectoryPath)")
        let indexes = [padIndex, sosIndex, eosIndex, unkIndex]
        
        let dataPreparator = DataPreparator(tokens: tokens, indexes: indexes)
        
        // Get raw data
        logPrint("Loading dataset...")
        let (rawTrainData, rawValData, rawTestData) = dataPreparator.importMulti30kDataset(path: "./dataset/")
        
        if rawTrainData.isEmpty {
            updateState(.error("Failed to load dataset. Please ensure dataset files are in the bundle."))
            logPrint("ERROR: No training data loaded!")
            return
        }
        
        logPrint("Loaded \(rawTrainData.count) training, \(rawValData.count) validation, \(rawTestData.count) test examples")
        
        // Clear all datasets at once
        let (clearedTrainData, clearedValData, clearedTestData) = dataPreparator.clearDataset(rawTrainData, rawValData, rawTestData).tuple
        
        // Build vocabulary from cleared training data
        logPrint("Building vocabulary...")
        let trainDataVocabs = dataPreparator.buildVocab(dataset: clearedTrainData, minFreq: 2)
        dataPreparator.vocabs = trainDataVocabs
        self.vocabs = trainDataVocabs
        
        logPrint("EN vocab size: \(trainDataVocabs.0.count), DE vocab size: \(trainDataVocabs.1.count)")
        
        // Process each dataset
        logPrint("Processing batches...")
        let trainDataBatches = dataPreparator.addTokens(dataset: clearedTrainData, batchSize: batchSize)
        let testDataBatches = dataPreparator.addTokens(dataset: clearedTestData, batchSize: batchSize)
        let valDataBatches = dataPreparator.addTokens(dataset: clearedValData, batchSize: batchSize)
        
        // Build final datasets
        let trainData = dataPreparator.buildDataset(dataset: trainDataBatches, vocabs: trainDataVocabs)
        let testData = dataPreparator.buildDataset(dataset: testDataBatches, vocabs: trainDataVocabs)
        let valData = dataPreparator.buildDataset(dataset: valDataBatches, vocabs: trainDataVocabs)
        
        let inputDim = trainDataVocabs.0.count
        let outputDim = trainDataVocabs.1.count
        
        logPrint("Train batches: \(trainData.0.count), Val batches: \(valData.0.count)")
        
        // Model dimensions and parameters
        let hidDim = 256
        let encLayers = 3
        let decLayers = 3
        let encHeads = 8
        let decHeads = 8
        let ffSize = 512
        let encDropout = 0.3
        let decDropout = 0.3
        let maxLen = 5000
        
        logPrint("Creating model with hidDim=\(hidDim), layers=\(encLayers), heads=\(encHeads)")
        
        let encoder = Encoder(srcVocabSize: inputDim, headsNum: encHeads, layersNum: encLayers, dModel: hidDim, dFF: ffSize, dropoutRate: Float(encDropout), maxLen: maxLen, dataType: dataType)
        let decoder = Decoder(trgVocabSize: outputDim, headsNum: decHeads, layersNum: decLayers, dModel: hidDim, dFF: ffSize, dropoutRate: Float(decDropout), maxLen: maxLen, dataType: dataType)
        
        let model = Seq2Seq(encoder: encoder, decoder: decoder, padIdx: padIndex)
        self.model = model
        
        model.compile(optimizer: Noam(optimizer: Adam(alpha: 1e-4, beta: 0.9, beta2: 0.98, epsilon: 1e-9), modelDim: Float(hidDim), scaleFactor: 2, warmupSteps: 4000), lossFunction: CrossEntropy(ignore_index: padIndex))
        
        let epochs = 5
        updateState(.training(epoch: 1, totalEpochs: epochs))
        logPrint("Starting training for \(epochs) epochs...")
        
        // Get save path in application support directory
        let savePath = getSavePath()
        logPrint("Models will be saved to: \(savePath)")
        
        let (trainLossHistory, valLossHistory) = model.fit(
            trainData: trainData,
            valData: valData,
            epochs: epochs,
            saveEveryEpochs: 1,
            savePath: savePath,
            validationCheck: true,
            onEpochStart: { [weak self] epoch, total in
                self?.updateState(.training(epoch: epoch, totalEpochs: total))
            },
            onEpochEnd: { [weak self] epoch, trainLoss, valLoss in
                DispatchQueue.main.async {
                    // Add new data points for this epoch
                    self?.lossData.append(LossDataPoint(epoch: epoch, loss: trainLoss, series: "Train Loss"))
                    self?.lossData.append(LossDataPoint(epoch: epoch, loss: valLoss, series: "Validation Loss"))
                }
            },
            onBatchProgress: { [weak self] batch, total, loss in
                DispatchQueue.main.async {
                    self?.currentBatch = batch
                    self?.totalBatches = total
                    self?.currentLoss = loss
                }
            }
        )
        
        // Final loss data is already updated via callbacks
        let trainLosses: [Float] = trainLossHistory.map { $0.item(Float.self) }
        let valLosses: [Float] = valLossHistory.map { $0.item(Float.self) }
        
        logPrint("Training complete! Final train loss: \(String(format: "%.4f", trainLosses.last ?? 0)), val loss: \(String(format: "%.4f", valLosses.last ?? 0))")
        
        updateState(.evaluating)
        logPrint("Generating example translations...")
        
        // Evaluate on some validation examples
        let sentencesNum = min(5, clearedValData.count)
        let totalSentences = clearedValData.count
        let randomIndices = (0..<sentencesNum).map { _ in Int.random(in: 0..<totalSentences) }
        let sentencesSelection = randomIndices.map { clearedValData[$0] }
        
        var examples: [ExampleTranslation] = []
        
        for (i, example) in sentencesSelection.enumerated() {
            logPrint("\nExample №\(i + 1)")
            if let inputSentence = example["en"], let targetSentence = example["de"] {
                let inputText = inputSentence.joined(separator: " ")
                logPrint("Input: \(inputText)")
                
                let (decodedSentence, _) = model.predict(sentence: inputSentence, vocabs: trainDataVocabs)
                let outputText = decodedSentence.joined(separator: " ")
                let targetText = targetSentence.joined(separator: " ")
                
                logPrint("Output: \(outputText)")
                logPrint("Target: \(targetText)")
                
                examples.append(ExampleTranslation(
                    input: inputText,
                    output: outputText,
                    target: targetText
                ))
            }
        }
        
        DispatchQueue.main.async {
            self.exampleTranslations = examples
        }
        
        updateState(.completed)
        logPrint("\n✅ Training completed successfully!")
    }
    
    func translate(sentence: [String]) -> [String]? {
        guard let model = model, let vocabs = vocabs else {
            logPrint("Model not ready for translation")
            return nil
        }
        
        let (result, _) = model.predict(sentence: sentence, vocabs: vocabs)
        return result
    }
    
    private func getSavePath() -> String {
        // Use Application Support directory for saving models
        let fileManager = FileManager.default
        
        if let appSupport = fileManager.urls(for: .applicationSupportDirectory, in: .userDomainMask).first {
            let modelDir = appSupport.appendingPathComponent("SwiftTransformer/models")
            
            // Create directory if it doesn't exist
            if !fileManager.fileExists(atPath: modelDir.path) {
                do {
                    try fileManager.createDirectory(at: modelDir, withIntermediateDirectories: true)
                    logPrint("Created model directory: \(modelDir.path)")
                } catch {
                    logPrint("Failed to create model directory: \(error)")
                }
            }
            
            return modelDir.path
        }
        
        // Fallback to current directory
        return FileManager.default.currentDirectoryPath
    }
}

// Define a struct for the data points for plotting
struct LossDataPoint: Identifiable {
    let id = UUID()
    let epoch: Int
    let loss: Float
    let series: String
}
