import SwiftUI
import MLX
import Charts

@main
struct swift_transformerApp: App {
    @StateObject var viewModel = TransformerViewModel()
    
    var body: some Scene {
        WindowGroup {
            ContentView(viewModel: viewModel)
                .onAppear {
                    viewModel.setupTransformer()
                }
        }
    }
}

class TransformerViewModel: ObservableObject {
    @Published var lossData: [LossDataPoint] = []
    
    func setupTransformer() {
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
        print("Directory: \(FileManager.default.currentDirectoryPath)") // checking directory
        let indexes = [padIndex, sosIndex, eosIndex, unkIndex]
        
        let dataPreparator = DataPreparator(tokens: tokens, indexes: indexes)
        
        /// Get raw data
        let (rawTrainData, rawValData, rawTestData) = dataPreparator.importMulti30kDataset(path: "./dataset/")
        
        // Clear all datasets at once
        let (clearedTrainData, clearedValData, clearedTestData) = dataPreparator.clearDataset(rawTrainData, rawValData, rawTestData).tuple
        
        // Build vocabulary from cleared training data
        let trainDataVocabs = dataPreparator.buildVocab(dataset: clearedTrainData, minFreq: 2)
        dataPreparator.vocabs = trainDataVocabs
        
        // Process each dataset
        let trainDataBatches = dataPreparator.addTokens(dataset: clearedTrainData, batchSize: batchSize)
        let testDataBatches = dataPreparator.addTokens(dataset: clearedTestData, batchSize: batchSize)
        let valDataBatches = dataPreparator.addTokens(dataset: clearedValData, batchSize: batchSize)
        
        // Build final datasets
        let trainData = dataPreparator.buildDataset(dataset: trainDataBatches, vocabs: trainDataVocabs)
        let testData = dataPreparator.buildDataset(dataset: testDataBatches, vocabs: trainDataVocabs)
        let valData = dataPreparator.buildDataset(dataset: valDataBatches, vocabs: trainDataVocabs)
        
        let inputDim = trainDataVocabs.0.count
        let outputDim = trainDataVocabs.1.count
        
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
        
        let encoder = Encoder(srcVocabSize: inputDim, headsNum: encHeads, layersNum: encLayers, dModel: hidDim, dFF: ffSize, dropoutRate: Float(encDropout), maxLen: maxLen, dataType: dataType)
        let decoder = Decoder(trgVocabSize: outputDim, headsNum: decHeads, layersNum: decLayers, dModel: hidDim, dFF: ffSize, dropoutRate: Float(decDropout), maxLen: maxLen, dataType: dataType)
        
        let model = Seq2Seq(encoder: encoder, decoder: decoder, padIdx: padIndex)
        
        model.compile(optimizer: Noam(optimizer: Adam(alpha: 1e-4, beta: 0.9, beta2: 0.98, epsilon: 1e-9), modelDim: Float(hidDim), scaleFactor: 2, warmupSteps: 4000), lossFunction: CrossEntropy(ignore_index: padIndex))
        
        let (trainLossHistory, valLossHistory) = model.fit(
            trainData: trainData,
            valData: valData,
            epochs: 5,
            saveEveryEpochs: 20,
            savePath: ".",
            validationCheck: true
        )
        
        let trainLosses: [Float] = trainLossHistory.map { $0.item(Float.self) }
        let valLosses: [Float] = valLossHistory.map { $0.item(Float.self) }
        
        self.lossData = trainLosses.enumerated().map { (index, loss) in
            LossDataPoint(epoch: index, loss: loss, series: "Train Loss")
        } + valLosses.enumerated().map { (index, loss) in
            LossDataPoint(epoch: index, loss: loss, series: "Validation Loss")
        }
        
        // Evaluate on some validation examples
        let sentencesNum = 10
        let totalSentences = clearedValData.count
        let randomIndices = (0..<sentencesNum).map { _ in Int.random(in: 0..<totalSentences) }
        let sentencesSelection = randomIndices.map { clearedValData[$0] }
        
        for (i, example) in sentencesSelection.enumerated() {
            print("\nExample №\(i + 1)")
            if let inputSentence = example["en"], let targetSentence = example["de"] {
                print("Input sentence: \(inputSentence.joined(separator: " "))")
                let (decodedSentence, attention) = model.predict(sentence: inputSentence, vocabs: trainDataVocabs)
                print("Decoded sentence: \(decodedSentence.joined(separator: " "))")
                print("Target sentence: \(targetSentence.joined(separator: " "))")
            }
        }
    }
}

// Define a struct for the data points for plotting
struct LossDataPoint: Identifiable {
    let id = UUID()
    let epoch: Int
    let loss: Float
    let series: String
}
