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
    
    func setupTransformer() -> Seq2Seq {
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
        print(FileManager.default.currentDirectoryPath) // checking directory
        let indexes = [padIndex, sosIndex, eosIndex, unkIndex]
        
        let dataPreparator = DataPreparator(tokens: tokens, indexes: indexes)
        
        print("run prepareData")
        let (trainData, testData, valData) = dataPreparator.prepareData(path: "./dataset/", batchSize: batchSize, minFreq: 2)
        let (source, target) = trainData
        
        print("run getVocabs")
        let trainDataVocabs = dataPreparator.getVocabs()!
        
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
        
        print("create encoder, decoder objects")
        let encoder = Encoder(srcVocabSize: inputDim, headsNum: encHeads, layersNum: encLayers, dModel: hidDim, dFF: ffSize, dropoutRate: Float(encDropout), maxLen: maxLen, dataType: dataType)
        let decoder = Decoder(trgVocabSize: outputDim, headsNum: decHeads, layersNum: decLayers, dModel: hidDim, dFF: ffSize, dropoutRate: Float(decDropout), maxLen: maxLen, dataType: dataType)
        
        print("create Seq2Seq object")
        
        let model = Seq2Seq(encoder: encoder, decoder: decoder, padIdx: padIndex)
        
        print("run compile")
        model.compile(optimizer: Noam(optimizer: Adam(alpha: 1e-4, beta: 0.9, beta2: 0.98, epsilon: 1e-9), modelDim: Float(hidDim), scaleFactor: 2, warmupSteps: 4000), lossFunction: CrossEntropy(ignore_index: padIndex))
        
        print("run fit")
        let (trainLossHistory, valLossHistory) = model.fit(
            trainData: trainData,
            valData: valData,
            epochs: 5,
            saveEveryEpochs: 20,
            savePath: "saved models/seq2seq_model",
            validationCheck: true
        )
        
        // Convert MLXArrays to Swift arrays of Floats for plotting
        let trainLosses: [Float] = trainLossHistory.map { $0.item(Float.self) }
        let valLosses: [Float] = valLossHistory.map { $0.item(Float.self) }
        
        // Prepare data for plotting with different series
        self.lossData = trainLosses.enumerated().map { (index, loss) in
            LossDataPoint(epoch: index, loss: loss, series: "Train Loss")
        } + valLosses.enumerated().map { (index, loss) in
            LossDataPoint(epoch: index, loss: loss, series: "Validation Loss")
        }
        
        print("Training Loss History MLX: \(trainLossHistory)")
        print("Validation Loss History MLX: \(valLossHistory)")
        
        
        print("Training Loss History Array: \(trainLosses)")
        print("Validation Loss History Array: \(valLosses)")
        
        return model
    }
}

// Define a struct for the data points for plotting
struct LossDataPoint: Identifiable {
    let id = UUID()
    let epoch: Int
    let loss: Float
    let series: String // New property to distinguish between series
}
