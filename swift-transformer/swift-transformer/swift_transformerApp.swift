import SwiftUI
import Accelerate

@main
struct swift_transformerApp: App {
    @ObservedObject var viewModel = TransformerViewModel()
    
    var body: some Scene {
        WindowGroup {
            ContentView()
                .onAppear(perform: {
                    viewModel.setupTransformer()
                })
        }
    }
}

class TransformerViewModel: ObservableObject {
    func setupTransformer() -> Seq2Seq {
        // Token and index setup
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
        
        let (trainData, testData, valData) = dataPreparator.prepareData(path: "./dataset/", batchSize: batchSize, minFreq: 2)
        let (source, target) = trainData

        let train_data_vocabs = dataPreparator.getVocabs()

        // Use tokens as keys to access dimensions
        let inputDim = train_data_vocabs[sosToken] ?? 5959
        let outputDim = train_data_vocabs[eosToken] ?? 7801
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

        let encoder = Encoder(srcVocabSize: inputDim, headsNum: encHeads, layersNum: encLayers, dModel: hidDim, dFF: ffSize, dropoutRate: Float(encDropout), maxLen: maxLen, dataType: Array<Float>())
        let decoder = Decoder(trgVocabSize: outputDim, headsNum: decHeads, layersNum: decLayers, dModel: hidDim, dFF: ffSize, dropoutRate: Float(decDropout), maxLen: maxLen, dataType: Array<Float>())

        let model = Seq2Seq(encoder: encoder, decoder: decoder, padIdx: padIndex)
        
        model.compile(optimizer: Noam(optimizer: Adam(alpha: 1e-4, beta: 0.9, beta2: 0.98, epsilon: 1e-9), modelDim: Float(hidDim), scaleFactor: 2, warmupSteps: 4000), lossFunction: CrossEntropy(ignoreIndex: padIndex))
        
        // Convert data to [[Float]]
        let trainDataFloat = (trainData.0.map { $0.map { Float($0) } }, trainData.1.map { $0.map { Float($0) } })
        let valDataFloat = (valData.0.map { $0.map { Float($0) } }, valData.1.map { $0.map { Float($0) } })
        
        var trainLossHistory: [Float]?, valLossHistory: [Float]?
        (trainLossHistory, valLossHistory) = model.fit(
            trainData: trainDataFloat,
            valData: valDataFloat,
            epochs: 5,
            saveEveryEpochs: 20,
            savePath: "saved models/seq2seq_model",
            validationCheck: true
        )
        
        let (_, valLossHistoryFinal) = model.fit(trainData: trainDataFloat, valData: valDataFloat, epochs: 5, saveEveryEpochs: 20, savePath: "saved models/seq2seq_model", validationCheck: true)

        print("Validation Loss History: \(valLossHistoryFinal)")
        
        return model
    }
}
