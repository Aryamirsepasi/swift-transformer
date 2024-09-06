import Foundation
import Accelerate
import MLX

class DataPreparator {
    var padToken: String
    var sosToken: String
    var eosToken: String
    var unkToken: String
    
    var padIndex: Int
    var sosIndex: Int
    var eosIndex: Int
    var unkIndex: Int
    
    var tokensAndIndices: [String: Int]
    var vocabs: ([String: Int], [String: Int])?
    
    init(tokens: [String], indexes: [Int]) {
        self.padToken = tokens[0]
        self.sosToken = tokens[1]
        self.eosToken = tokens[2]
        self.unkToken = tokens[3]
        
        self.padIndex = indexes[0]
        self.sosIndex = indexes[1]
        self.eosIndex = indexes[2]
        self.unkIndex = indexes[3]
        
        self.tokensAndIndices = [
            padToken: padIndex,
            sosToken: sosIndex,
            eosToken: eosIndex,
            unkToken: unkIndex
        ]
    }
    
    func prepareData(path: String = "dataset/", batchSize: Int = 1, minFreq: Int = 10) -> (([MLXArray], [MLXArray]), ([MLXArray], [MLXArray]), ([MLXArray], [MLXArray])) {
        
        print("entered prepare data")
        
        let (trainData, valData, testData) = importMulti30kDataset(path: path)
        let clearedTrainData = clearDataset(dataset: trainData)
        let clearedValData = clearDataset(dataset: valData)
        let clearedTestData = clearDataset(dataset: testData)
        
        print("Train data sequences num = \(clearedTrainData.count)")
        
        self.vocabs = buildVocab(dataset: clearedTrainData, minFreq: minFreq)
        print("EN vocab length = \(self.vocabs!.0.count); DE vocab length = \(self.vocabs!.1.count)")
        
        let trainDataBatches = addTokens(dataset: clearedTrainData, batchSize: batchSize)
        print("Batch num = \(trainDataBatches.count)")
        
        let trainSourceTarget = buildDataset(dataset: trainDataBatches, vocabs: self.vocabs!)
        let testDataBatches = addTokens(dataset: clearedTestData, batchSize: batchSize)
        let testSourceTarget = buildDataset(dataset: testDataBatches, vocabs: self.vocabs!)
        let valDataBatches = addTokens(dataset: clearedValData, batchSize: batchSize)
        let valSourceTarget = buildDataset(dataset: valDataBatches, vocabs: self.vocabs!)
        
        print("exited prepare data")
        
        return (trainSourceTarget, testSourceTarget, valSourceTarget)
    }
    
    func getVocabs() -> ([String: Int], [String: Int])? {
        
        print("exited getVocabs")
        
        return self.vocabs
    }
    
    func filterSeq(seq: String) -> String {
        
        print("entered filterSeq")
        
        let charsToRemove = CharacterSet(charactersIn: "!”#$%&'()*+,-./:;<=>?@[\\]^_`{|}~\t\n")
        
        print("exited filterSeq")
        
        return seq.components(separatedBy: charsToRemove).joined()
    }
    
    func lowercaseSeq(seq: String) -> String {
        
        print("exited lowercaseSeq")
        
        return seq.lowercased()
    }
    
    func importMulti30kDataset(path: String) -> ([[String: String]], [[String: String]], [[String: String]]) {
        
        print("entered importMulti30kDataset")
        
        let filenames = ["train", "val", "test"]
        var trainResults: [[String: String]] = []
        var valResults: [[String: String]] = []
        var testResults: [[String: String]] = []
        
        for filename in filenames {
            let enPath = "\(path)/\(filename).en"
            let dePath = "\(path)/\(filename).de"
            
            if let enLines = try? String(contentsOfFile: enPath).split(separator: "\n"),
               let deLines = try? String(contentsOfFile: dePath).split(separator: "\n"),
               enLines.count == deLines.count {
                
                let pairs = zip(enLines, deLines).map { ["en": String($0), "de": String($1)] }
                switch filename {
                case "train":
                    trainResults.append(contentsOf: pairs)
                case "val":
                    valResults.append(contentsOf: pairs)
                case "test":
                    testResults.append(contentsOf: pairs)
                default: break
                }
            }
        }
        
        print("exited importMulti30kDataset")
        
        
        print("First entry in trainResults: \(trainResults[0])")
        
        return (trainResults, valResults, testResults)
    }
    
    func clearDataset(dataset: [[String: String]]) -> [[String: String]] {
        
        print("exited clearDataset")
        
        return dataset.map { example in
            var clearedExample = example
            clearedExample["en"] = lowercaseSeq(seq: filterSeq(seq: example["en"]!)).split(separator: " ").map(String.init).joined(separator: " ")
            clearedExample["de"] = lowercaseSeq(seq: filterSeq(seq: example["de"]!)).split(separator: " ").map(String.init).joined(separator: " ")
            return clearedExample
        }
    }
    
    func buildVocab(dataset: [[String: String]], minFreq: Int = 1) -> ([String: Int], [String: Int]) {
        
        print("entered buildVocab")
        
        var enVocab: [String: Int] = tokensAndIndices
        var deVocab: [String: Int] = tokensAndIndices
        var enFreqs: [String: Int] = [:]
        var deFreqs: [String: Int] = [:]
        
        for example in dataset {
            for word in example["en"]!.split(separator: " ") {
                enFreqs[String(word), default: 0] += 1
            }
            for word in example["de"]!.split(separator: " ") {
                deFreqs[String(word), default: 0] += 1
            }
        }
        
        for (word, freq) in enFreqs where freq >= minFreq && enVocab[word] == nil {
            enVocab[word] = enVocab.count
        }
        for (word, freq) in deFreqs where freq >= minFreq && deVocab[word] == nil {
            deVocab[word] = deVocab.count
        }
        
        print("exited buildVocab")
        
        return (enVocab, deVocab)
    }
    
    func addTokens(dataset: [[String: String]], batchSize: Int) -> [[[String: String]]] {
        print("entered addTokens")
        
        // Step 1: Add special tokens (SOS, EOS) to each sequence
        let paddedData = dataset.map { example -> [String: String] in
            var paddedExample = example
            paddedExample["en"] = "\(sosToken) \(example["en"]!) \(eosToken)"
            paddedExample["de"] = "\(sosToken) \(example["de"]!) \(eosToken)"
            return paddedExample
        }
        
        // Step 2: Split data into batches
        var batchedData: [[[String: String]]] = stride(from: 0, to: paddedData.count, by: batchSize).map {
            Array(paddedData[$0..<min($0 + batchSize, paddedData.count)])
        }
        
        // Step 3: For each batch, find the max length of the sequences and pad them
        for i in 0..<batchedData.count {
            let batch = batchedData[i]
            
            // Determine the maximum sequence lengths in the current batch
            let maxEnSeqLen = batch.map { $0["en"]!.split(separator: " ").count }.max() ?? 0
            let maxDeSeqLen = batch.map { $0["de"]!.split(separator: " ").count }.max() ?? 0
            
            // Pad sequences in the batch to the maximum length
            batchedData[i] = batch.map { example in
                var paddedExample = example
                let enWords = example["en"]!.split(separator: " ").map(String.init)
                let deWords = example["de"]!.split(separator: " ").map(String.init)
                
                paddedExample["en"] = (enWords + Array(repeating: padToken, count: maxEnSeqLen - enWords.count)).joined(separator: " ")
                paddedExample["de"] = (deWords + Array(repeating: padToken, count: maxDeSeqLen - deWords.count)).joined(separator: " ")
                
                return paddedExample
            }
        }
        
        print("exited addTokens")
        return batchedData
    }
    
    
    
    /*func buildDataset(dataset: [[[String: String]]], vocabs: ([String: Int], [String: Int])) -> (MLXArray, MLXArray) {
        
        print("entered buildDataset")

        var source: [[[Int]]] = []
        var target: [[[Int]]] = []
        
        for batch in dataset {
            var sourceBatch: [[Int]] = []
            var targetBatch: [[Int]] = []
            
            for example in batch {
                let enIndices = example["en"]!.split(separator: " ").map { vocabs.0[String($0)] ?? unkIndex }
                let deIndices = example["de"]!.split(separator: " ").map { vocabs.1[String($0)] ?? unkIndex }
                
                sourceBatch.append(enIndices)
                targetBatch.append(deIndices)
            }
            
            source.append(sourceBatch)
            target.append(targetBatch)
        }
        
        // Flatten the 3D arrays to 1D
        var flatSource = source.flatMap { $0.flatMap { $0 } }
        var flatTarget = target.flatMap { $0.flatMap { $0 } }
        
        // Calculate the required total elements for reshape
        let requiredSourceElements = source.count * (source.first?.count ?? 0) * (source.first?.first?.count ?? 0)
        let requiredTargetElements = target.count * (target.first?.count ?? 0) * (target.first?.first?.count ?? 0)
        
        // Trim or pad the flat arrays to match the required size
        if flatSource.count > requiredSourceElements {
            flatSource = Array(flatSource.prefix(requiredSourceElements))
        } else if flatSource.count < requiredSourceElements {
            flatSource.append(contentsOf: Array(repeating: padIndex, count: requiredSourceElements - flatSource.count))
        }

        if flatTarget.count > requiredTargetElements {
            flatTarget = Array(flatTarget.prefix(requiredTargetElements))
        } else if flatTarget.count < requiredTargetElements {
            flatTarget.append(contentsOf: Array(repeating: padIndex, count: requiredTargetElements - flatTarget.count))
        }
        
        // Convert the flat arrays to MLXArrays
        var sourceMLX = MLXArray(flatSource)
        var targetMLX = MLXArray(flatTarget)
        
        // Reshape MLXArrays back to 3D
        sourceMLX = sourceMLX.reshaped([source.count, source.first?.count ?? 0, source.first?.first?.count ?? 0])
        targetMLX = targetMLX.reshaped([target.count, target.first?.count ?? 0, target.first?.first?.count ?? 0])
        
        print("exited buildDataset")
        
        return (sourceMLX, targetMLX)
    }*/
    func buildDataset(dataset: [[[String: String]]], vocabs: ([String: Int], [String: Int])) -> ([MLXArray], [MLXArray]) {
        
        print("entered buildDataset")

        var sourceArrays: [MLXArray] = []
        var targetArrays: [MLXArray] = []

        for batch in dataset {
            var sourceBatch: [[Int]] = []
            var targetBatch: [[Int]] = []

            for example in batch {
                let enIndices = example["en"]!.split(separator: " ").map { vocabs.0[String($0)] ?? unkIndex }
                let deIndices = example["de"]!.split(separator: " ").map { vocabs.1[String($0)] ?? unkIndex }
                
                sourceBatch.append(enIndices)
                targetBatch.append(deIndices)
            }

            // Flatten each batch to 1D
            let flatSourceBatch = sourceBatch.flatMap { $0 }
            let flatTargetBatch = targetBatch.flatMap { $0 }

            // Convert each batch to MLXArray
            var sourceMLX = MLXArray(flatSourceBatch)
            var targetMLX = MLXArray(flatTargetBatch)

            // Reshape each MLXArray back to 2D
            sourceMLX = sourceMLX.reshaped([sourceBatch.count, sourceBatch[0].count])
            targetMLX = targetMLX.reshaped([targetBatch.count, targetBatch[0].count])

            // Add the MLXArrays to the list
            sourceArrays.append(sourceMLX)
            targetArrays.append(targetMLX)
        }

        print("exited buildDataset")
        
        return (sourceArrays, targetArrays)
    }

    
}

