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
        let (trainData, valData, testData) = importMulti30kDataset(path: path)
        let (clearedTrainData, clearedValData, clearedTestData) = clearDataset(trainData, valData, testData).tuple
        
        print("train data sequences num =  \(clearedTrainData.count)")
        
        self.vocabs = buildVocab(dataset: clearedTrainData, minFreq: minFreq)
        print("EN vocab length =  \(self.vocabs!.0.count); DE vocab length = \(self.vocabs!.1.count)")
        
        let trainDataBatches = addTokens(dataset: clearedTrainData, batchSize: batchSize)
        print("batch num =  \(trainDataBatches.count)")
        
        let trainSourceTarget = buildDataset(dataset: trainDataBatches, vocabs: self.vocabs!)
        let testDataBatches = addTokens(dataset: clearedTestData, batchSize: batchSize)
        let testSourceTarget = buildDataset(dataset: testDataBatches, vocabs: self.vocabs!)
        let valDataBatches = addTokens(dataset: clearedValData, batchSize: batchSize)
        let valSourceTarget = buildDataset(dataset: valDataBatches, vocabs: self.vocabs!)
        
        return (trainSourceTarget, testSourceTarget, valSourceTarget)
    }
    
    func getVocabs() -> ([String: Int], [String: Int])? {
        return self.vocabs
    }
    
    func filterSeq(seq: String) -> String {
        let charsToRemove = CharacterSet(charactersIn: "!”#$%&'()*+,-./:;<=>?@[\\]^_`{|}~\t\n")
        return seq.components(separatedBy: charsToRemove).joined()
    }
    
    func lowercaseSeq(seq: String) -> String {
        return seq.lowercased()
    }
    
    func importMulti30kDataset(path: String) -> ([[String: String]], [[String: String]], [[String: String]]) {
        let filenames = ["train", "val", "test"]
        var trainResults: [[String: String]] = []
        var valResults: [[String: String]] = []
        var testResults: [[String: String]] = []
        
        let currentPath = FileManager.default.currentDirectoryPath

        // Get the container directory
        let containerURL = FileManager.default.containerURL(forSecurityApplicationGroupIdentifier: "com.aryamirsepasi.swift-transformer")

        guard let datasetURL = containerURL?.appendingPathComponent("dataset") else {
            print("Error: Could not find dataset directory")
            return (trainResults, valResults, testResults)
        }

        for filename in filenames {
            let enPath = datasetURL.appendingPathComponent("\(filename).en")
            let dePath = datasetURL.appendingPathComponent("\(filename).de")

            do {
                let enContent = try String(contentsOf: enPath)
                let deContent = try String(contentsOf: dePath)

                let enLines = enContent.split(separator: "\n")
                let deLines = deContent.split(separator: "\n")

                if enLines.count == deLines.count {
                    let pairs = zip(enLines, deLines).map { ["en": String($0), "de": String($1)] }
                    switch filename {
                    case "train":
                        trainResults.append(contentsOf: pairs)
                    case "val":
                        valResults.append(contentsOf: pairs)
                    case "test":
                        testResults.append(contentsOf: pairs)
                    default:
                        break
                    }
                } else {
                    print("Error: Mismatch in line counts for \(filename).en and \(filename).de")
                }
            } catch {
                print("Error reading files at paths \(enPath) or \(dePath): \(error)")
            }
        }

        return (trainResults, valResults, testResults)
    }


    
    func clearDataset(_ datasets: [[String: String]]...) -> [[[String: [String]]]] {
        return datasets.map { dataset in
            dataset.map { example in
                var clearedExample: [String: [String]] = [:]
                clearedExample["en"] = lowercaseSeq(seq: filterSeq(seq: example["en"]!)).split(separator: " ").map(String.init)
                clearedExample["de"] = lowercaseSeq(seq: filterSeq(seq: example["de"]!)).split(separator: " ").map(String.init)
                return clearedExample
            }
        }
    }
    
    func buildVocab(dataset: [[String: [String]]], minFreq: Int = 1) -> ([String: Int], [String: Int]) {
        var enVocab: [String: Int] = tokensAndIndices
        var deVocab: [String: Int] = tokensAndIndices
        var enFreqs: [String: Int] = [:]
        var deFreqs: [String: Int] = [:]
        
        for example in dataset {
            for word in example["en"]! {
                enFreqs[word, default: 0] += 1
            }
            for word in example["de"]! {
                deFreqs[word, default: 0] += 1
            }
        }
        
        for (word, freq) in enFreqs where freq >= minFreq && enVocab[word] == nil {
            enVocab[word] = enVocab.count
        }
        for (word, freq) in deFreqs where freq >= minFreq && deVocab[word] == nil {
            deVocab[word] = deVocab.count
        }
        
        return (enVocab, deVocab)
    }
    
    func addTokens(dataset: [[String: [String]]], batchSize: Int) -> [[[String: [String]]]] {
        // First add SOS and EOS tokens
        let datasetWithTokens = dataset.map { example -> [String: [String]] in
            var newExample = example
            newExample["en"] = [sosToken] + example["en"]! + [eosToken]
            newExample["de"] = [sosToken] + example["de"]! + [eosToken]
            return newExample
        }
        
        // Split into batches more similarly to numpy's array_split
        let numBatches = Int(ceil(Double(datasetWithTokens.count) / Double(batchSize)))
        var batches: [[[String: [String]]]] = []
        
        for i in 0..<numBatches {
            let start = i * batchSize
            let end = min(start + batchSize, datasetWithTokens.count)
            var batch = Array(datasetWithTokens[start..<end])
            
            // Find max lengths for this batch
            let maxEnSeqLen = batch.map { $0["en"]!.count }.max() ?? 0
            let maxDeSeqLen = batch.map { $0["de"]!.count }.max() ?? 0
            
            // Pad sequences in the batch
            for j in 0..<batch.count {
                let enPadding = Array(repeating: padToken, count: maxEnSeqLen - batch[j]["en"]!.count)
                let dePadding = Array(repeating: padToken, count: maxDeSeqLen - batch[j]["de"]!.count)
                
                batch[j]["en"]! += enPadding
                batch[j]["de"]! += dePadding
            }
            
            batches.append(batch)
        }
        
        return batches
    }
    
    func buildDataset(dataset: [[[String: [String]]]], vocabs: ([String: Int], [String: Int])) -> ([MLXArray], [MLXArray]) {
        var sourceArrays: [MLXArray] = []
        var targetArrays: [MLXArray] = []
        
        for batch in dataset {
            var sourceBatch: [[Int]] = []
            var targetBatch: [[Int]] = []
            
            for example in batch {
                let enIndices = example["en"]!.map { vocabs.0[$0] ?? unkIndex }
                let deIndices = example["de"]!.map { vocabs.1[$0] ?? unkIndex }
                
                sourceBatch.append(enIndices)
                targetBatch.append(deIndices)
            }
            
            // Create MLXArrays using flatMap and reshape
            let sourceMLX = MLXArray(sourceBatch.flatMap { $0 })
                .reshaped([sourceBatch.count, sourceBatch[0].count])
                .asType(DType.int32)
                
            let targetMLX = MLXArray(targetBatch.flatMap { $0 })
                .reshaped([targetBatch.count, targetBatch[0].count])
                .asType(DType.int32)
            
            sourceArrays.append(sourceMLX)
            targetArrays.append(targetMLX)
        }
        
        return (sourceArrays, targetArrays)
    }
}

// Helper extension to convert array to tuple
extension Array {
    var tuple: (Element, Element, Element) {
        guard count == 3 else { fatalError("Array must contain exactly 3 elements") }
        return (self[0], self[1], self[2])
    }
}
