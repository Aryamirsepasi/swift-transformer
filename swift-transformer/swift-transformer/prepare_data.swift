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

    func prepareData(path: String = "dataset/", batchSize: Int = 1, minFreq: Int = 10) -> ((MLXArray, MLXArray), (MLXArray, MLXArray), (MLXArray, MLXArray)) {
        
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

    func importMulti30kDataset(path: String) -> ([(String, String)], [(String, String)], [(String, String)]) {
        
        print("entered importMulti30kDataset")

        let filenames = ["train", "val", "test"]
        var trainResults: [(String, String)] = []
        var valResults: [(String, String)] = []
        var testResults: [(String, String)] = []

        for filename in filenames {
            let enPath = "\(path)/\(filename).en"
            let dePath = "\(path)/\(filename).de"

            if let enLines = try? String(contentsOfFile: enPath).split(separator: "\n"),
               let deLines = try? String(contentsOfFile: dePath).split(separator: "\n"),
               enLines.count == deLines.count {

                let pairs = zip(enLines, deLines).map { (String($0), String($1)) }
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

        return (trainResults, valResults, testResults)
    }

    func clearDataset(dataset: [(String, String)]) -> [(String, String)] {
        
        print("exited clearDataset")
        
        return dataset.map { (en, de) in
            let filteredEn = filterSeq(seq: en)
            let filteredDe = filterSeq(seq: de)
            let lowercasedEn = lowercaseSeq(seq: filteredEn)
            let lowercasedDe = lowercaseSeq(seq: filteredDe)
            return (lowercasedEn, lowercasedDe)
        }
    }

    func buildVocab(dataset: [(String, String)], minFreq: Int = 1) -> ([String: Int], [String: Int]) {
        
        print("entered buildVocab")
        
        var enVocab: [String: Int] = tokensAndIndices
        var deVocab: [String: Int] = tokensAndIndices
        var enFreqs: [String: Int] = [:]
        var deFreqs: [String: Int] = [:]

        for (en, de) in dataset {
            for word in en.split(separator: " ") {
                enFreqs[String(word), default: 0] += 1
            }
            for word in de.split(separator: " ") {
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

    func addTokens(dataset: [(String, String)], batchSize: Int) -> [[(String, String)]] {
        print("entered addTokens")

        // Step 1: Add special tokens (SOS, EOS) to each sequence
        let paddedData = dataset.map { (en, de) -> (String, String) in
            let paddedEn = "\(sosToken) \(en) \(eosToken)"
            let paddedDe = "\(sosToken) \(de) \(eosToken)"
            return (paddedEn, paddedDe)
        }
        
        // Step 2: Split data into batches
        var batchedData: [[(String, String)]] = stride(from: 0, to: paddedData.count, by: batchSize).map {
            Array(paddedData[$0..<min($0 + batchSize, paddedData.count)])
        }

        // Step 3: For each batch, find the max length of the sequences and pad them
        for i in 0..<batchedData.count {
            let batch = batchedData[i]

            // Determine the maximum sequence lengths in the current batch
            let maxEnSeqLen = batch.map { $0.0.split(separator: " ").count }.max() ?? 0
            let maxDeSeqLen = batch.map { $0.1.split(separator: " ").count }.max() ?? 0

            // Pad sequences in the batch to the maximum length
            batchedData[i] = batch.map { (en, de) in
                let enWords = en.split(separator: " ").map(String.init)
                let deWords = de.split(separator: " ").map(String.init)

                // Pad English and German sequences with padToken to max length
                let paddedEn = enWords + Array(repeating: padToken, count: maxEnSeqLen - enWords.count)
                let paddedDe = deWords + Array(repeating: padToken, count: maxDeSeqLen - deWords.count)

                return (paddedEn.joined(separator: " "), paddedDe.joined(separator: " "))
            }
        }

        print("exited addTokens")
        return batchedData
    }


    func buildDataset(dataset: [[(String, String)]], vocabs: ([String: Int], [String: Int])) -> (MLXArray, MLXArray) {
        
        print("entered buildDataset")

        var source: MLXArray = []
        var target: MLXArray = []
        var sourceBatchnew : MLXArray = []
        var targetBatchnew : MLXArray = []

        for batch in dataset {
            var sourceBatch: MLXArray = []
            var targetBatch: MLXArray = []

            for (en, de) in batch {
                let enIndices = en.split(separator: " ").map { vocabs.0[String($0)] ?? unkIndex }
                let deIndices = de.split(separator: " ").map { vocabs.1[String($0)] ?? unkIndex }
                
                for i in 0..<sourceBatch.count{
                    sourceBatchnew[i] = sourceBatch[i]
                }
                
                for i in 0..<enIndices.count{
                    sourceBatchnew[i + sourceBatch.count] = MLXArray(enIndices[i])
                }
                
                for i in 0..<targetBatch.count{
                    targetBatchnew[i] = targetBatch[i]
                }
                
                for i in 0..<deIndices.count{
                    targetBatchnew[i + targetBatch.count] = MLXArray(deIndices[i])
                }
            }
            
            var sourcenew : MLXArray = []
            for i in 0..<source.count{
                sourcenew[i] = source[i]
            }
            
            for i in 0..<sourceBatchnew.count{
                sourcenew[i + sourceBatchnew.count] = sourceBatchnew[i]
            }
            
            var targetnew : MLXArray = []
            for i in 0..<target.count{
                targetnew[i] = target[i]
            }
            
            for i in 0..<targetBatchnew.count{
                targetnew[i + targetBatchnew.count] = targetBatchnew[i]
            }
        }
        
        print("exited buildDataset")

        return (source, target)
    }
}
