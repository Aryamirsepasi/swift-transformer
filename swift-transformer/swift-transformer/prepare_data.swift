import Foundation
import Matft

import Accelerate

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
    var vocabs: [String: Int]

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
        
        self.vocabs = [:]
    }

    func prepareData(path: String = "dataset/", batchSize: Int = 1, minFreq: Int = 10) -> (([[Int]], [[Int]]), ([[Int]], [[Int]]), ([[Int]], [[Int]])) {
        let (trainData, valData, testData) = importMulti30kDataset(path: path)
        let clearedTrainData = clearDataset(data: trainData)
        let clearedValData = clearDataset(data: valData)
        let clearedTestData = clearDataset(data: testData)

        print("Train data sequences num = \(clearedTrainData.count)")
        
        let vocabs = buildVocab(dataset: clearedTrainData, minFreq: minFreq)
        print("EN vocab length = \(vocabs.count); DE vocab length = \(vocabs.count)") // Assuming one vocab for simplification
        
        let trainDataBatches = addTokens(data: clearedTrainData, batchSize: batchSize)
        print("Batch num = \(trainDataBatches.count)")

        let trainSourceTarget = buildDataset(data: trainDataBatches)
        let testDataBatches = addTokens(data: clearedTestData, batchSize: batchSize)
        let testSourceTarget = buildDataset(data: testDataBatches)
        let valDataBatches = addTokens(data: clearedValData, batchSize: batchSize)
        let valSourceTarget = buildDataset(data: valDataBatches)

        return (trainSourceTarget, testSourceTarget, valSourceTarget)
    }

    func importMulti30kDataset(path: String) -> ([(String, String)], [(String, String)], [(String, String)]) {
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

        return (trainResults, valResults, testResults)
    }

    func clearDataset(data: [(String, String)]) -> [(String, String)] {
        return data.map { (en, de) in
            let filteredEn = filterSeq(seq: en)
            let filteredDe = filterSeq(seq: de)
            let lowercasedEn = filteredEn.lowercased()
            let lowercasedDe = filteredDe.lowercased()
            return (lowercasedEn, lowercasedDe)
        }
    }
    
    func getVocabs() -> [String: Int] {
        return vocabs
    }

    func lowercaseSeq(seq: String) -> String {
        return seq.lowercased()
    }

    func filterSeq(seq: String) -> String {
        let charsToRemove = CharacterSet(charactersIn: "!”#$%&'()*+,-./:;<=>?@[\\]^_`{|}~\t\n")
        return seq.components(separatedBy: charsToRemove).joined()
    }

    func buildVocab(dataset: [(String, String)], minFreq: Int = 1) -> [String: Int] {
        var vocab: [String: Int] = tokensAndIndices
        var freqs: [String: Int] = [:]

        for (en, de) in dataset {
            let words = en.split(separator: " ") + de.split(separator: " ")
            for word in words {
                freqs[String(word), default: 0] += 1
            }
        }

        for (word, freq) in freqs {
            if freq >= minFreq && vocab[word] == nil {
                vocab[word] = vocab.count
            }
        }

        return vocab
    }

    func addTokens(data: [(String, String)], batchSize: Int) -> [[(String, String)]] {
        let paddedData = data.map { (en, de) -> (String, String) in
            let paddedEn = "\(sosToken) \(en) \(eosToken)"
            let paddedDe = "\(sosToken) \(de) \(eosToken)"
            return (paddedEn, paddedDe)
        }

        return stride(from: 0, to: paddedData.count, by: batchSize).map {
            Array(paddedData[$0..<min($0 + batchSize, paddedData.count)])
        }
    }

    func buildDataset(data: [[(String, String)]]) -> ([[Int]], [[Int]]) {
        var source: [[Int]] = []
        var target: [[Int]] = []

        for batch in data {
            var sourceBatch: [[Int]] = []
            var targetBatch: [[Int]] = []

            for (en, de) in batch {
                let enIndices = en.split(separator: " ").map { vocabs[String($0)] ?? unkIndex }
                let deIndices = de.split(separator: " ").map { vocabs[String($0)] ?? unkIndex }
                sourceBatch.append(enIndices)
                targetBatch.append(deIndices)
            }

            source.append(contentsOf: sourceBatch)
            target.append(contentsOf: targetBatch)
        }

        return (source, target)
    }
}
