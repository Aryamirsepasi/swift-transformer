//
//  prepare_data.swift
//  metal-transformer
//
//  Created by Arya Mirsepasi on 26.04.24.
//

import Foundation
import Matft

class DataPreparator {
    let padToken: String
    let sosToken: String
    let eosToken: String
    let unkToken: String
    
    let padIndex: Int
    let sosIndex: Int
    let eosIndex: Int
    let unkIndex: Int
    
    var toksAndInds: [String: Int]
    var vocabs: [String: Int]?
    
    init(tokens: [String], indexes: [Int]) {
        self.padToken = tokens[0]
        self.sosToken = tokens[1]
        self.eosToken = tokens[2]
        self.unkToken = tokens[3]
        
        self.padIndex = indexes[0]
        self.sosIndex = indexes[1]
        self.eosIndex = indexes[2]
        self.unkIndex = indexes[3]
        
        self.toksAndInds = [
            tokens[0]: indexes[0],
            tokens[1]: indexes[1],
            tokens[2]: indexes[2],
            tokens[3]: indexes[3]
        ]
    }

    func filterSeq(_ seq: String) -> String {
        let charsToRemove = CharacterSet(charactersIn: "\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~\t\n")
        return seq.components(separatedBy: charsToRemove).joined()
    }

    func lowercaseSeq(_ seq: String) -> String {
        return seq.lowercased()
    }

    func importDataset(fromPath path: String) -> [[String: [String]]] {
        var examples: [[String: [String]]] = []
        let fileManager = FileManager.default
        let directoryURL = fileManager.urls(for: .documentDirectory, in: .userDomainMask).first!.appendingPathComponent(path)

        let enPath = directoryURL.appendingPathComponent("train.en").path
        let dePath = directoryURL.appendingPathComponent("train.de").path

        do {
            let enContent = try String(contentsOfFile: enPath, encoding: .utf8)
            let deContent = try String(contentsOfFile: dePath, encoding: .utf8)
            
            let enLines = enContent.components(separatedBy: "\n").map { self.filterSeq($0) }
            let deLines = deContent.components(separatedBy: "\n").map { self.filterSeq($0) }

            for (enLine, deLine) in zip(enLines, deLines) {
                examples.append(["en": enLine.split(separator: " ").map(String.init),
                                 "de": deLine.split(separator: " ").map(String.init)])
            }
        } catch {
            print("Error reading files: \(error)")
        }

        return examples
    }

    
    // Further methods to build vocabulary, prepare data batches, etc.
    
    func addTokens(to dataset: [[String: [String]]], batchSize: Int) -> [[[String: [String]]]] {
        var batches: [[[String: [String]]]] = []
        var currentBatch: [[String: [String]]] = []

        dataset.forEach { example in
            var modExample = example
            modExample["en"] = [sosToken] + (example["en"] ?? []) + [eosToken]
            modExample["de"] = [sosToken] + (example["de"] ?? []) + [eosToken]
            currentBatch.append(modExample)

            if currentBatch.count == batchSize {
                batches.append(currentBatch)
                currentBatch = []
            }
        }

        if !currentBatch.isEmpty {
            batches.append(currentBatch)
        }

        // Pad each batch
        batches = batches.map { batch in
            let maxEnLen = batch.max { $0["en"]!.count < $1["en"]!.count }!["en"]!.count
            let maxDeLen = batch.max { $0["de"]!.count < $1["de"]!.count }!["de"]!.count

            return batch.map { example in
                var example = example
                example["en"] = example["en"]! + Array(repeating: padToken, count: maxEnLen - example["en"]!.count)
                example["de"] = example["de"]! + Array(repeating: padToken, count: maxDeLen - example["de"]!.count)
                return example
            }
        }

        return batches
    }

    
    func buildDataset(from batchedData: [[[String: [String]]]], vocabs: [String: Int]) -> ([[[Int]]], [[[Int]]]) {
        var source: [[[Int]]] = []
        var target: [[[Int]]] = []
        
        for batch in batchedData {
            var sourceBatch: [[Int]] = []
            var targetBatch: [[Int]] = []
            
            for example in batch {
                let sourceIndices = example["en"]?.map { word -> Int in
                    vocabs[word] ?? unkIndex
                } ?? []
                
                let targetIndices = example["de"]?.map { word -> Int in
                    vocabs[word] ?? unkIndex
                } ?? []
                
                sourceBatch.append(sourceIndices)
                targetBatch.append(targetIndices)
            }
            
            source.append(sourceBatch)
            target.append(targetBatch)
        }
        
        return (source, target)
    }
    
    func buildVocab(from dataset: [[String: [String]]], minFreq: Int = 1) -> [String: Int] {
        var vocab: [String: Int] = toksAndInds
        var wordFrequencies: [String: Int] = [:]

        dataset.flatMap { $0.values.flatMap { $0 } }.forEach { word in
            wordFrequencies[word, default: 0] += 1
        }

        for (word, count) in wordFrequencies where count >= minFreq && vocab[word] == nil {
            vocab[word] = vocab.count + toksAndInds.count
        }

        return vocab
    }

}

// Usage
/*let tokens = ["<pad>", "<sos>", "<eos>", "<unk>"]
let indexes = [0, 1, 2, 3]
let dataPreparator = DataPreparator(tokens: tokens, indexes: indexes)

let dataset = dataPreparator.importDataset(fromPath: "./dataset")
// Process the dataset further as needed
let batchedData = dataPreparator.addTokens(to: dataset, batchSize: 32)
let (source, target) = dataPreparator.buildDataset(from: batchedData, vocabs: dataPreparator.vocabs ?? [:])*/

// Now `source` and `target` contain the indices of words ready for use in training





