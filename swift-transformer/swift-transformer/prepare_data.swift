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
        let fileManager = FileManager.default
        let trainPath = "\(path)/train.txt"  // Adjust file paths and extensions as necessary
        let data = try? String(contentsOfFile: trainPath, encoding: .utf8)
        var examples: [[String: [String]]] = []
        
        data?.enumerateLines { line, _ in
            let parts = line.components(separatedBy: "\t")
            if parts.count == 2 {
                let enSeq = self.filterSeq(parts[0])
                let deSeq = self.filterSeq(parts[1])
                examples.append(["en": enSeq.split(separator: "").map(String.init),
                                 "de": deSeq.split(separator: "").map(String.init)])
            }
        }
        
        return examples
    }
    
    // Further methods to build vocabulary, prepare data batches, etc.
    
    func addTokens(to dataset: [[String: [String]]], batchSize: Int) -> [[[String: [String]]]] {
        var batchedData: [[[String: [String]]]] = []
        var currentBatch: [[String: [String]]] = []
        
        for example in dataset {
            var modifiedExample = example
            modifiedExample["en"] = [sosToken] + (example["en"] ?? []) + [eosToken]
            modifiedExample["de"] = [sosToken] + (example["de"] ?? []) + [eosToken]
            currentBatch.append(modifiedExample)
            
            if currentBatch.count >= batchSize {
                batchedData.append(currentBatch)
                currentBatch = []
            }
        }
        
        if !currentBatch.isEmpty {
            batchedData.append(currentBatch)
        }
        
        return batchedData
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
        var vocab: [String: Int] = [:]
        var wordFrequencies: [String: Int] = [:]
        
        for example in dataset {
            guard let words = example["en"] else { continue }
            for word in words {
                wordFrequencies[word, default: 0] += 1
            }
        }
        
        for (word, count) in wordFrequencies {
            if count >= minFreq {
                if vocab[word] == nil {
                    vocab[word] = vocab.count
                }
            }
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





