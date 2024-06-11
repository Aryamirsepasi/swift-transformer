//
//  ContentView.swift
//  metal-transformer
//
//  Created by Arya Mirsepasi on 26.04.24.
//

import SwiftUI

struct ContentView: View {
    @ObservedObject var viewModel = TransformerViewModel()

    var body: some View {
        VStack {

                Button("Start Training", action: {
                    viewModel.setupTransformer()
                })
                .padding()
                .background(Color.blue)
                .foregroundColor(.white)
                .cornerRadius(10)

        }
    }
}


