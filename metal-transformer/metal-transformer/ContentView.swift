//
//  ContentView.swift
//  metal-transformer
//
//  Created by Arya Mirsepasi on 26.04.24.
//

import SwiftUI
import Matft

func createAndManipulateMatrix() {
    let a = MfArray([[[ -8,  -7,  -6,  -5],
                      [ -4,  -3,  -2,  -1]],
            
                     [[ 0,  1,  2,  3],
                      [ 4,  5,  6,  7]]])
    let aa = Matft.arange(start: -8, to: 8, by: 1, shape: [2,2,4])
    
    print(a)
    print(aa)
}

struct ContentView: View {
    var body: some View {
        VStack {
            Image(systemName: "globe")
                .imageScale(.large)
                .foregroundStyle(.tint)
            Text("Hello, world!")
            Button("Run Matrix Function") {
                createAndManipulateMatrix()
            }
        }
        .padding()
        .onAppear {
            createAndManipulateMatrix()
        }
    }
}



#Preview {
    ContentView()
}
