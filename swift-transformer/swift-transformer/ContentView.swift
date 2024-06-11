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


