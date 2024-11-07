import SwiftUI
import Charts

struct ContentView: View {
    @ObservedObject var viewModel: TransformerViewModel
    
    var body: some View {
        VStack {
            Text("Training Progress")
                .font(.headline)
                .padding()
            
            if viewModel.lossData.isEmpty {
                ProgressView("Waiting for training data...")
            } else {
                Chart(viewModel.lossData) {
                    LineMark(
                        x: .value("Epoch", $0.epoch),
                        y: .value("Loss", $0.loss)
                    )
                    .foregroundStyle(by: .value("Series", $0.series))
                }
                .frame(height: 300)
                .padding()
                .chartXAxisLabel("Epoch")
                .chartYAxisLabel("Loss")
                // Remove fixed Y-axis range to allow dynamic scaling
                .chartLegend(position: .bottom)
            }
            
            Text("Latest Loss: \(viewModel.lossData.last?.loss ?? 0, specifier: "%.4f")")
                .padding()
        }
        .padding()
    }
}
