import SwiftUI
import Charts

struct ContentView: View {
    @ObservedObject var viewModel: TransformerViewModel
    
    var body: some View {
        VStack {
            // Use Swift Charts to plot the graph
            Chart(viewModel.lossData) {
                LineMark(
                    x: .value("Epoch", $0.epoch),
                    y: .value("Loss", $0.loss)
                )
                .foregroundStyle(by: .value("Series", $0.series)) // Differentiate lines by series
            }
            .frame(height: 200)
            .chartXAxisLabel("Epoch")
            .chartYAxisLabel("Loss")
            
            
        }
    }
    
}
