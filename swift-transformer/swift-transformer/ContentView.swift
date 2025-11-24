import SwiftUI
import Charts

// MARK: - Log Manager for capturing console output
class LogManager: ObservableObject {
    static let shared = LogManager()
    
    @Published var logs: [LogEntry] = []
    
    struct LogEntry: Identifiable {
        let id = UUID()
        let message: String
        let timestamp: Date
    }
    
    func addLog(_ message: String) {
        DispatchQueue.main.async {
            let entry = LogEntry(message: message, timestamp: Date())
            self.logs.append(entry)
            // Keep only last 500 logs to prevent memory issues
            if self.logs.count > 500 {
                self.logs.removeFirst(self.logs.count - 500)
            }
        }
    }
    
    func clear() {
        DispatchQueue.main.async {
            self.logs.removeAll()
        }
    }
}

// Custom print function that also logs to LogManager
func logPrint(_ items: Any..., separator: String = " ", terminator: String = "\n") {
    let message = items.map { "\($0)" }.joined(separator: separator)
    Swift.print(message, terminator: terminator)
    LogManager.shared.addLog(message)
}

// MARK: - Training State
enum TrainingState: Equatable {
    case idle
    case preparingData
    case training(epoch: Int, totalEpochs: Int)
    case evaluating
    case completed
    case error(String)
    
    var description: String {
        switch self {
        case .idle:
            return "Ready to start"
        case .preparingData:
            return "Preparing data..."
        case .training(let epoch, let total):
            return "Training epoch \(epoch)/\(total)"
        case .evaluating:
            return "Evaluating model..."
        case .completed:
            return "Training completed!"
        case .error(let message):
            return "Error: \(message)"
        }
    }
}

// MARK: - Content View
struct ContentView: View {
    @ObservedObject var viewModel: TransformerViewModel
    @ObservedObject var logManager = LogManager.shared
    @State private var userInput: String = ""
    @State private var translationResult: String = ""
    @State private var isTranslating: Bool = false
    @FocusState private var isInputFocused: Bool
    
    var body: some View {
        NavigationSplitView {
            // Sidebar with logs
            VStack(alignment: .leading, spacing: 0) {
                HStack {
                    Text("Training Logs")
                        .font(.headline)
                    Spacer()
                    Button(action: { logManager.clear() }) {
                        Image(systemName: "trash")
                            .foregroundColor(.secondary)
                    }
                    .buttonStyle(.plain)
                }
                .padding()
                
                Divider()
                
                ScrollViewReader { proxy in
                    ScrollView {
                        LazyVStack(alignment: .leading, spacing: 4) {
                            ForEach(logManager.logs) { entry in
                                Text(entry.message)
                                    .font(.system(.caption, design: .monospaced))
                                    .foregroundColor(.secondary)
                                    .textSelection(.enabled)
                                    .id(entry.id)
                            }
                        }
                        .padding(.horizontal)
                    }
                    .onChange(of: logManager.logs.count) { _, _ in
                        if let lastLog = logManager.logs.last {
                            withAnimation {
                                proxy.scrollTo(lastLog.id, anchor: .bottom)
                            }
                        }
                    }
                }
            }
            .frame(minWidth: 300)
        } detail: {
            // Main content
            VStack(spacing: 20) {
                // Header with status
                headerSection
                
                Divider()
                
                if viewModel.trainingState == .completed {
                    // Translation interface
                    translationSection
                } else {
                    // Training progress
                    trainingProgressSection
                }
                
                Spacer()
            }
            .padding()
        }
        .onAppear {
            viewModel.setupTransformer()
        }
    }
    
    // MARK: - Header Section
    private var headerSection: some View {
        HStack {
            VStack(alignment: .leading) {
                Text("Swift Transformer")
                    .font(.largeTitle)
                    .fontWeight(.bold)
                
                Text("English → German Translation")
                    .font(.subheadline)
                    .foregroundColor(.secondary)
            }
            
            Spacer()
            
            // Status indicator
            HStack(spacing: 8) {
                statusIndicator
                Text(viewModel.trainingState.description)
                    .font(.subheadline)
            }
            .padding(.horizontal, 12)
            .padding(.vertical, 8)
            .background(statusBackgroundColor)
            .cornerRadius(8)
        }
    }
    
    @ViewBuilder
    private var statusIndicator: some View {
        switch viewModel.trainingState {
        case .idle:
            Circle()
                .fill(.gray)
                .frame(width: 10, height: 10)
        case .preparingData, .training, .evaluating:
            ProgressView()
                .scaleEffect(0.6)
        case .completed:
            Image(systemName: "checkmark.circle.fill")
                .foregroundColor(.green)
        case .error:
            Image(systemName: "exclamationmark.triangle.fill")
                .foregroundColor(.red)
        }
    }
    
    private var statusBackgroundColor: Color {
        switch viewModel.trainingState {
        case .completed:
            return .green.opacity(0.1)
        case .error:
            return .red.opacity(0.1)
        default:
            return .blue.opacity(0.1)
        }
    }
    
    // MARK: - Training Progress Section
    private var trainingProgressSection: some View {
        VStack(spacing: 20) {
            if viewModel.lossData.isEmpty && viewModel.totalBatches == 0 {
                VStack(spacing: 16) {
                    ProgressView()
                        .scaleEffect(1.5)
                    Text("Initializing training...")
                        .foregroundColor(.secondary)
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
            } else {
                // Batch progress bar
                if viewModel.totalBatches > 0 {
                    VStack(alignment: .leading, spacing: 8) {
                        HStack {
                            if case .training(let epoch, let total) = viewModel.trainingState {
                                Text("Epoch \(epoch)/\(total)")
                                    .font(.headline)
                            }
                            Spacer()
                            Text("Batch \(viewModel.currentBatch)/\(viewModel.totalBatches)")
                                .font(.subheadline)
                                .foregroundColor(.secondary)
                        }
                        
                        ProgressView(value: Double(viewModel.currentBatch), total: Double(viewModel.totalBatches))
                            .progressViewStyle(.linear)
                            .tint(.blue)
                        
                        HStack {
                            Text("Loss: \(String(format: "%.4f", viewModel.currentLoss))")
                                .font(.caption)
                                .foregroundColor(.secondary)
                            Spacer()
                            Text("Perplexity: \(String(format: "%.2f", exp(viewModel.currentLoss)))")
                                .font(.caption)
                                .foregroundColor(.secondary)
                        }
                    }
                    .padding()
                    .background(Color.blue.opacity(0.05))
                    .cornerRadius(12)
                }
                
                // Loss chart
                if !viewModel.lossData.isEmpty {
                    VStack(alignment: .leading) {
                        Text("Training Progress")
                            .font(.headline)
                    
                        Chart(viewModel.lossData) {
                            LineMark(
                                x: .value("Epoch", $0.epoch),
                                y: .value("Loss", $0.loss)
                            )
                            .foregroundStyle(by: .value("Series", $0.series))
                            
                            PointMark(
                                x: .value("Epoch", $0.epoch),
                                y: .value("Loss", $0.loss)
                            )
                            .foregroundStyle(by: .value("Series", $0.series))
                        }
                        .frame(height: 300)
                        .chartXAxisLabel("Epoch")
                        .chartYAxisLabel("Loss")
                        .chartLegend(position: .bottom)
                    }
                    .padding()
                    .background(Color.gray.opacity(0.05))
                    .cornerRadius(12)
                }
                
                // Stats
                HStack(spacing: 40) {
                    StatCard(
                        title: "Current Loss",
                        value: String(format: "%.4f", viewModel.currentLoss)
                    )
                    
                    StatCard(
                        title: "Perplexity",
                        value: String(format: "%.2f", exp(viewModel.currentLoss))
                    )
                    
                    if case .training(let epoch, let total) = viewModel.trainingState {
                        StatCard(
                            title: "Epoch",
                            value: "\(epoch)/\(total)"
                        )
                    }
                    
                    if viewModel.totalBatches > 0 {
                        StatCard(
                            title: "Batch",
                            value: "\(viewModel.currentBatch)/\(viewModel.totalBatches)"
                        )
                    }
                }
            }
        }
    }
    
    // MARK: - Translation Section
    private var translationSection: some View {
        VStack(spacing: 20) {
            Text("Translation Ready!")
                .font(.title2)
                .foregroundColor(.green)
            
            // Input field
            VStack(alignment: .leading, spacing: 8) {
                Text("Enter English text:")
                    .font(.headline)
                
                HStack {
                    TextField("Type your sentence here...", text: $userInput)
                        .textFieldStyle(.roundedBorder)
                        .focused($isInputFocused)
                        .onSubmit {
                            translateText()
                        }
                    
                    Button(action: translateText) {
                        if isTranslating {
                            ProgressView()
                                .scaleEffect(0.8)
                        } else {
                            Text("Translate")
                        }
                    }
                    .buttonStyle(.borderedProminent)
                    .disabled(userInput.isEmpty || isTranslating)
                }
            }
            
            // Result display
            if !translationResult.isEmpty {
                VStack(alignment: .leading, spacing: 8) {
                    Text("German Translation:")
                        .font(.headline)
                    
                    Text(translationResult)
                        .font(.body)
                        .padding()
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .background(Color.blue.opacity(0.1))
                        .cornerRadius(8)
                        .textSelection(.enabled)
                }
            }
            
            Divider()
            
            // Example translations section
            VStack(alignment: .leading, spacing: 12) {
                Text("Example Translations")
                    .font(.headline)
                
                ForEach(viewModel.exampleTranslations, id: \.input) { example in
                    ExampleTranslationView(example: example)
                }
            }
        }
        .padding()
        .background(Color.gray.opacity(0.05))
        .cornerRadius(12)
    }
    
    private func translateText() {
        guard !userInput.isEmpty else { return }
        
        isTranslating = true
        let inputText = userInput.lowercased()
            .components(separatedBy: CharacterSet(charactersIn: "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~\t\n"))
            .joined()
            .split(separator: " ")
            .map(String.init)
        
        DispatchQueue.global(qos: .userInitiated).async {
            if let result = viewModel.translate(sentence: inputText) {
                DispatchQueue.main.async {
                    translationResult = result.joined(separator: " ")
                    isTranslating = false
                }
            } else {
                DispatchQueue.main.async {
                    translationResult = "Translation failed. Please try again."
                    isTranslating = false
                }
            }
        }
    }
}

// MARK: - Supporting Views
struct StatCard: View {
    let title: String
    let value: String
    
    var body: some View {
        VStack(spacing: 4) {
            Text(title)
                .font(.caption)
                .foregroundColor(.secondary)
            Text(value)
                .font(.title3)
                .fontWeight(.semibold)
        }
        .padding()
        .background(Color.gray.opacity(0.1))
        .cornerRadius(8)
    }
}

struct ExampleTranslation: Equatable {
    let input: String
    let output: String
    let target: String
}

struct ExampleTranslationView: View {
    let example: ExampleTranslation
    
    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack {
                Text("EN:")
                    .font(.caption)
                    .foregroundColor(.secondary)
                    .frame(width: 30, alignment: .leading)
                Text(example.input)
                    .font(.caption)
            }
            HStack {
                Text("DE:")
                    .font(.caption)
                    .foregroundColor(.blue)
                    .frame(width: 30, alignment: .leading)
                Text(example.output)
                    .font(.caption)
                    .foregroundColor(.blue)
            }
            HStack {
                Text("REF:")
                    .font(.caption)
                    .foregroundColor(.secondary)
                    .frame(width: 30, alignment: .leading)
                Text(example.target)
                    .font(.caption)
                    .foregroundColor(.secondary)
            }
        }
        .padding(8)
        .background(Color.gray.opacity(0.05))
        .cornerRadius(6)
    }
}
