# Swift-Transformer

Swift-Transformer is a comprehensive Swift-based project designed to bring advanced machine learning capabilities to Apple's ecosystem, specifically optimized for Apple Silicon devices. This project leverages [MLX-Swift](https://github.com/ml-explore/mlx-swift/tree/db6e838c7bbfc1bb8a1475bfa7cea0baf1ba8835), an array framework for machine learning research, ensuring seamless integration and performance enhancements on macOS platforms.

## Features

- **Native Swift Implementation**: Entirely rewritten in Swift, leveraging powerful features of the language for high performance.
- **Apple Silicon Optimization**: Specifically optimized for Apple Silicon, utilizing the full potential of the latest hardware accelerations.
- **MLX-Swift Integration**: Replaces Numpy with MLX-Swift, a Swift library designed for array operations in machine learning research, providing faster and more efficient computations on Apple Silicon.
- **Native GPU Utilization**: By default, MLX-Swift leverages the native GPU capabilities of Apple Silicon, resulting in significantly faster and more efficient performance compared to standard Numpy and Python implementations.
- **Metal Support**: Transitions from traditional GPU usage to Apple’s Metal, enhancing computational capabilities for machine learning tasks.

Based on the original Numpy-Transformer Project by AmritanshuV [Link](https://github.com/AmritanshuV/Numpy-Transformer)


## Performance Comparison (MacBook Pro with M1 Pro CPU, 32 GB LPDDR5 RAM, MacOS Sonoma 14.6.1)

#### Training:
- **227 Batches, epoch 1/5 (Swift)**: Training completed in 1219.4579420089722 seconds
- **227 Batches, epoch 1/5 (Python)**: Training completed in 8225.50 seconds
- **227 Batches, epoch 2/5 (Swift)**: Training completed in 1291.4347389936447 seconds
- **227 Batches, epoch 2/5 (Python)**: Training completed in 33907.19 seconds
- **227 Batches, epoch 3/5 (Swift)**: Training completed in 1365.4138170480728 seconds
- **227 Batches, epoch 3/5 (Python)**: Training completed in - seconds
- **227 Batches, epoch 4/5 (Swift)**: Training completed in 1439.8610639572144 seconds
- **227 Batches, epoch 4/5 (Python)**: Training completed in - seconds
- **227 Batches, epoch 5/5 (Swift)**: Training completed in 1492.612431049347 seconds
- **227 Batches, epoch 5/5 (Python)**: Training completed in - seconds

#### Testing:
- **227 Batches, epoch 1/5 (Swift)**: Testing completed in 35.09237003326416 seconds
- **227 Batches, epoch 1/5 (Python)**: Testing completed in 303 seconds
- **227 Batches, epoch 2/5 (Swift)**: Testing completed in 34.56067502498627 seconds
- **227 Batches, epoch 2/5 (Python)**: Testing completed in - seconds
- **227 Batches, epoch 3/5 (Swift)**: Testing completed in 36.549415946006775 seconds
- **227 Batches, epoch 3/5 (Python)**: Testing completed in - seconds
- **227 Batches, epoch 4/5 (Swift)**: Testing completed in 38.06288194656372 seconds
- **227 Batches, epoch 4/5 (Python)**: Testing completed in - seconds
- **227 Batches, epoch 5/5 (Swift)**: Testing completed in 36.32816696166992 seconds
- **227 Batches, epoch 5/5 (Python)**: Testing completed in - seconds

