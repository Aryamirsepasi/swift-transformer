# Swift-Transformer

Swift-Transformer is a comprehensive Swift-based project designed to bring advanced machine learning capabilities to Apple's ecosystem, specifically optimized for Apple Silicon devices. This project leverages [MLX-Swift](https://github.com/ml-explore/mlx-swift/tree/db6e838c7bbfc1bb8a1475bfa7cea0baf1ba8835), an array framework for machine learning research, ensuring seamless integration and performance enhancements on macOS platforms.

## Features

- **Native Swift Implementation**: Entirely rewritten in Swift, leveraging powerful features of the language for high performance.
- **Apple Silicon Optimization**: Specifically optimized for Apple Silicon, utilizing the full potential of the latest hardware accelerations.
- **MLX-Swift Integration**: Replaces Numpy with MLX-Swift, a Swift library designed for array operations in machine learning research, providing faster and more efficient computations on Apple Silicon.
- **Native GPU Utilization**: By default, MLX-Swift leverages the native GPU capabilities of Apple Silicon, resulting in significantly faster and more efficient performance compared to standard Numpy and Python implementations.
- **Metal Support**: Transitions from traditional GPU usage to Apple’s Metal, enhancing computational capabilities for machine learning tasks.

Based on the original Numpy-Transformer project by AmritanshuV. [Link](https://github.com/AmritanshuV/Numpy-Transformer)

**For additional notes, dependencies, and acknowledgements, please refer to [NOTES.md](NOTES.md).**

## Performance Comparison (MacBook Pro with M1 Pro CPU, 32 GB LPDDR5 RAM, macOS Sonoma 14.6.1)

#### Training:

- **227 Batches, Epoch 1/5 (Swift)**: Training completed in 1219.46 seconds
- **227 Batches, Epoch 1/5 (Python)**: Training completed in 10,350.19 seconds
- **227 Batches, Epoch 2/5 (Swift)**: Training completed in 1291.44 seconds
- **227 Batches, Epoch 2/5 (Python)**: Training completed in 28,678.68 seconds
- **227 Batches, Epoch 3/5 (Swift)**: Training completed in 1365.42 seconds
- **227 Batches, Epoch 3/5 (Python)**: Training completed in 8324.21 seconds
- **227 Batches, Epoch 4/5 (Swift)**: Training completed in 1439.86 seconds
- **227 Batches, Epoch 4/5 (Python)**: Training completed in 8312.97 seconds
- **227 Batches, Epoch 5/5 (Swift)**: Training completed in 1492.61 seconds
- **227 Batches, Epoch 5/5 (Python)**: Training completed in 19,090.69 seconds

#### Testing:

- **227 Batches, Epoch 1/5 (Swift)**: Testing completed in 35.09 seconds
- **227 Batches, Epoch 1/5 (Python)**: Testing completed in 303 seconds
- **227 Batches, Epoch 2/5 (Swift)**: Testing completed in 34.56 seconds
- **227 Batches, Epoch 2/5 (Python)**: Testing completed in 219 seconds
- **227 Batches, Epoch 3/5 (Swift)**: Testing completed in 36.55 seconds
- **227 Batches, Epoch 3/5 (Python)**: Testing completed in 218 seconds
- **227 Batches, Epoch 4/5 (Swift)**: Testing completed in 38.06 seconds
- **227 Batches, Epoch 4/5 (Python)**: Testing completed in 219 seconds
- **227 Batches, Epoch 5/5 (Swift)**: Testing completed in 36.33 seconds
- **227 Batches, Epoch 5/5 (Python)**: Testing completed in 219 seconds

#### Summary:

On average, the Swift variant is approximately **8.92×** faster during training and **6.54×** faster during testing compared to the Python variant.

## Loss History Graph

The graph below illustrates the training and validation loss over the epochs:

![Loss History Graph](loss_history.png)
