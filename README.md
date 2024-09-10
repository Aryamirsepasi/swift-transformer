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

- **227 Batches, epoch 1/5 (Swift)**: Training completed in 1238.6281290054321 seconds
- **227 Batches, epoch 1/5 (Python)**: Training completed in 8225.50 seconds
