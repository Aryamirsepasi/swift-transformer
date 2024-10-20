# Performance Measurements

For performance measurements, I relied on Xcode's Instrument panels, which provided all the necessary tools to evaluate the program's performance without needing any external software. In this folder, you will find several snapshots that capture key metrics such as CPU usage, memory consumption, and energy usage at various stages of the program.

Additionally, you can access a comprehensive set of detailed measurements in the `.trace` files available via the Google Drive link below. These files can be opened and analyzed using the Instruments app on macOS.

[View Performance Measurement Files](https://drive.google.com/drive/folders/1l94hx23KmZ6R0HTXCil01ruhZi1YzL-g?usp=share_link)

## Performance Comparison (MacBook Pro with M1 Pro CPU, 32 GB LPDDR5 RAM, macOS Sequoia 15.0.1)

#### Training:

- **227 Batches, Epoch 1/5 (Swift)**: Training completed in 121.486 seconds
- **227 Batches, Epoch 1/5 (Python)**: Training completed in 10,350.19 seconds
- **227 Batches, Epoch 2/5 (Swift)**: Training completed in 101.564 seconds
- **227 Batches, Epoch 2/5 (Python)**: Training completed in 28,678.68 seconds
- **227 Batches, Epoch 3/5 (Swift)**: Training completed in 133.440 seconds
- **227 Batches, Epoch 3/5 (Python)**: Training completed in 8324.21 seconds
- **227 Batches, Epoch 4/5 (Swift)**: Training completed in 184.864 seconds
- **227 Batches, Epoch 4/5 (Python)**: Training completed in 8312.97 seconds
- **227 Batches, Epoch 5/5 (Swift)**: Training completed in 1492.61 seconds
- **227 Batches, Epoch 5/5 (Python)**: Training completed in 19,090.69 seconds

#### Testing:

- **227 Batches, Epoch 1/5 (Swift)**: Testing completed in 2.226 seconds
- **227 Batches, Epoch 1/5 (Python)**: Testing completed in 303 seconds
- **227 Batches, Epoch 2/5 (Swift)**: Testing completed in 1.905 seconds
- **227 Batches, Epoch 2/5 (Python)**: Testing completed in 219 seconds
- **227 Batches, Epoch 3/5 (Swift)**: Testing completed in 1.682 seconds
- **227 Batches, Epoch 3/5 (Python)**: Testing completed in 218 seconds
- **227 Batches, Epoch 4/5 (Swift)**: Testing completed in 2.965 seconds
- **227 Batches, Epoch 4/5 (Python)**: Testing completed in 219 seconds
- **227 Batches, Epoch 5/5 (Swift)**: Testing completed in 36.33 seconds
- **227 Batches, Epoch 5/5 (Python)**: Testing completed in 219 seconds


#### Summary:

On average, the Swift variant is approximately **36.75×** faster during training and **26.12×** faster during testing compared to the Python variant.