# Additional Notes and Acknowledgements

## Dependencies

The only external Swift package used in this project, aside from Apple's default packages like SwiftUI and Charts, is:

- [**MLX-Swift**](https://github.com/ml-explore/mlx-swift)

## File Path Considerations

Throughout the project, the use of the path `"./"` does not reference the project root folder by default. Xcode automatically sets this path to: 
`"/Users/UserName/Library/Containers/com.UserName.swift-transformer/Data"`

I did not change this path since I used the same directory. If you prefer to place the dataset in the project root folder instead of this path, you will need to adjust the file paths accordingly in the code.

## Dataset

The dataset can be downloaded from the original Python project created by AmritanshuV:

- [**Numpy-Transformer Dataset**](https://github.com/AmritanshuV/Numpy-Transformer)

## Acknowledgements

I would like to express my gratitude to the following individuals:

- **Dr. Harold Köstler**: For proposing this interesting and challenging project.
- **David Koski**: For assisting with issues related to the MLX library.
- **My Classmates**: For their helpful support throughout this project.
