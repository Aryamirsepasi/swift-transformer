import Foundation
import Accelerate
import MLX

//needed
class PositionwiseFeedforward {
    var fc1: Dense
    var activation: Activation
    var dropout: Dropout
    var fc2: Dense

    init(dModel: Int = 512, dFF: Int = 2048, dropoutRate: Float = 0.1) {
        
        self.fc1 = Dense(unitsNum: dFF, inputsNum: dModel)
        self.activation = ReLU()
        self.fc2 = Dense(unitsNum: dModel, inputsNum: dFF)
        
        self.dropout = Dropout(rate: dropoutRate)
        
    }

    func forward(X: MLXArray, training: Bool = true) -> MLXArray {
        
        var x = X
        x = self.fc1.forward(X: x)
        x = self.activation.forward(x: x)
        x = self.dropout.forward(X: x)
        x = self.fc2.forward(X: x)
        
        return x
    }

    func backward(error: MLXArray) -> MLXArray{
        
        var err = fc2.backward(error: error)
        err = dropout.backward(err)
        err = activation.backward(grad: err)
        err = fc1.backward(error: err)
        
        return err
    }

    func setOptimizer(optimizer: Optimizer) {

        self.fc1.setOptimizer(optimizer: optimizer)
        self.fc2.setOptimizer(optimizer: optimizer)
        
    }

    func updateWeights(startingLayerNum: Int) -> Int {
        autoreleasepool {
            
            var layerNum = startingLayerNum
            layerNum = fc1.updateWeights(layerNum: layerNum)
            layerNum = fc2.updateWeights(layerNum: layerNum)
            
            return layerNum
        }
    }
}

