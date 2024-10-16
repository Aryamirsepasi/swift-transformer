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
        
        print ("entered positionwise_feed_forward init")

        self.fc1 = Dense(unitsNum: dFF, inputsNum: dModel)
        self.activation = ReLU()
        self.fc2 = Dense(unitsNum: dModel, inputsNum: dFF)
        
        self.dropout = Dropout(rate: dropoutRate)
        
        print ("exited positionwise_feed_forward init")

    }

    func forward(X: MLXArray, training: Bool = true) -> MLXArray {
        
        print ("entered positionwise_feed_forward forward")

        var x = X
        x = self.fc1.forward(X: x)
        x = self.activation.forward(x: x)
        x = self.dropout.forward(X: x)
        x = self.fc2.forward(X: x)
        
        print ("exited positionwise_feed_forward forward")

        return x
    }

    func backward(error: MLXArray) -> MLXArray{
        
        print("entered positionwise_feed_forward backward")
        //print("PWFF ERROR: ", error.shape)
        var err = fc2.backward(error: error)
        err = dropout.backward(err)
        err = activation.backward(grad: err)
        err = fc1.backward(error: err)
        
        print("exited positionwise_feed_forward backward")

        return err
    }

    func setOptimizer(optimizer: Optimizer) {
        print("entered positionwise_feed_forward setOptimizer")

        self.fc1.setOptimizer(optimizer: optimizer)
        self.fc2.setOptimizer(optimizer: optimizer)
        
        print("exited positionwise_feed_forward setOptimizer")

    }

    func updateWeights(startingLayerNum: Int) -> Int {
        
        print("entered positionwise_feed_forward updateWeights")

        var layerNum = startingLayerNum
        layerNum = fc1.updateWeights(layerNum: layerNum)
        layerNum = fc2.updateWeights(layerNum: layerNum)
        
        print("exited positionwise_feed_forward updateWeights")

        return layerNum
    }
}

