import Foundation
import Accelerate
import MLX
import MLXRandom

//needed 
class Dropout {
    var rate: Float
    var inputShape: [Int]
    var outputShape: [Int]
    var mask: MLXArray
    var dataType: DType
    
    init(rate: Float = 0.1, dataType: DType = DType.float32) {
        
        print ("entered dropout init")

        self.rate = rate
        self.inputShape = []
        self.outputShape = []
        self.mask = []
        self.dataType = dataType
        
        print ("exited dropout init")

    }
    
    func build() {
        print ("entered dropout build")

        self.outputShape = self.inputShape
        
        print ("exited dropout build")

    }
    
    func forward(X: MLXArray, training: Bool = true) -> MLXArray {
        
        print ("entered dropout forward")

        //print(X.shape)
        
        var tempmask = 1.0
        
        if (training){
            self.mask = MLXRandom.bernoulli(1 - self.rate, X.shape, stream: .gpu).asType(self.dataType)
            
            //print((X * self.mask).shape)
            //print ("exited dropout forward")

            return X * self.mask
        }
        
        print ("exited dropout forward")

        return X * tempmask
        
    }
    
    func backward(_ error: MLXArray) -> MLXArray {
        
        print("entered dropout backward")

        return error * self.mask
    }

}
