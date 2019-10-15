package NeuralNetworks;

import java.io.Serializable;

public interface Layer extends Serializable {
    Tensor propagate(Tensor inputTensor);
    Tensor backpropagate(Tensor outputTensor);
    void learn();
    void clear();
}
