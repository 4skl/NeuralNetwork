package NeuralNetworks.LayerTypes;

import NeuralNetworks.ActivationType;
import NeuralNetworks.Layer;
import NeuralNetworks.LossType;
import NeuralNetworks.Tensor;

public class FullyConnectedLayer implements Layer {
    double errors[];
    double weights[];
    double biases[];
    boolean dropout = false;
    ActivationType activationType = ActivationType.Sigmoid;
    LossType lossType = LossType.Quadratic;

    private int inputSize, outputSize;

    FullyConnectedLayer(int inputSize, int outputSize, ActivationType activationType, LossType lossType) {
        this.activationType = activationType;
        this.inputSize = inputSize;
        this.outputSize = outputSize;
    }

    FullyConnectedLayer(int inputSize, int outputSize, ActivationType activationType, boolean dropout) {
        this.activationType = activationType;
        this.dropout = dropout;
        this.inputSize = inputSize;
        this.outputSize = outputSize;
    }

    //can possibly remove input size with inputTensor
    @Override
    public Tensor propagate(Tensor inputTensor) {
        Tensor outputTensor = new Tensor(new int[]{outputSize});
        for(int i = 0;i<outputSize;i++){
            double value = 0;
            for(int j = 0;j<inputSize;j++){
                value+=inputTensor.getDataAt(j)*weights[j+i*inputSize];
            }
            value+=biases[i];
            outputTensor.setDataAt(value,i);
        }
        return null;
    }

    @Override
    public Tensor backpropagate(Tensor outputTensor) {
        return null;
    }

    @Override
    public void learn() {

    }

    @Override
    public void clear() {

    }

    private void initializeNetworkRandom(int inputSize, int outputSize){
        weights = new double[inputSize*outputSize];
        for(int i = 0;i<weights.length;i++){
            weights[i] = Math.random();
        }
        biases = new double[outputSize];
        for (int i = 0;i<biases.length;i++){
            biases[i] = Math.random();
        }
        errors = new double[outputSize];
    }

}
