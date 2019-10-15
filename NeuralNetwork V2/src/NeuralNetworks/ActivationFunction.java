package NeuralNetworks;

public class ActivationFunction {
    ActivationType activationType;
    public ActivationFunction(ActivationType activationType){
        this.activationType = activationType;
    }

    public ActivationType getActivationType() {
        return activationType;
    }

    public double activation(double input) throws UnknownActivationType {
        switch (activationType){
            case ReLU:
                return input>0?input:0;
            case Sigmoid:
                return 1/(1+Math.exp(-input));
            case TanH:
                return Math.tanh(input);
            case BinaryStep:
                return input<0?0:1;
            default:
                throw new UnknownActivationType();
        }
    }

}

class UnknownActivationType extends Exception {

}
