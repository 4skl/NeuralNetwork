package Network;

public class Neuron {
    double value;
    double[] weights;
    double[] weightsError;
    double bias;
    double biasError;

    public Neuron(double value){
        this.value = value;
    }


    public void setValue(double value) {
        this.value = value;
    }

    public double getValue() {
        return value;
    }

    public void setBias(double bias) {
        this.bias = bias;
    }

    public double getBias() {
        return bias;
    }

    public double getBiasError() {
        return biasError;
    }

    public void setWeights(double[] weights) {
        this.weights = weights;
    }

    public double[] getWeights() {
        return weights;
    }

    public double[] getWeightsError() {
        return weightsError;
    }

    public double calculate(double[] input){

    }

    public double calculate(Neuron[] input){
        double[] rawInput = new double[input.length];
        for(int i = 0;i<rawInput.length;i++){
            rawInput[i] = input[i].getValue();
        }
    }


    public void calculateError(){

    }

}
