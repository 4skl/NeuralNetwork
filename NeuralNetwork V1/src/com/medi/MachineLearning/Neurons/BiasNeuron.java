package com.medi.MachineLearning.Neurons;

public class BiasNeuron extends Neuron {
    //x or sigmoid for values between 0 and 1, but in this case need to set value with setNotActivated()
    @Override
    public double activation(double x) {
        return x;
    }

    @Override
    public double derivativeActivation(double x) {
        return 1;
    }

    @Override
    public double reciprocalActivation(double x) {
        return x;
    }
}
