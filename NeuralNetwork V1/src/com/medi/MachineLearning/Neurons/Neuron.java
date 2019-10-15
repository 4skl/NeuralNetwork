package com.medi.MachineLearning.Neurons;

import java.io.Serializable;

public abstract class Neuron implements Serializable {

    public static final long serialVersionUID = 2L;

    protected double value;
    protected double notActivated;
    protected int neuronPosition;


    public abstract double activation(double x);
    public abstract double derivativeActivation(double x);
    public abstract double reciprocalActivation(double x);

    public void setNeuronPosition(int position){
        neuronPosition = position;
    }

    public int neuronPosition(){
        return neuronPosition;
    }

    public void setValue(double value){
        this.value = value;
        this.notActivated = reciprocalActivation(value);
    }
    public void setNotActivated(double notActivated){
        this.notActivated = notActivated;
        this.value = activation(notActivated);
    }

    public double Value(){
        return value;
    }

    public double NotActivated(){
        return notActivated;
    }
}
