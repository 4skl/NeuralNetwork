package com.medi.MachineLearning.Layers;

import com.medi.MachineLearning.Neurons.Neuron;

import java.io.Serializable;

abstract public class Layer implements Serializable {

    public static final long serialVersionUID = 1L;

    protected int[] size;
    protected Neuron[] neurons;

    public int neuronCount(){
        return neurons.length;
    }
    int neuronCount(Class type){
        int count = 0;
        for(int i = 0;i<neurons.length;i++){
            if(type == neurons[i].getClass()){
                count++;
            }
        }
        return count;
    }
    public Neuron[] getNeurons(){
        return neurons;
    }


    public void setValues(double[] values){
        for(int i = 0;i<neurons.length;i++) {
            neurons[i].setValue(values[i]);
        }
    }

    public double[] getValues(){
        double[] values = new double[neurons.length];
        for(int i = 0;i<values.length;i++){
            values[i] = neurons[i].Value();
        }
        return values;
    }

    public int[] getSize(){
        return size;
    }

    public void setSize(int[] size){
        this.size = size;
    }


}
