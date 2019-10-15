package com.medi.MachineLearning.Layers;

import com.medi.MachineLearning.Neurons.BiasNeuron;
import com.medi.MachineLearning.Neurons.InputNeuron;
import com.medi.MachineLearning.Neurons.Neuron;

public class InputLayer extends Layer {
    public InputLayer(int neuronCount){
        size = new int[]{neuronCount};
        neurons = new Neuron[neuronCount];
        for(int i = 0;i<neuronCount;i++){
            neurons[i] = new InputNeuron();
            neurons[i].setNeuronPosition(i);
        }
    }

    public InputLayer(int[] size){
        this.size = size;
        int neuronCount = 1;
        for(int i : size)neuronCount*=i;
        neurons = new Neuron[neuronCount];
        for(int i = 0;i<neuronCount;i++){
            neurons[i] = new InputNeuron();
            neurons[i].setNeuronPosition(i);
        }
    }
}
