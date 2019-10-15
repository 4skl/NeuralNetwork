package com.medi.MachineLearning.Layers;

import com.medi.MachineLearning.Exceptions.ArraysSizeDifferents;
import com.medi.MachineLearning.Neurons.Neuron;
import com.medi.MachineLearning.Neurons.OutputNeuron;
import com.medi.MachineLearning.Neurons.RetrobakeNeuron;

public class OutputLayer extends RetrobakeLayer {

    public OutputLayer(int previousLayerNeuronCount, int neuronCount){
        neurons = new OutputNeuron[neuronCount];
        for(int i = 0;i<neuronCount;i++){
            neurons[i] = new OutputNeuron();
            ((RetrobakeNeuron) neurons[i]).setIncidentRandomWeights(previousLayerNeuronCount);
            neurons[i].setNeuronPosition(i);
        }
    }

    @Override
    public RetrobakeNeuron[] getRetrobakeNerons() {
        return (RetrobakeNeuron[]) getNeurons();
    }

    @Override
    public void propagation(Layer previousLayer) throws ArraysSizeDifferents {
        for(int i = 0; i<neurons.length;i++){
            ((RetrobakeNeuron) neurons[i]).bake(previousLayer.getNeurons());
        }
    }

    @Override
    public void retropropagation(RetrobakeLayer nextLayer) throws ArraysSizeDifferents {
        Neuron[] retrobakeNeurons = nextLayer.getNeurons();
        for (int i = 0; i < neurons.length; i++) {
            ((OutputNeuron) neurons[i]).retrobake(retrobakeNeurons);
        }
    }

    public void retropropagation(Layer nextLayer) throws ArraysSizeDifferents {
        Neuron[] retrobakeNeurons = nextLayer.getNeurons();
        for (int i = 0; i < neurons.length; i++) {
            ((OutputNeuron) neurons[i]).retrobake(retrobakeNeurons);
        }
    }

    public void retropropagation(double cost){
        for (int i = 0; i < neurons.length; i++) {
            ((OutputNeuron) neurons[i]).retrobake(cost);
        }
    }

    public void retropropagation(double[] need){
        for (int i = 0; i < neurons.length; i++) {
            ((OutputNeuron) neurons[i]).retrobake(need[i]);
        }
    }

    @Override
    public void actualiseWeights(Layer previousLayer,double learningRate){
        for(int i = 0;i<neurons.length;i++){
            ((RetrobakeNeuron)neurons[i]).actualiseWeights(previousLayer.getNeurons(),learningRate);
        }
    }

    @Override
    public void resetErrors(){
        for(int i = 0;i<neurons.length;i++){
            ((RetrobakeNeuron)neurons[i]).resetError();
        }
    }

}
