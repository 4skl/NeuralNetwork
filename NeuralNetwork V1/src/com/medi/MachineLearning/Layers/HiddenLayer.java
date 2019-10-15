package com.medi.MachineLearning.Layers;

import com.medi.MachineLearning.Exceptions.ArraysSizeDifferents;
import com.medi.MachineLearning.Neurons.BiasNeuron;
import com.medi.MachineLearning.Neurons.HiddenNeuron;
import com.medi.MachineLearning.Neurons.Neuron;
import com.medi.MachineLearning.Neurons.RetrobakeNeuron;

public class HiddenLayer extends RetrobakeLayer{

    public HiddenLayer(int previousLayerNeuronCount, int neuronCount){
        neurons = new Neuron[neuronCount];
        for(int i = 0;i<neuronCount;i++){
            neurons[i] = new HiddenNeuron();
            ((RetrobakeNeuron) neurons[i]).setIncidentRandomWeights(previousLayerNeuronCount+1);
            neurons[i].setNeuronPosition(i);
        }
    }

    @Override
    public RetrobakeNeuron[] getRetrobakeNerons() {
        RetrobakeNeuron[] retrobakeNeurons = new RetrobakeNeuron[neurons.length];
        for(int i = 0; i<neurons.length;i++) {
            retrobakeNeurons[i] = (RetrobakeNeuron) neurons[i];
        }
        return retrobakeNeurons;
    }

    @Override
    public void propagation(Layer previousLayer) throws ArraysSizeDifferents {
        Neuron[] previousLayerNeurons = new Neuron[previousLayer.neuronCount()+1];
        BiasNeuron bias = new BiasNeuron();
        bias.setNeuronPosition(previousLayer.neuronCount());
        bias.setValue(1);
        for(int i = 0;i<previousLayerNeurons.length-1;i++){
            previousLayerNeurons[i] = previousLayer.getNeurons()[i];
        }
        previousLayerNeurons[previousLayerNeurons.length-1] = bias;
        for(int i = 0; i<neurons.length;i++) {
            ((RetrobakeNeuron) neurons[i]).bake(previousLayerNeurons);
        }
    }

    @Override
    public void retropropagation(RetrobakeLayer nextLayer) throws ArraysSizeDifferents {
        RetrobakeNeuron[] retrobakeNeurons = nextLayer.getRetrobakeNerons();
        for (int i = 0; i < neurons.length; i++) {
            ((RetrobakeNeuron) neurons[i]).retrobake(retrobakeNeurons);
        }
    }

    @Override
    public void actualiseWeights(Layer previousLayer,double learningRate){
        Neuron[] previousLayerNeurons = new Neuron[previousLayer.neuronCount()+1];
        BiasNeuron bias = new BiasNeuron();
        bias.setNeuronPosition(previousLayer.neuronCount());
        bias.setValue(1);
        for(int i = 0;i<previousLayerNeurons.length-1;i++){
            previousLayerNeurons[i] = previousLayer.getNeurons()[i];
        }
        previousLayerNeurons[previousLayerNeurons.length-1] = bias;
        for(int i = 0;i<neurons.length;i++){
            ((RetrobakeNeuron)neurons[i]).actualiseWeights(previousLayerNeurons,learningRate);
        }
    }

    @Override
    public void resetErrors(){
        for(int i = 0;i<neurons.length;i++){
            ((RetrobakeNeuron)neurons[i]).resetError();
        }
    }

}
