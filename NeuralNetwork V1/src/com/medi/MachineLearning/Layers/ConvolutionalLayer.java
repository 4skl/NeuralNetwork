package com.medi.MachineLearning.Layers;

import com.medi.MachineLearning.Exceptions.ArraysSizeDifferents;
import com.medi.MachineLearning.Functions;
import com.medi.MachineLearning.Neurons.*;

import java.io.FileWriter;

public class ConvolutionalLayer extends RetrobakeLayer{
    private int[][] filterSize;
    private double[][] weights;
    public ConvolutionalLayer(int[][] filterSize){//size : x-y-z-...
        weights = new double[filterSize.length][];
        for(int i = 0;i<filterSize.length;i++) {
            int weightsCount = 1;
            for (int j : filterSize[i]) weightsCount *= j;
            weights[i] = Functions.getRandomArray(weightsCount);
        }
        this.filterSize = filterSize;
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
        /*Neuron[] previousLayerNeurons = new Neuron[previousLayer.neuronCount()+1];
        BiasNeuron bias = new BiasNeuron();
        bias.setNeuronPosition(previousLayer.neuronCount());
        bias.setValue(1);
        for(int i = 0;i<previousLayerNeurons.length-1;i++){
            previousLayerNeurons[i] = previousLayer.getNeurons()[i];
        }*///need bias
        //Verify the sizes
        int[] size = previousLayer.getSize();
        int tmpNeuronCount = 1;
        for(int i : size) tmpNeuronCount*=i;
        if(previousLayer.neuronCount() != tmpNeuronCount) throw new ArraysSizeDifferents();

        neurons = new InputNeuron[tmpNeuronCount*filterSize.length];
        for(int i = 0;i<filterSize.length;i++) {
            double[] result = Functions.convolutionAgregation(previousLayer.getSize(), previousLayer.getValues(), filterSize[i], weights[i]);
            for (int j = 0; j < tmpNeuronCount; j++) {
                neurons[j].setValue(result[j]);
            }
        }



    }

    @Override
    public void retropropagation(RetrobakeLayer nextLayer) throws ArraysSizeDifferents {

    }

    @Override
    public void actualiseWeights(Layer previousLayer, double learningRate) {

    }

    @Override
    public void resetErrors() {

    }
}
