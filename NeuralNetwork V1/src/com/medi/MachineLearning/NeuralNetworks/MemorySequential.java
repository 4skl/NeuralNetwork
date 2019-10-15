package com.medi.MachineLearning.NeuralNetworks;

import com.medi.MachineLearning.Exceptions.ArraysSizeDifferents;
import com.medi.MachineLearning.Functions;
import com.medi.MachineLearning.Layers.*;

public class MemorySequential {

    double[] memory;
    double oldNext;

    private Layer[] memoryNetwork,network;
    public MemorySequential(int memorySize, int[] memoryHiddenLayers, int[] networkHiddenLayers){

        //Memory
        memory = new double[memorySize];
        //Network for memory
        memoryNetwork = new Layer[memoryHiddenLayers.length+2];
        int[] memoryRepresentation = new int[memoryHiddenLayers.length+2];
        memoryRepresentation[memoryRepresentation.length-1] = memorySize;//The memory
        memoryRepresentation[0] = memorySize+1;//The last memory + the output
        for(int i = 0;i<memoryHiddenLayers.length;i++){
            memoryRepresentation[i+1] = memoryHiddenLayers[i];
        }
        memoryNetwork[0] = new InputLayer(memoryRepresentation[0]);
        for(int i = 1; i<memoryNetwork.length;i++){
            memoryNetwork[i] = new HiddenLayer(memoryRepresentation[i-1],memoryRepresentation[i]);
        }

        //Network
        network = new Layer[networkHiddenLayers.length+2];
        int[] representation = new int[networkHiddenLayers.length+2];
        representation[representation.length-1] = 1;//The output
        representation[0] = memorySize;//The memory
        for(int i = 0;i<networkHiddenLayers.length;i++){
            representation[i+1] = networkHiddenLayers[i];
        }
        network[0] = new InputLayer(memorySize);
        for(int i = 1; i<network.length-1;i++){
            network[i] = new HiddenLayer(representation[i-1],representation[i]);
        }
        network[network.length-1] = new OutputLayer(network[network.length-2].neuronCount(),representation[representation.length-1]);

    }

    public void learnSequence(double[] sequence,double learningRate) throws ArraysSizeDifferents {
        for(int i = 0;i<sequence.length;i++){
            setNext(sequence[i],learningRate);
        }
    }

    public double[] getSequence(int length) throws ArraysSizeDifferents {
        double[] sequence = new double[length];
        for(int i = 0;i<length;i++){
            sequence[i] = getNext();
            oldNext = sequence[i];
        }
        return sequence;
    }

    public void setNext(double next,double learningRate) throws ArraysSizeDifferents {
        memory = propagateMemory();
        propagateNetwork();
        retropropagateAll(next);
        actualiseWeightsAndResetErrorsAll(learningRate);
    }
    public double getNext() throws ArraysSizeDifferents {
        memory = propagateMemory();
        return propagateNetwork()[0];
    }

    public double[] propagateMemory() throws ArraysSizeDifferents {
        memoryNetwork[0].setValues(Functions.concatenate(memory,new double[]{oldNext}));
        for(int i = 1;i<memoryNetwork.length;i++){
            ((RetrobakeLayer)memoryNetwork[i]).propagation(memoryNetwork[i-1]);
        }
        return memoryNetwork[memoryNetwork.length-1].getValues();
    }

    public double[] propagateNetwork() throws ArraysSizeDifferents {
        network[0].setValues(memory);
        for(int i = 1;i<network.length;i++){
            ((RetrobakeLayer)network[i]).propagation(network[i-1]);
        }
        return network[network.length-1].getValues();
    }

    public void retropropagateAll(double nededOutput) throws ArraysSizeDifferents {
        double[] outp = new double[]{nededOutput};
        ((OutputLayer) network[network.length-1]).retropropagation(outp);
        for(int i = network.length-2;i>0;i--){
            ((RetrobakeLayer) network[i]).retropropagation((RetrobakeLayer)network[i+1]);
        }
        ((RetrobakeLayer) memoryNetwork[memoryNetwork.length-1]).retropropagation((RetrobakeLayer)network[1]);
        for(int i = memoryNetwork.length-2;i>0;i--){
            ((RetrobakeLayer) memoryNetwork[i]).retropropagation((RetrobakeLayer)memoryNetwork[i+1]);
        }
    }

    public void actualiseWeightsAndResetErrorsAll(double learningRate){
        for(int i = 1;i<memoryNetwork.length;i++){
            ((RetrobakeLayer)memoryNetwork[i]).actualiseWeights(memoryNetwork[i-1],learningRate);
        }
        //Memory ?
        for(int i = 1;i<network.length;i++){
            ((RetrobakeLayer)network[i]).actualiseWeights(network[i-1],learningRate);
        }

        for(int i = 1;i<memoryNetwork.length;i++){
            ((RetrobakeLayer)memoryNetwork[i]).resetErrors();
        }
        for(int i = 1;i<network.length;i++){
            ((RetrobakeLayer)network[i]).resetErrors();
        }
    }

}
