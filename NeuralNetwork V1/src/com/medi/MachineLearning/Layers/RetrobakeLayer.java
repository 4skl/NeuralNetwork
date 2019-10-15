package com.medi.MachineLearning.Layers;

import com.medi.MachineLearning.Exceptions.ArraysSizeDifferents;
import com.medi.MachineLearning.Neurons.RetrobakeNeuron;

public abstract class RetrobakeLayer extends Layer{
    public abstract RetrobakeNeuron[] getRetrobakeNerons();

    public double[] getErrors(){
        RetrobakeNeuron[] retrobakeNeurons = getRetrobakeNerons();
        double[] errors = new double[retrobakeNeurons.length];
        for(int i = 0;i<errors.length;i++){
            errors[i] = retrobakeNeurons[i].Error();
        }
        return errors;
    }
    public void setErrors(double[] errors){
        RetrobakeNeuron[] retrobakeNeurons = getRetrobakeNerons();
        for(int i = 0;i<retrobakeNeurons.length;i++) {
            retrobakeNeurons[i].setError(errors[i]);
        }
    }

    public void setIncidentWeights(double incidentWeights[][]) throws ArraysSizeDifferents {
        RetrobakeNeuron[] retrobakeNeuron = getRetrobakeNerons();
        if(incidentWeights.length!=retrobakeNeuron.length)throw new ArraysSizeDifferents();
        for(int i = 0;i<incidentWeights.length;i++){
            retrobakeNeuron[i].setIncidentWeights(incidentWeights[i]);
        }
    }

    public double[][] getIncidentWeights(){
        RetrobakeNeuron[] retrobakeNeuron = getRetrobakeNerons();
        double[][] incidentWeights = new double[retrobakeNeuron.length][];
        for(int i = 0;i<retrobakeNeuron.length;i++){
            incidentWeights[i] = retrobakeNeuron[i].IncidentWeights();
        }
        return incidentWeights;
    }

    public abstract void propagation(Layer previousLayer) throws ArraysSizeDifferents;
    public abstract void retropropagation(RetrobakeLayer nextLayer) throws ArraysSizeDifferents;
    public abstract void actualiseWeights(Layer previousLayer, double learningRate);
    public abstract void resetErrors();
}
/*class PropagateNeuronRunnable implements Runnable{
    RetrobakeNeuron neuron;
    Layer previousLayer;
    PropagateNeuronRunnable(RetrobakeNeuron neuron,Layer previousLayer){
        this.neuron=neuron;
        this.previousLayer=previousLayer;
    }

    @Override
    public void run() {
        try {
            neuron.bake(previousLayer.getNeurons());
        } catch (ArraysSizeDifferents e) {
            e.printStackTrace();
        }
    }
}

class RetropropagateNeuronRunnable implements Runnable{
    RetrobakeNeuron neuron;
    RetrobakeLayer nextLayer;
    RetropropagateNeuronRunnable(RetrobakeNeuron neuron,RetrobakeLayer nextLayer){
        this.neuron=neuron;
        this.nextLayer=nextLayer;
    }

    @Override
    public void run() {
        try {
            neuron.retrobake(nextLayer.getRetrobakeNerons());
        } catch (ArraysSizeDifferents e) {
            e.printStackTrace();
        }
    }
}

class ActualiseNeuronRunnable implements Runnable{
    RetrobakeNeuron neuron;
    Layer previousLayer;
    double learningRate;
    ActualiseNeuronRunnable(RetrobakeNeuron neuron,Layer previousLayer,double learningRate){
        this.neuron=neuron;
        this.previousLayer=previousLayer;
        this.learningRate=learningRate;
    }

    @Override
    public void run() {
        neuron.actualiseWeights(previousLayer.getNeurons(),learningRate);
    }
}

class ResetNeuronRunnable implements Runnable{
    RetrobakeNeuron neuron;
    ResetNeuronRunnable(RetrobakeNeuron neuron){
        this.neuron=neuron;
    }

    @Override
    public void run() {
        neuron.resetError();
    }
}*/
