package com.medi.MachineLearning.Neurons;

import com.medi.MachineLearning.Functions;
import com.medi.MachineLearning.Exceptions.ArraysSizeDifferents;

public class OutputNeuron extends RetrobakeNeuron {

    @Override
    public double agregation(double[] incident) throws ArraysSizeDifferents {
        notActivated = Functions.scalarProduct(incidentWeights,incident);
        return notActivated;
    }

    @Override
    public double agregation(Neuron[] incident) throws ArraysSizeDifferents {
        double[] incidentValue = new double[incident.length];
        for(int i = 0;i<incident.length;i++){
            incidentValue[i] = incident[i].Value();
        }
        notActivated = Functions.scalarProduct(incidentWeights,incidentValue);
        return notActivated;
    }

    @Override
    public void bake(double[] incident) throws ArraysSizeDifferents {
        value = activation(agregation(incident));
    }

    @Override
    public double[] back() {
        double[] incidentValues = new double[incidentWeights.length];
        for(int i = 0;i<incidentWeights.length;i++){
            incidentValues[i] = reciprocalActivation(value)/incidentWeights[i];
        }
        return incidentValues;
    }

    @Override
    public void bake(Neuron[] incident) throws ArraysSizeDifferents {
        value = activation(agregation(incident));
    }

    @Override
    public void retrobake(RetrobakeNeuron[] nextNeurons) throws ArraysSizeDifferents {
        error += derivativeActivation(notActivated)*(nextNeurons[neuronPosition].Value()-value);
    }

    public void retrobake(Neuron[] nextNeurons) throws ArraysSizeDifferents {
        error += derivativeActivation(notActivated)*(nextNeurons[neuronPosition].Value()-value);
    }

    public void retrobake(double need){
        error += derivativeActivation(notActivated)*(need-value);
    }

    @Override
    public void actualiseWeights(double[] incidentNeurons, double delta) {

        for(int i = 0;i<incidentWeights.length-1;i++){
            incidentWeights[i] += delta*error*incidentNeurons[i];
        }
    }

    @Override
    public void actualiseWeights(Neuron[] incidentNeurons, double delta) {
        for(int i = 0;i<incidentWeights.length-1;i++){
            incidentWeights[i] += delta*error*incidentNeurons[i].Value();
        }
    }

    @Override
    public double activation(double x) {
        return Functions.sigmoid(x);
    }

    @Override
    public double derivativeActivation(double x) {
        return Functions.derivativeSigmoid(x);
    }

    @Override
    public double reciprocalActivation(double x) {
        return Functions.reciprocalSigmoid(x);
    }
}
