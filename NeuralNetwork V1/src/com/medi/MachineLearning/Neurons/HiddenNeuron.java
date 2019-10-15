package com.medi.MachineLearning.Neurons;

import com.medi.MachineLearning.Functions;
import com.medi.MachineLearning.Exceptions.ArraysSizeDifferents;

public class HiddenNeuron extends RetrobakeNeuron {

    @Override
    public  double agregation(double[] incident) throws ArraysSizeDifferents {
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

    /*@Override
    void retrobake(double[] nextWeights, double[] errors) throws ArraysSizeDifferents {
        error += derivativeActivation(notActivated)*Functions.scalarProduct(nextWeights,errors);
    }*/

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

    public void retrobake(RetrobakeNeuron[] nextNeurons) throws ArraysSizeDifferents {
        double[] errors = new double[nextNeurons.length];
        double[] nextWeights = new double[nextNeurons.length];
        for(int i = 0;i<nextNeurons.length;i++){
                errors[i] = nextNeurons[i].Error();
                nextWeights[i] = nextNeurons[i].IncidentWeight(neuronPosition);
        }
        error += derivativeActivation(notActivated)*Functions.scalarProduct(nextWeights,errors);
    }

    @Override
    public void actualiseWeights(double[] incidentNeurons, double delta) {//maybe error on error position or value position
        /*incidentWeights[incidentWeights.length-1] = incidentWeights[incidentWeights.length-1]*(time-1) +
                momentum*delta*error+
                (1-momentum)*delta*incidentWeights[incidentWeights.length-1]*(time-1);;//bias*/
        incidentWeights[incidentWeights.length-1] += delta*error;
        for(int i = 0;i<incidentWeights.length-1;i++){
            /*incidentWeights[i] = incidentWeights[i]*(time-1) +
                    momentum*delta*error*incidentNeurons[i] +
                    (1-momentum)*delta*incidentWeights[i]*(time-1);*/
            incidentWeights[i] += delta*error*incidentNeurons[i];
        }
    }

    @Override
    public void actualiseWeights(Neuron[] incidentNeurons, double delta) {//maybe error on error position or value position

        for(int i = 0;i<incidentWeights.length;i++){
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
