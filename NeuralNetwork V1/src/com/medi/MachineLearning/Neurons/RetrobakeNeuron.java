package com.medi.MachineLearning.Neurons;

import com.medi.MachineLearning.Functions;
import com.medi.MachineLearning.Exceptions.ArraysSizeDifferents;

public abstract class  RetrobakeNeuron extends Neuron {
    protected double[] incidentWeights;
    protected double error;

    public abstract double agregation(double[] incident) throws ArraysSizeDifferents;
    public abstract double agregation(Neuron[] incident) throws ArraysSizeDifferents;

    public abstract void bake(double[] incident) throws ArraysSizeDifferents;
    //abstract void retrobake(double[] nextWeights, double[] errors) throws ArraysSizeDifferents;
    public abstract double[] back();
    public abstract void bake(Neuron[] incident) throws ArraysSizeDifferents;
    public abstract void retrobake(RetrobakeNeuron[] nextNeurons) throws ArraysSizeDifferents ;
    public abstract void actualiseWeights(Neuron[] incidentNeurons, double delta);
    public abstract void actualiseWeights(double[] incidentNeurons, double delta);
    public void resetError(){
        error = 0;
    }

    public void setIncidentWeights(double[] incidentWeights){
        this.incidentWeights = incidentWeights;
    }
    public void setIncidentRandomWeights(int weightsCount){
        incidentWeights = new double[weightsCount];
        for(int i = 0;i<weightsCount;i++){
            incidentWeights[i] = Functions.random();
        }
    }

    public double Error(){
        return error;
    }
    public void setError(double error){
        this.error = error;
    }


    public double IncidentWeight(int index){
        return incidentWeights[index];
    }

    public double[] IncidentWeights(){
        return incidentWeights;
    }
}


