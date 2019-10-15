package com.medi.MachineLearning.NeuralNetworks;

import com.medi.MachineLearning.Functions;
import com.medi.MachineLearning.Layers.*;
import com.medi.MachineLearning.Exceptions.ArraysSizeDifferents;

import java.io.Serializable;

public class LongShortTermMemory implements Serializable {
    int time = 0;
    double[][] lastPrediction;
    double[][] lastPreselection;
    double[][] lastFilteredPossibilities;
    double[][] lastSave;
    double[][] lastMemory;

    int[] representation;

    Layer[] network;
    Layer[] ignoring;
    Layer[] forgetting;
    Layer[] selection;

    /*public LongShortTermMemory(int[] representation){

        this.representation = representation;

        network = new Layer[representation.length];
        ignoring = new Layer[representation.length];
        forgetting = new Layer[representation.length];
        selection = new Layer[representation.length];
        lastPrediction = new double[representation.length-1][];
        lastSave = new double[representation.length-1][];
        lastPreselection = new double[representation.length-1][];
        lastFilteredPossibilities = new double[representation.length-1][];
        lastMemory = new double[representation.length-1][];

        network[0] = new InputLayer(representation[0]);
        ignoring[0] = new InputLayer(representation[0]);
        forgetting[0] = new InputLayer(representation[0]);
        selection[0] = new InputLayer(representation[0]);
        lastPrediction[0] = new double[representation[1]];
        lastSave[0] = new double[representation[1]];
        lastPreselection[0] = new double[representation[1]];
        lastFilteredPossibilities[0] = new double[representation[1]];
        lastMemory[0] = new double[representation[1]];

        for(int i = 1;i<representation.length-1;i++){
            network[i] = new HiddenLayer(network[i-1].neuronCount()+representation[i],representation[i]);
            ignoring[i] = new HiddenLayer(ignoring[i-1].neuronCount()+representation[i],representation[i]);
            forgetting[i] = new HiddenLayer(forgetting[i-1].neuronCount()+representation[i],representation[i]);
            selection[i] = new HiddenLayer(selection[i-1].neuronCount()+representation[i],representation[i]);
            lastPrediction[i] = new double[representation[i+1]];
            lastSave[i] = new double[representation[i+1]];
            lastPreselection[i] = new double[representation[i+1]];
            lastFilteredPossibilities[i] = new double[representation[i+1]];
            lastMemory[i] = new double[representation[i+1]];
        }

        network[representation.length-1] = new OutputLayer(network[representation.length-2].neuronCount()+representation[representation.length-1],representation[representation.length-1]);
        ignoring[representation.length-1] = new OutputLayer(ignoring[representation.length-2].neuronCount()+representation[representation.length-1],representation[representation.length-1]);
        forgetting[representation.length-1] = new OutputLayer(forgetting[representation.length-2].neuronCount()+representation[representation.length-1],representation[representation.length-1]);
        selection[representation.length-1] = new OutputLayer(selection[representation.length-2].neuronCount()+representation[representation.length-1],representation[representation.length-1]);

    }

    public void apprend(double[][][] dataSet, double learningRate) throws ArraysSizeDifferents {
        for(int i = 0;i<dataSet[0].length;i++){
            propagation(dataSet[0][i]);
            retropropagation(dataSet[1][i]);
            actualiseWeights(learningRate);
            resetErrors();
        }
        time ++;
    }

    public void propagation(double[] newInput) throws ArraysSizeDifferents {//check copy of arrays (.clone())

        double[] tempValues = newInput.clone();

        for(int i = 1;i<representation.length;i++) {
            double[] input = Functions.concatenate(lastPrediction[i], tempValues);
            InputLayer inputLayer = new InputLayer(input.length);
            inputLayer.setValues(input);
            ((RetrobakeLayer)network[i]).propagation(inputLayer);
            ((RetrobakeLayer)ignoring[i]).propagation(inputLayer);
            ((RetrobakeLayer)forgetting[i]).propagation(inputLayer);
            ((RetrobakeLayer)selection[i]).propagation(inputLayer);

            tempValues = Functions.product(network[i].getValues(), ignoring[i].getValues());
            lastFilteredPossibilities[i] = tempValues;
            double[] tempSave = Functions.product(forgetting[i].getValues(), lastSave[i]);
            lastMemory[i] = tempSave;
            tempValues = Functions.sum(tempSave, tempValues);
            lastSave[i] = tempValues.clone();
            tempValues = Functions.sigmoid(tempValues);
            lastPreselection[i] = tempValues;
            tempValues = Functions.product(selection[i].getValues(), tempValues);
            lastPrediction[i] = tempValues;
        }
    }

    public void retropropagation(double[] desiredOutput) throws ArraysSizeDifferents {

        RetrobakeLayer selectionOut = new OutputLayer(representation[representation.length-1]+representation[representation.length-2],representation[representation.length-1]);
        selectionOut.setValues(Functions.div(desiredOutput,lastPreselection[representation.length-2]));

        double[] selectionOutput = selection[representation.length-1].getValues();
        ((RetrobakeLayer)selection[representation.length-1]).retropropagation(selectionOut);
        double[] desiredCollectedPossibilities = Functions.reciprocalSigmoid(Functions.div(desiredOutput,selectionOutput));
        ((RetrobakeLayer)forgetting[representation.length-1]).retropropagation(Functions.div(Functions.sub(desiredCollectedPossibilities,lastFilteredPossibilities[representation.length-2]),lastSave[representation.length-2]));
        double[] desiredFilteredPossibilities = Functions.sub(desiredCollectedPossibilities,lastMemory[representation.length-2]);
        double[] ignoringOutput = ignoring[representation.length-1].getValues();
        ((RetrobakeLayer)ignoring[representation.length-1]).retropropagation(Functions.div(desiredFilteredPossibilities,network[representation.length-1].getValues()));
        ((RetrobakeLayer)network[representation.length-1]).retropropagation(Functions.div(desiredFilteredPossibilities,ignoringOutput));

        for(int i = representation.length-2;i>0;i--){
            double[] selectionOutput = selection[i].getValues();
            ((RetrobakeLayer)selection[i]).retropropagation(Functions.div(desiredOutput,lastPreselection[i]));
            double[] desiredCollectedPossibilities = Functions.reciprocalSigmoid(Functions.div(desiredOutput,selectionOutput));
            ((RetrobakeLayer)forgetting[i]).retropropagation(Functions.div(Functions.sub(desiredCollectedPossibilities,lastFilteredPossibilities[i]),lastSave[i]));
            double[] desiredFilteredPossibilities = Functions.sub(desiredCollectedPossibilities,lastMemory[i]);
            double[] ignoringOutput = ignoring[i].getValues();
            ((RetrobakeLayer)ignoring[i]).retropropagation(Functions.div(desiredFilteredPossibilities,network[i].getValues()));
            ((RetrobakeLayer)network[i]).retropropagation(Functions.div(desiredFilteredPossibilities,ignoringOutput));
            //Modifier desiredOutput
        }
    }

    public void actualiseWeights(double learningRate){
        for(int i = representation.length-1;i>0;i--){
            ((RetrobakeLayer)selection[i]).actualiseWeights(selection[i-1],learningRate);
            ((RetrobakeLayer)forgetting[i]).actualiseWeights(selection[i-1],learningRate);
            ((RetrobakeLayer)ignoring[i]).actualiseWeights(selection[i-1],learningRate);
            ((RetrobakeLayer)network[i]).actualiseWeights(selection[i-1],learningRate);
        }
    }

    public void resetErrors(){
        for(int i = representation.length-1;i>0;i--){
            ((RetrobakeLayer)selection[i]).resetErrors();
            ((RetrobakeLayer)forgetting[i]).resetErrors();
            ((RetrobakeLayer)ignoring[i]).resetErrors();
            ((RetrobakeLayer)network[i]).resetErrors();
        }
    }

    public double[] getOutput(){
        return lastPrediction[lastPrediction.length-1];
    }

    public void setLastPrediction(double[][] lastPrediction){
        this.lastPrediction = lastPrediction;
    }*/
}
