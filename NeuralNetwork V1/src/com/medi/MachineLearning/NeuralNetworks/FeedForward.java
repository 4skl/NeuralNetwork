package com.medi.MachineLearning.NeuralNetworks;

import Extern.RandomNoRepeat;
import com.medi.MachineLearning.Functions;
import com.medi.MachineLearning.Layers.*;
import com.medi.MachineLearning.Exceptions.ArraysSizeDifferents;

import java.io.Serializable;
import java.util.ArrayList;

public class FeedForward implements Serializable, Genetic {

    public static final long serialVersionUID = 4L;


    public Layer[] layers; //[layer] [neuron]
    public int[] representation;
    public int epoch = 0;
    public ArrayList<Double> scores;
    double[][][] testSet;
    double[][][] trainSet;
    //double[][][] weights;//[layer] [neuron] [incidentWeights]
    //bias defined in weights+1 (1)

    public FeedForward(int[] representation) {
        this.representation = representation;
        layers = new Layer[representation.length];
        layers[0] = new InputLayer(representation[0]);
        for(int i = 1;i<representation.length-1;i++){
            layers[i] = new HiddenLayer(layers[i-1].neuronCount(),representation[i]);
        }
        layers[representation.length-1] = new OutputLayer(layers[representation.length-2].neuronCount(),representation[representation.length-1]);
    }

    public void gradientDescent(double[][][] dataSet, int miniSetSize, double rate) throws ArraysSizeDifferents {
        RandomNoRepeat rnr = new RandomNoRepeat();
            for(int j = 0;j<dataSet[0].length/miniSetSize;j++){
                double[][][] miniSet = new double[2][miniSetSize][];
                for(int k = 0;k<miniSetSize;k++){
                    int rnd = rnr.next(dataSet[0].length);
                    miniSet[0][k] = dataSet[0][rnd];
                    miniSet[1][k] = dataSet[1][rnd];
                    propagation(miniSet[0][k]);
                    retropropagation(miniSet[1][k]);
                    actualiseWeights(rate/miniSetSize);
                }
                //for(int k = 0;k<miniSetSize;k++){

                //}
                resetErrors();
            }
            epoch++;
    }

    public void gradientDescent(double[][][] dataSet, double rate) throws ArraysSizeDifferents {
        for(int j = 0;j<dataSet[0].length;j++){
                propagation(dataSet[0][j]);
                retropropagation(dataSet[1][j]);
                actualiseWeights(rate);
                resetErrors();
        }
        epoch++;
    }

    public void gradientDescent(double[][][] dataSet, double[][][] testSet, int miniSetSize, double rate) throws ArraysSizeDifferents {

        RandomNoRepeat rnr = new RandomNoRepeat();
        for(int j = 0;j<dataSet[0].length/miniSetSize;j++){
            double[][][] miniSet = new double[2][miniSetSize][];
            for(int k = 0;k<miniSetSize;k++){
                int rnd = rnr.next(dataSet[0].length);
                miniSet[0][k] = dataSet[0][rnd];
                miniSet[1][k] = dataSet[1][rnd];
                propagation(miniSet[0][k]);
                retropropagation(miniSet[1][k]);
                actualiseWeights(rate/miniSetSize);
            }
            //for(int k = 0;k<miniSetSize;k++){

            //}
            resetErrors();
        }
        epoch++;
    }

    public void propagation(double[] inputValues) throws ArraysSizeDifferents {
        ((InputLayer) layers[0]).setValues(inputValues);
        for(int i = 1;i<representation.length;i++){
            ((RetrobakeLayer) layers[i]).propagation(layers[i-1]);
        }
    }

    /*void retropropagationAndActualise(OutputNeuron[] desiredOutput, double delta) throws ArraysSizeDifferents {
        for (int i = 0; i < neurons[neurons.length-1].length; i++) {
            neurons[neurons.length - 1][i].retrobake(desiredOutput);
            neurons[neurons.length - 1][i].actualiseWeights(neurons[neurons.length - 2], delta);
        }
        for (int i = neurons.length - 2; i > 1; i--) {
            for (int j = 0; j < neurons[i].length; j++) {
                neurons[i][j].retrobake(neurons[i + 1]);
                neurons[i][j].actualiseWeights(neurons[i - 1], delta);
            }
        }

        for (int j = 0; j < neurons[0].length; j++) {
            neurons[0][j].retrobake(neurons[1]);
            neurons[0][j].actualiseWeights(input, delta);
        }
        resetErrors();
    }*/

    public void retropropagation(Layer desiredOutput) throws ArraysSizeDifferents {
        ((OutputLayer) layers[layers.length-1]).retropropagation(desiredOutput);
        for(int i = layers.length-2;i>0;i--){
            ((HiddenLayer)layers[i]).retropropagation((RetrobakeLayer) layers[i+1]);
        }
    }

    public void retropropagation(double[] desiredOutput) throws ArraysSizeDifferents {
        if(layers[layers.length-1].neuronCount() != desiredOutput.length) throw new ArraysSizeDifferents();

        OutputLayer desiredOutputLayer = new OutputLayer(layers[layers.length-1].neuronCount(),desiredOutput.length);
        desiredOutputLayer.setValues(desiredOutput);
        retropropagation(desiredOutputLayer);
    }

    public void retropropagation(double cost) throws ArraysSizeDifferents {
        ((OutputLayer) layers[layers.length-1]).retropropagation(cost);
        for(int i = layers.length-2;i>0;i--){
            ((HiddenLayer)layers[i]).retropropagation((RetrobakeLayer) layers[i+1]);
        }
    }
    public void actualiseWeights(double learningRate){
        for(int i = 1;i<layers.length;i++){
            ((RetrobakeLayer) layers[i]).actualiseWeights(layers[i-1],learningRate);
        }
    }

    public void resetErrors(){
        for(int i = 1;i<layers.length;i++){
            ((RetrobakeLayer) layers[i]).resetErrors();
        }
    }

    public double test(double[][][] testData) throws ArraysSizeDifferents {
        double win = 0;
        for (int i = 0; i < testData[1].length; i++) {
            propagation(testData[0][i]);
            int think = Functions.indexOf(getOutput(), Functions.maxInArray(getOutput()));
            int real = Functions.indexOf(testData[1][i], 1);
            if (think == real) {
                win++;
            }
        }
        double score = win / testData[1].length;
        return score;
    }

    public void addScore(double score){
        if(scores == null) {
            scores = new ArrayList<Double>();
            scores.add(score);
        }else if(epoch>=scores.size()) {
            scores.add(score);
        }
    }

    public double test(){
        double win = 0;
        for (int i = 0; i < testSet[1].length; i++) {
            try {
                propagation(testSet[0][i]);
            }catch (ArraysSizeDifferents e){//do something
                }
            int think = Functions.indexOf(getOutput(), Functions.maxInArray(getOutput()));
            int real = Functions.indexOf(testSet[1][i], 1);
            if (think == real) {
                win++;
            }
        }
        double score = win / testSet[1].length;
        if(scores == null) {
            scores = new ArrayList<Double>();
            scores.add(score);
        }else if(epoch>=scores.size()) {
            scores.add(score);
        }
        return score;
    }

    public double absoluteError(double[][][] testData) throws ArraysSizeDifferents {
        double error = 0;
        for(int i = 0;i<testData[0].length;i++) {
            propagation(testData[0][i]);
            error += Functions.summedAbsoluteCost(getOutput(), testData[1][i]);
        }
        error/=testData[0].length;
        return error;
    }

    public void train(double learningRate) throws ArraysSizeDifferents {
        gradientDescent(trainSet,learningRate);
    }


    public void setTestSet(double[][][] testSet){
        this.testSet = testSet;
    }
    public void setTrainSet(double[][][] trainSet){
        this.trainSet = trainSet;
    }


    public double[] getOutput(){
        return ((OutputLayer) layers[layers.length-1]).getValues();
    }


    @Override
    public double getScore() {
        return test();
    }

    @Override
    public FeedForward[] getChilds(int childCount, double dispersion) {
        FeedForward[] networks = new FeedForward[childCount];
        for(int i = 0;i<childCount;i++){
                networks[i] = new FeedForward(representation);
                networks[i].setTestSet(testSet);
                networks[i].setTrainSet(trainSet);
            /*for(int l = 0;l<trainSet[0].length;l++) {
                try {
                    networks[i].propagation(trainSet[0][l]);
                } catch (ArraysSizeDifferents arraysSizeDifferents) {
                    arraysSizeDifferents.printStackTrace();
                }*/
                for (int j = 1; j < networks[i].layers.length; j++) {
                    double[][] newWeights = ((RetrobakeLayer)this.layers[j]).getIncidentWeights();
                    for(int k = 0;k<newWeights.length;k++){
                        for(int l = 0;l<newWeights[k].length;l++){
                            newWeights[k][l] = newWeights[k][l]+(Math.random()*2-1)*dispersion;
                        }
                    }
                    try {
                        ((RetrobakeLayer)networks[i].layers[j]).setIncidentWeights(newWeights);
                    } catch (ArraysSizeDifferents arraysSizeDifferents) {
                        //Do something
                    }
                }
           // }
        }
        return networks;
    }

    /*public double[] back(double[] output){//Bad
        for(int i = neurons.length-1; i>0;i--){
            double[][] incidentWeight = new double[neurons[i].length][neurons[i-1].length];
            double[] nextNeurons = new double[neurons[i].length];
                for(int j = 0;j<neurons[i].length;j++) {
                    incidentWeight[i] = neurons[i][j].IncidentWeights();
                    nextNeurons[i] = neurons[i][j].NotActivated();
                }
                Matrice b = new Matrice(incidentWeight);
                Matrice c = new Matrice(new double[][]{nextNeurons});
                Matrice a = c.multiply(b.getMatriceTranspose());
                double[] incidentNeurons = a.getMatrice()[0];
                for(int j = 0;j<neurons[i-1].length-1;j++) {
                    neurons[i-1][j].setValue(incidentNeurons[j]);
                }
                //calculer l'inverse d'incidentWeights, on cherche incidentNeurons
                //faire l'inverse de incidentWeights * nextNeurons = incidentNeurons
                //appliquer la valeur d'incidentNeurons a neurons[i-1]
        }
        return input;
    }*/
}
