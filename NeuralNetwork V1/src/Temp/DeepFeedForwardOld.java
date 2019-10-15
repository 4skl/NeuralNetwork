package Temp;
import Extern.RandomNoRepeat;
import com.medi.MachineLearning.Functions;
import com.medi.MachineLearning.Exceptions.ArraysSizeDifferents;

public class DeepFeedForwardOld {
    int[] representation;
    double[][] biases;
    double[][][] weights;
    public double[][] values;
    double[][] noNormalizedValues;

    public DeepFeedForwardOld(int[] representation){
        this.representation = representation;
        biases = new double[representation.length-1][];
        weights = new double[representation.length-1][][];
        values = new double[representation.length][];
        noNormalizedValues = new double[representation.length-1][];

        values[0] = new double[representation[0]];
        for(int i = 0;i<representation.length-1;i++){
            biases[i] = new double[representation[i+1]];
            weights[i] = new double[representation[i]][representation[i+1]];
            values[i+1] = new double[representation[i+1]];
            noNormalizedValues[i] = new double[representation[i+1]];
        }
        randomizeWeightsAndBiases();
    }

    public void randomizeWeightsAndBiases(){
        for(int x = 0;x<weights.length;x++){
            for(int y = 0;y<values[x+1].length;y++){
                for(int z = 0;z<values[x].length;z++){
                    weights[x][z][y] = Functions.random();
                }
                biases[x][y] = Functions.random();
            }
        }
    }

    public void gradientDescent(double[][][] trainingData, int epochs, int miniSetSize, double rate) throws ArraysSizeDifferents {
        for(int i = 0;i<epochs;i++){
            double[][][][] miniSets = new double[trainingData[0].length/miniSetSize][miniSetSize][2][];//verif 2
            RandomNoRepeat rnr = new RandomNoRepeat();
            for(int j = 0;j<trainingData[0].length/miniSetSize;j++) {
                for(int k = 0;k<miniSetSize;k++) {
                    int index = rnr.next(trainingData[0].length);
                    miniSets[j][k][0] = trainingData[0][index];
                    miniSets[j][k][1] = trainingData[1][index];
                }
                update(miniSets[j],rate);
            }

        }
    }

    public double[] feedforward(double[] input){
        values[0] = input;//.clone() ?
        for(int x = 0;x<weights.length;x++){
            for(int y = 0;y<values[x+1].length;y++){
                for(int z = 0;z<values[x].length;z++){
                        values[x+1][y] += values[x][z] * weights[x][z][y];
                }
                values[x+1][y] += biases[x][y];
                noNormalizedValues[x][y] = values[x+1][y];
                values[x+1][y] = activation(values[x+1][y]);
            }
        }

        return values[representation.length-1];//.clone() ?
    }

    public void update(double[][][] miniSet,double rate) throws ArraysSizeDifferents {
        for(double[][] set : miniSet){
            double[][][] errors = backpropagation(set[0],set[1]);
            double[][] errorsBiases = errors[0];
            double[][] errorsWeights = errors[1];
            //some error here
            for(int i = 0;i<weights.length;i++){
                for (int j = 0;j<weights[i].length;j++){
                    for(int k = 0;k<weights[i][j].length;k++) {
                        weights[i][j][k] += (rate/miniSet.length)*errorsWeights[i][k]*values[i][k];
                        if(j == 0)
                        biases[i][k] += (rate/miniSet.length) * errorsBiases[i][k];
                    }
                }
            }

        }
    }

    public double[][][] backpropagation(double[] input, double[] output) throws ArraysSizeDifferents {
        double[][] errorsWeights = new double[values.length-1][];
        double[][] errorsBiases = new double[values.length-1][];
        //double[][] errorsWeights = new double[weights.length][];//some doubts here
        for(int i = 0;i<errorsBiases.length;i++){
            errorsBiases[i] = new double[values[i+1].length];
            errorsWeights[i] = new double[values[i+1].length];
            //errorsWeights[i] = new double[weights[i].length];
            /*for(int j = 0;j<errorsWeights[i].length;j++){
                errorsWeights[i][j] = new double[weights[i][j].length];
            }*/
        }

        feedforward(input);

        //calculus of the first errors
        for(int i = 0;i<errorsBiases[errorsBiases.length-1].length;i++){
            errorsBiases[errorsBiases.length-1][i] = costDerivative(values[values.length-1][i],output[i])*
            derivativeActivation(noNormalizedValues[noNormalizedValues.length-1][i]);
        }
        //errorsBiases[errorsBiases.length-1] = delta1;//.clone() ?
        for(int i = 0;i<errorsWeights[errorsWeights.length-1].length;i++){
            errorsWeights[errorsWeights.length-1][i] = errorsBiases[errorsBiases.length-1][i]*values[values.length-1][i];//can contain an error
        }

        //calculus of the next errors
        /*for(int i = 2;i<values.length;i++){
            double[] backDerivate = new double[errorsBiases[errorsBiases.length-i].length];
            double[] delta2 = new double[errorsBiases[errorsBiases.length-i].length];
            for(int j = 0;j<errorsBiases[errorsBiases.length-i].length;j++) {
                backDerivate[j] = derivativeActivation(noNormalizedValues[i][j]);
                delta2[j] = Functions.scalarProduct(weights[errorsBiases[errorsBiases.length-1-i].length][j],delta1)*backDerivate[j];//can contain an error
            }
            errorsBiases[errorsBiases.length-i] = delta2;//.clone() ?
            for(int j = 0;j<errorsBiases[errorsBiases.length-i].length;j++) {
                errorsWeights[errorsWeights.length-1-i][j] = Functions.scalarProduct(delta2,values[values.length-1-i]);
            }
        }*/
        for(int i = errorsBiases.length-2;i>=0;i--){
            for(int j = 0;j<errorsBiases[i].length;j++) {
                double nextError = 0;
                nextError+=Functions.scalarProduct(errorsBiases[i],weights[i][j]);//contain an error
                errorsBiases[i][j] = derivativeActivation(noNormalizedValues[i][j]) * nextError;
                errorsWeights[i][j] = errorsBiases[i][j] * values[i][j];
            }
        }

        return new double[][][]{errorsBiases,errorsWeights};//can contain an error

    }

    /*public double evaluate(double[][][] testData){
        double result = 0;

        for(int i = 0;i<testData.length;i++){
            feedforward(testData[i][0]);
            if(Functions.maxInArray(values[values.length-1]) == testData[])
        }

        return result;
    }*/

    double cost(double input, double output){
        return Functions.quadraticCost(input,output);
    }
    double costDerivative(double input, double output){
        return Functions.derivativeQuadraticCost(input,output);
    }
    double activation(double x){
        return Functions.sigmoid(x);
    }
    double derivativeActivation(double x){
        return Functions.derivativeSigmoid(x);
    }

}
