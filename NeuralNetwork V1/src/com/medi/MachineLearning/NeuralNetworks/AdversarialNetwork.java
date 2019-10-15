package com.medi.MachineLearning.NeuralNetworks;

import com.medi.MachineLearning.Exceptions.ArraysSizeDifferents;
import com.medi.MachineLearning.Functions;
import com.medi.MachineLearning.Layers.*;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.function.Function;

public class AdversarialNetwork {

    private Layer[] generator,discriminator;
    int[] generatorRepresentation,discriminatorRepresentation;
    private int epoch;
    public int oldWin = 0;
    public int discriminatorWin;

    public AdversarialNetwork(int[] generatorRepresentation, int[] discriminatorRepresentation){
        this.generatorRepresentation = generatorRepresentation;
        this.discriminatorRepresentation = discriminatorRepresentation;

        generator = new Layer[generatorRepresentation.length];
        generator[0] = new InputLayer(generatorRepresentation[0]);
        for(int i = 1;i<generatorRepresentation.length;i++){
            generator[i] = new HiddenLayer(generator[i-1].neuronCount(),generatorRepresentation[i]);
        }

        discriminator = new Layer[discriminatorRepresentation.length];
        discriminator[0] = new InputLayer(generatorRepresentation[generatorRepresentation.length-1]);
        for(int i = 1;i<discriminatorRepresentation.length-1;i++){
            discriminator[i] = new HiddenLayer(discriminator[i-1].neuronCount(),discriminatorRepresentation[i]);
        }
        discriminator[discriminator.length-1] = new OutputLayer(discriminator[discriminator.length-2].neuronCount(),1);


        epoch = 0;
    }

    double propagateDiscriminator(double[] input) throws ArraysSizeDifferents {
        discriminator[0].setValues(input);
        for(int i = 1;i<discriminator.length;i++){
            ((RetrobakeLayer)discriminator[i]).propagation(discriminator[i-1]);
        }
        return discriminator[discriminator.length-1].getValues()[0];
    }

    void retropropagateDiscriminator(double[] need) throws ArraysSizeDifferents {
        ((OutputLayer) discriminator[discriminator.length-1]).retropropagation(need);
        for(int i = discriminator.length-2;i>0;i--){
            ((RetrobakeLayer)discriminator[i]).retropropagation((RetrobakeLayer) discriminator[i+1]);
        }
    }
    void actualiseDiscriminatorWeightsAndResetError(double learningRate){
        for(int i = 1;i<discriminator.length;i++){
            ((RetrobakeLayer)discriminator[i]).actualiseWeights(discriminator[i-1],learningRate);
        }
        for(int i = 1;i<discriminator.length;i++){
            ((RetrobakeLayer)discriminator[i]).resetErrors();
        }
    }

    void resetDiscriminatorError(){
        for(int i = 1;i<discriminator.length;i++){
            ((RetrobakeLayer)discriminator[i]).resetErrors();
        }
    }

    void propagateGenerator(double[] input) throws ArraysSizeDifferents {
        generator[0].setValues(input);
        for(int i = 1;i<generator.length;i++){
            ((RetrobakeLayer)generator[i]).propagation(generator[i-1]);
        }
    }

    void retropropagateAll(double[] need) throws ArraysSizeDifferents {
        retropropagateDiscriminator(need);
        ((RetrobakeLayer)generator[generator.length-1]).retropropagation(((RetrobakeLayer) discriminator[1]));//?
        for(int i = generator.length-2;i>0;i--){
            ((RetrobakeLayer)generator[i]).retropropagation((RetrobakeLayer) generator[i+1]);
        }
    }

    void actualiseGeneratorWeightsAndResetError(double learningRate){
        for(int i = 1;i<generator.length;i++){
            ((RetrobakeLayer)generator[i]).actualiseWeights(generator[generator.length-1],learningRate);
        }
        for(int i = 1;i<generator.length;i++){
            ((RetrobakeLayer)generator[i]).resetErrors();
        }
    }

    public void train(double[][] data, double generatorLearningRate, double discriminatorLearningRate) throws ArraysSizeDifferents {//1 = true; 0 = false
        discriminatorWin = 0;
        for(int i = 0;i<data.length;i++){

            double discriminatorOutput;
                discriminatorOutput = propagateDiscriminator(data[i]);
                retropropagateDiscriminator(new double[]{1});
                actualiseDiscriminatorWeightsAndResetError(discriminatorLearningRate);
                discriminatorWin += (discriminatorOutput > 0.5) ? 1 : 0;

                propagateGenerator(Functions.getRandomArray(generatorRepresentation[0]));

                discriminatorOutput = propagateDiscriminator(generator[generator.length - 1].getValues());
                discriminatorWin += (discriminatorOutput < 0.5) ? 1 : 0;
                retropropagateDiscriminator(new double[]{0});
                actualiseDiscriminatorWeightsAndResetError(discriminatorLearningRate);

            //((RetrobakeLayer)generator.layers[generator.layers.length-1]).setErrors(error);
            //discriminator.retropropagation(new double[]{1});
            propagateGenerator(Functions.getRandomArray(generatorRepresentation[0]));
            //BufferedImage img = Functions.arrayToImg(generator[generator.length-1].getValues(),28);
            propagateDiscriminator(generator[generator.length-1].getValues());
            retropropagateAll(new double[]{1});
            actualiseGeneratorWeightsAndResetError(generatorLearningRate);
            resetDiscriminatorError();

            /*try {
                ImageIO.write(Functions.arrayToImg(generator[generator.length-1].getValues(),28),"png",new File("GenerationTest\\"+
                        "Gen " + epoch + " image number " + i + ".png"));
            } catch (IOException e) {
                e.printStackTrace();
            }*/

        }
        epoch++;
        oldWin = discriminatorWin;
    }

    public void train(double[][] inData, double[][] outData, double generatorLearningRate, double discriminatorLearningRate) throws ArraysSizeDifferents {//1 = true; 0 = false
        discriminatorWin = 0;
        for(int i = 0;i<inData.length;i++){

            double discriminatorOutput;

                propagateDiscriminator(inData[i]);
                retropropagateDiscriminator(outData[i]);
                actualiseDiscriminatorWeightsAndResetError(discriminatorLearningRate);

                propagateGenerator(Functions.concatenate(outData[i],Functions.getRandomArray(generatorRepresentation[0]-outData[i].length)));
                propagateDiscriminator(generator[generator.length - 1].getValues());
                retropropagateDiscriminator(new double[outData[i].length]);
                actualiseDiscriminatorWeightsAndResetError(discriminatorLearningRate);

            //((RetrobakeLayer)generator.layers[generator.layers.length-1]).setErrors(error);
            //discriminator.retropropagation(new double[]{1});
            propagateGenerator(Functions.concatenate(outData[i],Functions.getRandomArray(generatorRepresentation[0]-outData[i].length)));
            //BufferedImage img = Functions.arrayToImg(generator[generator.length-1].getValues(),28);
            propagateDiscriminator(generator[generator.length-1].getValues());
            retropropagateAll(outData[i]);
            actualiseGeneratorWeightsAndResetError(generatorLearningRate);
            resetDiscriminatorError();
        }
        epoch++;
        oldWin = discriminatorWin;
    }

    public double[] getGenerated() throws ArraysSizeDifferents {
        propagateGenerator(Functions.getRandomArray(generatorRepresentation[0]));
        return generator[generator.length-1].getValues();
    }
    public double[] getGenerated(int i, int maxI) throws ArraysSizeDifferents {
        double[] ari = new double[maxI];
        ari[i] = 1;
        propagateGenerator(Functions.concatenate(ari,Functions.getRandomArray(generatorRepresentation[0]-maxI)));
        return generator[generator.length-1].getValues();
    }

    public boolean isTrue(double[] data) throws ArraysSizeDifferents {

        return propagateDiscriminator(data)<0.5;
    }

}
