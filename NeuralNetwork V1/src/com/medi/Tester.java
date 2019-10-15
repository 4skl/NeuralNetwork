package com.medi;

import Extern.Exceptions.BadIdxType;
import Extern.Exceptions.ItemsOutOfIndex;
import Extern.IdxFormat;
import Extern.MNISTFeature;
import Extern.Matrice;
import com.medi.MachineLearning.Functions;
import com.medi.MachineLearning.Layers.RetrobakeLayer;
import com.medi.MachineLearning.NeuralNetworks.*;
import com.medi.MachineLearning.Exceptions.ArraysSizeDifferents;
import sun.text.resources.cldr.yav.FormatData_yav;

import javax.imageio.ImageIO;
import javax.sound.midi.SysexMessage;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.*;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Random;
import java.util.Scanner;

public class Tester {

    public static double[] textToArray(String text, String charset){
        double[] value = new double[text.length()];
        for(int i = 0;i<text.length();i++){
            value[i] = (double)charset.indexOf(text.charAt(i))/charset.length();
        }
        return value;
    }

    public static String arrayToText(double[] array, String charset){
        String text = "";
        for(int i = 0;i<array.length;i++){
            text+=charset.charAt((int) (array[i]*((double)charset.length())));
        }
        return text;
    }

    public static void passTest() throws ArraysSizeDifferents {


        String charset = "AZERTYUIOPQSDFGHJKLMWXCVBN0123456789";
        //Test set
        String[] idsT = {"LEKESIS"};
        String[] passT = { "SBS2SN"};
        double[][] inT = new double[idsT.length][];
        double[][] outT = new double[passT.length][];
        for(int i = 0;i<idsT.length;i++){
            inT[i] = textToArray(idsT[i],charset);
        }
        for(int i = 0;i<passT.length;i++){
            outT[i] = textToArray(passT[i],charset);
        }

        //Training set
        String[] ids = {"BRENIET","OLIVIEM","GANITTM","ABADIEM"};
        String[] pass = {"H4DCHT", "K4BN3H", "3RBFBM", "11CM3Q"};
        double[][] in = new double[ids.length][];
        double[][] out = new double[pass.length][];
        for(int i = 0;i<ids.length;i++){
            in[i] = textToArray(ids[i],charset);
        }
        for(int i = 0;i<pass.length;i++){
            out[i] = textToArray(pass[i],charset);
        }
        long seed = new Random().nextLong();
        seed = 6739994676423694242L;
        Functions.random.setSeed(seed);
        int[] representation = {7,16,16,6};
        double learningRate = 10;
        int epochs = 100000;
        int minI = 0;
        double min = 100;
        FeedForward ff = new FeedForward(representation);
        for(int i = 0;i<epochs;i++) {
            ff.gradientDescent(new double[][][]{in, out}, learningRate);
            double trainError = ff.absoluteError(new double[][][]{in, out});
            double realError = ff.absoluteError(new double[][][]{inT,outT});
            if(realError<min){
                min = realError;
                minI = i;
            }
            System.out.println("@Epoch " + i + " Train error = " + trainError + " Test error = " + realError);
        }
        String predictedID = "ABADIEM";
        ff.propagation(textToArray(predictedID,charset));
        System.out.println(Arrays.toString(ff.getOutput())+"\n"+Arrays.toString(textToArray(predictedID,charset)));
        String predictedPass = arrayToText(ff.getOutput(),charset);

        System.out.println(predictedID + " : " + predictedPass);

        System.out.println("Min is : " + min + " at " + minI + " with seed " + seed);



    }

    public static void miniTraining() throws ArraysSizeDifferents {
        double[] input = {0.1,-0.5,0.9};
        double[] output = {0.9,0.1};
        double rate = 1;
        int epochs = 1000;
        int[] representation = {3,2};
        FeedForward neuralNetwork = new FeedForward(representation);
        for(int i = 0;i<epochs;i++){
            neuralNetwork.propagation(input);
            neuralNetwork.retropropagation(output);
            neuralNetwork.actualiseWeights(rate);
            neuralNetwork.resetErrors();
            System.out.println(i + " epoch : " + Arrays.toString(neuralNetwork.getOutput()));
        }
    }

    public static void miniTester() throws ArraysSizeDifferents {
        double[][] incidentWeight = {{1,2,3},{4,5,6}};
        double[] incidentNeurons = {1,2,3};
        double[][] nextNeurons = {{Functions.scalarProduct(incidentWeight[0],incidentNeurons)},
        {Functions.scalarProduct(incidentWeight[1],incidentNeurons)}};
        Matrice b = new Matrice(incidentWeight);
        Matrice c = new Matrice(nextNeurons);
        System.out.println(Arrays.toString(b.multiply(new Matrice(new double[][]{incidentNeurons})).getMatrice()) + "\n" +
                            Arrays.toString(nextNeurons));
        System.out.println((b.multiply(b.getMatriceTranspose())).getDeterminant());
        Matrice a = c.multiply(b.getMatriceTranspose()).multiply((b.multiply(b.getMatriceTranspose())).getMatriceInverse());
        System.out.println(Arrays.toString(a.getMatrice()[0]) +"*"+ Arrays.toString(incidentWeight) +"="+Arrays.toString(nextNeurons)+"\n"+
                Arrays.toString(incidentNeurons) +"*"+ Arrays.toString(incidentWeight) +"="+Arrays.toString(nextNeurons));
    }


    static String trainLabelPath = "D:\\Programmation\\workspace\\IntelliJ\\NeuralNetwork\\src\\train-labels.idx1-ubyte";
    static String trainImgPath = "D:\\Programmation\\workspace\\IntelliJ\\NeuralNetwork\\src\\train-images.idx3-ubyte";

    static String testLabelPath = "D:\\Programmation\\workspace\\IntelliJ\\NeuralNetwork\\src\\t10k-labels.idx1-ubyte";
    static String testImgPath = "D:\\Programmation\\workspace\\IntelliJ\\NeuralNetwork\\src\\t10k-images.idx3-ubyte";
    static String networkSaveName = "testNetwork.nn";

    public static void testTrainingImg() throws BadIdxType, ItemsOutOfIndex {
        long t1 = System.nanoTime();
        int maxSzX = 1;
        int maxSzY = 1;
        int sZX = (int) Math.ceil(28.0/maxSzX);
        int sZY = (int) Math.ceil(28.0/maxSzY);
        int[] representation = {(sZX)*(sZY),100,10};
        System.out.println("Complexity is : " + Functions.complexity(representation) + " for configuration : " + Arrays.toString(representation));
        double rate = 0.3;
        int trainingSetSize = 10000;
        int testSetSize = 1000;
        int epochs = 50;
        int minisetSize = 1;
        FeedForward network = null;
        try {
            ObjectInputStream objIn = new ObjectInputStream(new FileInputStream(networkSaveName));
            network = (FeedForward) objIn.readObject();
            objIn.close();
        } catch (IOException e) {
            System.out.println("No File Creating new network");
            network = new FeedForward(representation);
        } catch (ClassNotFoundException e) {
            System.out.println("Bad Class Creating new network");
            network = new FeedForward(representation);
        }
        //MNISTFeature[] mnistFeatures = new MNISTFeature[trainingSetSize];
        double[][] mnistFeaturesInput = new double[trainingSetSize][(sZX)*(sZY)];
        double[][] mnistFeaturesOutput = new double[trainingSetSize][10];
        System.out.println("Read Images");
        long tr1 = System.nanoTime();
        MNISTFeature[] mnistFeature = new MNISTFeature[trainingSetSize];
        try {
            mnistFeature = IdxFormat.getIdx(trainLabelPath,trainImgPath,0,trainingSetSize);
        } catch (IOException e) {
            e.printStackTrace();
        }
        for(int i = 0;i<trainingSetSize;i++){

            /*double[][] featureTemp = mnistFeature[i].getArrayInput();
                                double[][][] inputTemp = ConvolutionalNN.getFormOfThis(featureTemp,4,4);
                                for(int t = 0;t<inputTemp.length;t++){
                                    inputTemp[t] = ConvolutionalNN.maxPooling(ConvolutionalNN.convolutionAnReLU(featureTemp,inputTemp[t]),maxSzX,maxSzY);
                                }
            featureTemp = ConvolutionalNN.maxPooling(featureTemp,maxSzX,maxSzY);
            */
            mnistFeaturesInput[i] = mnistFeature[i].getInput();
            mnistFeaturesOutput[i] = mnistFeature[i].getOutput();

        }

        double[][] testFeaturesInput = new double[testSetSize][(sZX)*(sZY)];
        double[][] testFeaturesOutput = new double[testSetSize][10];
        MNISTFeature[] mnistFeature1 = new MNISTFeature[testSetSize];
        try {
            mnistFeature1 = IdxFormat.getIdx(testLabelPath,testImgPath,0,testSetSize);
        } catch (IOException e) {
            e.printStackTrace();
        }
        for(int i = 0;i<testSetSize;i++){
            /*double[][] featureTemp = mnistFeature1[i].getArrayInput();
                                double[][][] inputTemp = ConvolutionalNN.getFormOfThis(featureTemp,4,4);
                                for(int t = 0;t<inputTemp.length;t++){
                                    inputTemp[t] = ConvolutionalNN.maxPooling(ConvolutionalNN.convolutionAnReLU(featureTemp,inputTemp[t]),maxSzX,maxSzY);
                                }
            featureTemp = ConvolutionalNN.maxPooling(featureTemp,maxSzX,maxSzY);
            */
            testFeaturesInput[i] = mnistFeature1[i].getInput();
            testFeaturesOutput[i] = mnistFeature1[i].getOutput();
        }
        System.out.println("Reading Time : " + (System.nanoTime()-tr1)/1e6 + "ms");
        BufferedImage scoreCurve = new BufferedImage(1000,200,BufferedImage.TYPE_INT_RGB);
        Graphics2D graphics = scoreCurve.createGraphics();
        graphics.setPaint (new Color(255,255,255));
        graphics.fillRect (0,0, scoreCurve.getWidth(), scoreCurve.getHeight());
        graphics.setPaint (new Color(0,0,0));

        System.out.println("Train on " + trainingSetSize + "features");
        long tt1 = System.nanoTime();
        double[] score = new double[epochs+1];
        score[0] = 0;
        for(int i = 0;i<epochs;i++) {
            //for(int j = 0;j<trainingSetSize;j++) {
                try {
                    network.gradientDescent(new double[][][]{testFeaturesInput, testFeaturesOutput}, rate);
                        //network.propagation(mnistFeaturesInput[j]);
                    //if(!MNISTFeature.isGood(network.getOutput(),mnistFeaturesOutput[j])) {
                        /*network.retropropagation(mnistFeaturesOutput[j]);
                        network.actualiseWeights(rate);
                        network.resetErrors();*/
                    //}
                } catch (ArraysSizeDifferents arraysSizeDifferents) {
                    arraysSizeDifferents.printStackTrace();
                }
            //}
            try {
                ObjectOutputStream objOut = new ObjectOutputStream(new FileOutputStream(networkSaveName));
                objOut.writeObject(network);
                objOut.flush();
                objOut.close();
            } catch (IOException e) {
                e.printStackTrace();
            }
            try {
                score[i + 1] = ((MNISTFeature.test(network, new double[][][]{testFeaturesInput, testFeaturesOutput})));
                /*graphics.drawLine((int) (i * ((double) scoreCurve.getWidth() / epochs)),
                        (int) (score[i] * ((double) scoreCurve.getHeight())),
                        (int) ((i + 1) * ((double) scoreCurve.getWidth() / epochs)),
                        (int) (score[i + 1] * ((double) scoreCurve.getHeight())));*/
                System.out.println("Score : " + score[i + 1] * 100 + "% at epoch " + network.epoch + " learning rate is : " + rate);

            } catch (ArraysSizeDifferents arraysSizeDifferents) {
                arraysSizeDifferents.printStackTrace();
            }/*
            if((i+1)%10 == 0){
                //rate /= 2;
            }*/
            //rate /= 1.08-((i+1)/(epochs*trainingSetSize));
        }
        System.out.println("Training Time : " + (System.nanoTime()-tt1)/1e6 + "ms");

        try {
            ObjectOutputStream objOut = new ObjectOutputStream(new FileOutputStream(networkSaveName));
            objOut.writeObject(network);
            objOut.flush();
            objOut.close();
        } catch (IOException e) {
            e.printStackTrace();
        }

        long teT = System.nanoTime();
        try {
            System.out.println("Score : " + ((MNISTFeature.test(network,new double[][][]{testFeaturesInput,testFeaturesOutput})))*100 + "%");
        } catch (ArraysSizeDifferents arraysSizeDifferents) {
            arraysSizeDifferents.printStackTrace();
        }
        teT = System.nanoTime()-teT;
        try {
            ImageIO.write(scoreCurve,"png",new File("D:\\Programmation\\workspace\\IntelliJ\\NeuralNetwork\\Results\\" +
                    "pna sigmoid network " + Arrays.toString(representation) +
                    " rate = " + rate + " epoch = " + epochs +
                    " set = " + trainingSetSize + ".png"));
        } catch (IOException e) {
            e.printStackTrace();
        }
        /*for(int i = 0;i<10;i++){
            double[] outVal = new double[10];
            outVal[i] = 0.9;
            BufferedImage thingImg = MNISTFeature.inverse(network,outVal);
            try {
                ImageIO.write(thingImg,"png",new File("Think " + i + ".png"));
            } catch (IOException e) {
                e.printStackTrace();
            }
        }*/
        System.out.println("Test time : " + teT/1e6 + "ms");
        System.out.println("Total Time : " + (System.nanoTime()-t1)/1e6 + "ms");
    }

    public static void generateImg() throws ItemsOutOfIndex, IOException, BadIdxType, ArraysSizeDifferents {
        int trainingSetSize = 6000;
        int epoch = 3;

        int[] representation = {10,100,28*28};
        double learningRate = 0.3;
        FeedForward network = new FeedForward(representation);

        System.out.println("Read Dataset");
        long time = System.nanoTime();
        MNISTFeature[] features = IdxFormat.getIdx(trainLabelPath,trainImgPath,0,trainingSetSize);
        double[][] featureInput = new double[trainingSetSize][representation[0]];
        double[][] featureOutput = new double[trainingSetSize][representation[representation.length-1]];
        for(int i = 0;i<trainingSetSize;i++){
            featureInput[i] = features[i].getOutput();
            featureOutput[i] = features[i].getInput();
        }
        time = System.nanoTime()-time;
        System.out.println("Reading time : " + time/1e6 + "ms");

        System.out.println("Start Training");
        for(int i = 0;i<epoch;i++){
            time = System.nanoTime();
            network.gradientDescent(new double[][][]{featureInput,featureOutput},learningRate);
            double error = 0;
            for(int j = 0;j<10;j++) {
                double[] input = new double[10];
                input[j] = 1;
                network.propagation(input);
                double[] output = network.getOutput();
                error += midError(featureOutput[(int) (Math.random()*trainingSetSize)],output);
            }
            error/=10;
            time = System.nanoTime()-time;
            System.out.println("At epoch " + i + " total medium error was " + error + "(Duration : " + time/1e6 + "ms)");
        }

        System.out.println("End of Training");
        for(int i = 0;i<10;i++){
            double[] input = new double[10];
            input[i] = 1;
            network.propagation(input);
            double[] output = network.getOutput();
            BufferedImage img = new BufferedImage(28,28,BufferedImage.TYPE_3BYTE_BGR);
            for(int x = 0;x<28;x++){
                for(int y = 0;y<28;y++){
                    int greyscale = (int)(output[x * 28 + y]*255);
                    img.setRGB(x,y,new Color(greyscale,greyscale,greyscale).getRGB());
                }
            }
            ImageIO.write(img,"png",new File("think " + i + " .png"));
        }

    }

    /*public static void testLSTM() throws ArraysSizeDifferents {
        int[] representation = {256,512,512,256};
        double learningRate = 0.03;
        LongShortTermMemory lstm = new LongShortTermMemory(representation);
        Scanner scn = new Scanner(System.in);
        char lastLetter = 0;
        while(true) {
            String word = scn.next();
            double[] input = new double[256];
            input[lastLetter] = 1;
            lstm.propagation(input);
            double[] output = new double[256];
            output[word.charAt(0)] = 1;
            lstm.retropropagation(output);
            lstm.actualiseWeights(learningRate);
            lstm.resetErrors();
            for(int i = 1;i<word.length()-1;i++){
                input = new double[256];
                input[word.charAt(i)] = 1;
                lstm.propagation(input);
                output = new double[256];
                output[word.charAt(i+1)] = 1;
                lstm.retropropagation(output);
                lstm.actualiseWeights(learningRate);
                lstm.resetErrors();
            }
            lastLetter = word.charAt(word.length()-1);
            char tempChar = lastLetter;
            for(int i = 0;i<10;i++){
                input = new double[256];
                input[tempChar] = 1;
                lstm.propagation(input);
                tempChar = (char) Functions.indexOf(lstm.getOutput(), Functions.maxInArray(lstm.getOutput()));
                System.err.print(tempChar);
            }
        }
    }*/

    static void adversarialTest() throws ItemsOutOfIndex, IOException, BadIdxType, ArraysSizeDifferents {
        int outputSize = 28*28;
        int[] generatorRepresentation = {100,100,230,outputSize};
        int[] discriminatorRepresentation = {outputSize,100,1};
        AdversarialNetwork network = new AdversarialNetwork(generatorRepresentation,discriminatorRepresentation);

        double generatorLearningRate = 0.3;
        double discriminatorLearningRate = 0.3;
        int epochs = 30000;
        int trainingSetSize = 100;

        double[][] dataIn = new double[trainingSetSize][outputSize];
        double[][] dataOut = new double[trainingSetSize][discriminatorRepresentation[discriminatorRepresentation.length-1]];
        MNISTFeature[] features = IdxFormat.getIdx(trainLabelPath,trainImgPath,0,trainingSetSize);
        for(int i = 0;i<trainingSetSize;i++){
            dataIn[i] = features[i].getInput();
            dataOut[i] = features[i].getOutput();
        }
        for(int i = 0;i<epochs;i++) {
            network.train(dataIn, generatorLearningRate, discriminatorLearningRate);
            if(true) {
                System.out.println("Epoch : " + i + " discrimator win " + ((double)network.discriminatorWin/(trainingSetSize*2))*100 + "%");
                for (int j = 0; j < 1; j++) {
                    double[] gen = network.getGenerated();
                    ImageIO.write(Functions.arrayToImg(gen, 28), "png", new File("GenerationTest\\" +
                            "Gen " + i + " image number " + j + ".png"));
                }
            }
        }
    }


    public static void autoencoderTest() throws ItemsOutOfIndex, IOException, BadIdxType, ArraysSizeDifferents {

        int imgSz = 28*28;
        int trainingSetSize = 6000;
        double learningRate = 0.05;
        int epochs = 100;
        int nbTest = 100;

        System.out.println("Reading");
        double[][] dataIn = new double[trainingSetSize][imgSz];
        MNISTFeature[] features = IdxFormat.getIdx(trainLabelPath,trainImgPath,0,trainingSetSize);
        for(int i = 0;i<trainingSetSize;i++){
            dataIn[i] = features[i].getInput();
        }

        System.out.println("Training");
        int[] representation = {imgSz,100,10,1,10,100,300,imgSz};
        int modPos = Functions.indexOf(representation,1);

        FeedForward network = new FeedForward(representation);
        for(int j = 0;j<epochs;j++) {
            for (int i = 0; i < trainingSetSize; i++) {
                network.propagation(dataIn[i]);
                network.retropropagation(dataIn[i]);
                network.actualiseWeights(learningRate);
                network.resetErrors();
            }
            System.out.println("Epoch : " + j);
            for(int i = 0;i<nbTest;i++){
                network.layers[modPos].setValues(new double[]{1d/nbTest*i});
                for(int k = modPos;k<representation.length-1;k++){
                    ((RetrobakeLayer)network.layers[k+1]).propagation(network.layers[k]);
                }
                double[] out = network.getOutput();
                BufferedImage img = Functions.arrayToImg(out,28);
                ImageIO.write(img,"png",new File("Autoencoder\\epoch "+ j +" "+(1d/nbTest*i)+" .png"));
            }
        }


    }

    public static void geneticTest() throws BadIdxType, ItemsOutOfIndex, ArraysSizeDifferents {

        int firstGenerationCount = 4;
        int epochs = 1000;
        int epoch2 = 0;
        double mutationRate = 1;

        int[] representation = {28*28,100,10};
        double learningRate = 0;
        int trainingSetSize = 200;
        int testSetSize = 150;

        double[][] mnistFeaturesInput = new double[trainingSetSize][(28)*(28)];
        double[][] mnistFeaturesOutput = new double[trainingSetSize][10];
        System.out.println("Read Images");
        long tr1 = System.nanoTime();
        MNISTFeature[] mnistFeature = new MNISTFeature[trainingSetSize];
        try {
            mnistFeature = IdxFormat.getIdx(trainLabelPath,trainImgPath,0,trainingSetSize);
        } catch (IOException e) {
            e.printStackTrace();
        }
        for(int i = 0;i<trainingSetSize;i++){

            /*double[][] featureTemp = mnistFeature[i].getArrayInput();
                                double[][][] inputTemp = ConvolutionalNN.getFormOfThis(featureTemp,4,4);
                                for(int t = 0;t<inputTemp.length;t++){
                                    inputTemp[t] = ConvolutionalNN.maxPooling(ConvolutionalNN.convolutionAnReLU(featureTemp,inputTemp[t]),maxSzX,maxSzY);
                                }
            featureTemp = ConvolutionalNN.maxPooling(featureTemp,maxSzX,maxSzY);
            */
            mnistFeaturesInput[i] = mnistFeature[i].getInput();
            mnistFeaturesOutput[i] = mnistFeature[i].getOutput();

        }

        double[][] testFeaturesInput = new double[testSetSize][(28)*(28)];
        double[][] testFeaturesOutput = new double[testSetSize][10];
        MNISTFeature[] mnistFeature1 = new MNISTFeature[testSetSize];
        try {
            mnistFeature1 = IdxFormat.getIdx(testLabelPath,testImgPath,0,testSetSize);
        } catch (IOException e) {
            e.printStackTrace();
        }
        for(int i = 0;i<testSetSize;i++){
            /*double[][] featureTemp = mnistFeature1[i].getArrayInput();
                                double[][][] inputTemp = ConvolutionalNN.getFormOfThis(featureTemp,4,4);
                                for(int t = 0;t<inputTemp.length;t++){
                                    inputTemp[t] = ConvolutionalNN.maxPooling(ConvolutionalNN.convolutionAnReLU(featureTemp,inputTemp[t]),maxSzX,maxSzY);
                                }
            featureTemp = ConvolutionalNN.maxPooling(featureTemp,maxSzX,maxSzY);
            */
            testFeaturesInput[i] = mnistFeature1[i].getInput();
            testFeaturesOutput[i] = mnistFeature1[i].getOutput();
        }

        System.out.println("Train");

        Genetical<FeedForward> genetical;
        FeedForward[] firstGen = new FeedForward[firstGenerationCount];
        for(int i = 0 ;i<firstGenerationCount;i++){
            firstGen[i] = new FeedForward(representation);
            firstGen[i].setTrainSet(new double[][][]{mnistFeaturesInput,mnistFeaturesOutput});
            firstGen[i].setTestSet(new double[][][]{testFeaturesInput,testFeaturesOutput});
        }
        genetical = new Genetical<>(firstGen);
        for(int i = 0 ;i<epochs;i++){
            ArrayList<FeedForward> genNetworks = genetical.getGeneration();
            for(int k = 0;k<epoch2;k++){
            for(int j = 0; j<genNetworks.size();j++){
                genNetworks.get(j).train(learningRate);
            }
            }
            genetical.setGeneration(genNetworks);
            genetical.scoring();

            FeedForward best = genetical.getBest();
            double score = best.test(new double[][][]{testFeaturesInput,testFeaturesOutput});
            System.out.println("Score of the best at gen " + i + " : " + score);

            genetical.killAndBorn(firstGenerationCount/2,firstGenerationCount/2,1,mutationRate);

        }
    }

    public static void microGeneticTest(){
        int firstGenerationCount = 4;
        int epochs = 10;
        double mutationRate = 0.01;
        double target = 0.03;

        TestGenetic[] firstGeneration = new TestGenetic[firstGenerationCount];
        for(int i = 0;i<firstGenerationCount;i++){
            firstGeneration[i] = new TestGenetic(target);
            firstGeneration[i].setValue(Math.random()*2-1);
        }
        Genetical<TestGenetic> genetical = new Genetical<>(firstGeneration);

        for(int i = 0;i<epochs;i++){
            genetical.scoring();
            ArrayList<TestGenetic> tmp = genetical.getGeneration();
            System.out.print("Epoch " + i + " : [Value : Score]");
            for(int j = 0;j<tmp.size();j++){
                System.out.print(" ["+tmp.get(j).getValue()+" : "+tmp.get(j).getScore()+"]");
            }
            System.out.println();
            genetical.killAndBorn(firstGenerationCount/2,firstGenerationCount/2,1,mutationRate);
        }
    }

    public static void memoryTest() throws ArraysSizeDifferents {
        double[] trainSequence = {0.1,0.2,0.3};
        double learningRate = 0.1;

        int memorySize = 3;
        int[] memoryForm = {3,3};
        int[] networkForm = {3,2};
        MemorySequential network = new MemorySequential(memorySize,memoryForm,networkForm);
        for(int i = 0;i<10;i++) {
            network.learnSequence(trainSequence, learningRate);
        }
        System.out.println("Predicted : " + Arrays.toString(network.getSequence(10)));
    }

    static double midError(double[] need, double[] have){
        double error = 0;
        for(int i = 0;i<need.length;i++){
            error += Math.sqrt(Functions.quadraticCost(need[i],have[i]));
        }
        error/=need.length;
        return error;
    }
}
