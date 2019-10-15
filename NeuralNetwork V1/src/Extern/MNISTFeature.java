package Extern;

import com.medi.MachineLearning.Functions;
import com.medi.MachineLearning.NeuralNetworks.FeedForward;
import Temp.DeepFeedForwardOld;
import com.medi.MachineLearning.Exceptions.ArraysSizeDifferents;

import java.awt.*;
import java.awt.image.BufferedImage;

public class MNISTFeature{
    private int number;
    private BufferedImage img;
    double[] input = null;
    public MNISTFeature(int number, BufferedImage img){
        this.number = number;
        this.img = img;
    }
    public int number(){
        return this.number;
    }
    public BufferedImage img(){
        return this.img;
    }

    public double[] getInput(){
        if(input == null) {
            int width = img.getWidth();
            int height = img.getHeight();

            double[] out = new double[width * height];
            //System.out.print("it's going to be returned : [");//#43
            for (int x = 0; x < width; x++) {
                for (int y = 0; y < height; y++) {
                    Color color = new Color(img.getRGB(x, y));
                    out[x * height + y] = color.getRed() / 255.0;//because greyscale r = g = b
                    //System.out.print(", "+(double) (color.getRed()/255.0));//#43
                }
            }
            //System.out.println("]");//#43
            return out;
        }else{
            return input;
        }
    }

    public double[][] getArrayInput(){
        int width = img.getWidth();
        int height = img.getHeight();

        double[][] out = new double[width][height];
        //System.out.print("it's going to be returned : [");//#43
        for(int x = 0;x<width;x++){
            for(int y = 0;y<height;y++){
                out[x][y] = new Color(img.getRGB(x,y)).getRed()/255.0;//beacause greyscale
            }
        }
        return out;
    }

    public double[] getOutput() {
        double[] output = new double[10];
        output[number] = 1;
        return output;
    }

    public boolean isGood(double[] output){
        return number == Functions.indexOf(output,output[number]);
    }
    public static boolean isGood(double[] output, double[] nededOutput){
        int think = Functions.indexOf(output,Functions.maxInArray(output));
        int real = Functions.indexOf(nededOutput,1);
        return think == real;
    }

    public static double test(DeepFeedForwardOld nn, double[][][] testData){
        double win = 0;
        for (int i = 0;i<testData[1].length;i++){
            nn.feedforward(testData[0][i]);
            int think = Functions.indexOf(nn.values[nn.values.length-1],Functions.maxInArray(nn.values[nn.values.length-1]));
            int real = Functions.indexOf(testData[1][i],1);
            if(think == real){
                win++;
            }
        }
        return win/testData[1].length;
    }
    public static double test(FeedForward nn, double[][][] testData) throws ArraysSizeDifferents {
        return nn.test(testData);
    }

    /*public static BufferedImage inverse(DeepFeedForward nn, double[] output){
        BufferedImage img = new BufferedImage(28,28,BufferedImage.TYPE_USHORT_GRAY);
        double[] input = nn.back(output);
        for(int x = 0;x<img.getWidth();x++){
            for (int y = 0;y<img.getHeight();y++){
                int value = (int) (input[x*img.getHeight()+y] * 255);
                img.setRGB(x,y,new Color(value,value,value).getRGB());
            }
        }
        return img;
    }*/
}

