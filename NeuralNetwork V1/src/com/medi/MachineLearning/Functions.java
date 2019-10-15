package com.medi.MachineLearning;

import com.medi.MachineLearning.Exceptions.ArraysSizeDifferents;

import java.awt.*;
import java.awt.image.BufferedImage;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Random;

public class Functions{

    final public static Random random = new Random();


    public static double[] convolutionAgregation(int[] arraySize, double[] array, int[] filterSize, double[] filter){

        double[] result = new double[array.length];

        for(int i = 0;i<array.length;i++){//pour chaque element de array

            int position = 0;
            for(int j = 0;j<filterSize.length;j++){//itérer filterSize
                for(int k = 0;k<filterSize[j];k++){//itérer dans filterSize
                    int[] tempNewPosFilter = positions(filterSize,position);
                    int[] tempNewPosArray = positions(arraySize,i);
                    for(int l = 0;l<tempNewPosArray.length;l++){
                        tempNewPosArray[l]+=tempNewPosFilter[l];//Décale le filtre
                    }
                    result[i] += filter[position]*array[position(arraySize,tempNewPosArray)%array.length];//Si dernier element, prend premier (boucle)
                    position++;
                }
            }
            result[i]/=position;
        }
        return result;
    }

    static int[] positions(int[] sizes, int position){
        int[] positions = new int[sizes.length];
        for(int i = 0;i<sizes.length;i++){
            if(position != 0 && sizes[i]!=0) {
                int temp = position % sizes[i];
                positions[i] = temp;
                position -= temp;
                position /= sizes[i];
            }else{
                positions[i] = 0;
            }
        }
        return positions;
    }

    static int position(int[] sizes, int[] positions){
        int position = positions[0];
        int factor = 1;
        for(int i = 1;i<positions.length;i++){
            factor*=sizes[i-1];
            position+=positions[i]*factor;
        }
        return position;
    }


    public static BufferedImage arrayToImg(double[] arrayImg,int width){
        int height = arrayImg.length/width;
        BufferedImage out = new BufferedImage(width,height,BufferedImage.TYPE_USHORT_GRAY);
        for (int x = 0; x < width; x++) {
            for(int y = 0;y<height;y++) {
                int color = (int) (arrayImg[x * height + y]*255);
                out.setRGB(x,y,new Color(color,color,color).getRGB());
            }
        }
        return out;
    }

    public static BufferedImage arrayToImg(double[][] arrayImg){
        return null;
    }

    public static double[] double1DToPrimitive(Object[] in){
        double[] out = new double[in.length];
        for(int x = 0;x<in.length;x++){
            out[x] = (double) in[x];
        }
        return out;
    }
    public static double[] array3DinArray1D(double[][][] array3D){
        ArrayList<Double> tempArray = new ArrayList<Double>();
        for(int x = 0;x<array3D.length;x++){
            for(int y = 0;y<array3D[x].length;y++){
                for(int z = 0;z<array3D[y].length;z++){
                    tempArray.add(array3D[x][y][z]);
                }
            }
        }
        return Functions.double1DToPrimitive(tempArray.toArray());
    }
    public static double[] array2DinArray1D(double[][] array2D){
        ArrayList<Double> tempArray = new ArrayList<Double>();
        for(int x = 0;x<array2D.length;x++){
            for(int y = 0;y<array2D[x].length;y++){
                    tempArray.add(array2D[x][y]);
            }
        }
        return Functions.double1DToPrimitive(tempArray.toArray());
    }
    public static double[][] double2DToPrimitive(Object[][] in){
        double[][] out = new double[in.length][];
        for(int x = 0;x<in.length;x++){
            out[x] = new double[in[x].length];
            for(int y = 0;y<in.length;y++) {
                out[x][y] = (double) in[x][y];
            }
        }
        return out;
    }
    public static double[][][] double3DToPrimitive(Object[][][]  in){
        double[][][] out = new double[in.length][][];
        for(int x = 0;x<in.length;x++){
            out[x] = new double[in[x].length][];
            for(int y = 0;y<in.length;y++) {
                out[x][y] = new double[in[x].length];
                for (int z = 0; z < in.length; z++) {
                    out[x][y][z] = (double) in[x][y][z];
                }
            }
        }
        return out;
    }
    public static <T> boolean isSame2DArray(T[][] array1, T[][] array2){
        if(array1.length != array2.length)return false;
        for(int x = 0;x<array1.length;x++){
            if(array1[x].length != array2[x].length) return false;
            for(int y = 0;y<array1[x].length;y++){
                if(array1[x][y]!=array2[x][y]) return false;
            }
        }
        return true;
    }

    public static long complexity(int[] representation){
        long size = 0;
        for(int i = 0; i<representation.length-1;i++){
            size += representation[i]*representation[i+1];
        }
        return size;
    }

    public static double scalarProduct(double[] value1, double[] value2) throws ArraysSizeDifferents {
        if(value1.length != value2.length){
            throw new ArraysSizeDifferents();
        }
        double sum = 0;
        for(int i = 0;i<value1.length;i++){
            sum += value1[i]*value2[i];
        }
        return sum;
    }
    public static double[] product(double[] value1, double[] value2) throws ArraysSizeDifferents {
        if(value1.length != value2.length){
            throw new ArraysSizeDifferents();
        }
        double[] product = new double[value1.length];
        for(int i = 0;i<value1.length;i++){
            product[i] = value1[i]*value2[i];
        }
        return product;
    }
    public static double[] sum(double[] value1, double[] value2) throws ArraysSizeDifferents {
        if(value1.length != value2.length){
            throw new ArraysSizeDifferents();
        }
        double[] sum = new double[value1.length];
        for(int i = 0;i<value1.length;i++){
            sum[i] = value1[i]+value2[i];
        }
        return sum;
    }
    public static double[] sub(double[] value1, double[] value2) throws ArraysSizeDifferents {
        if(value1.length != value2.length){
            throw new ArraysSizeDifferents();
        }
        double[] sub = new double[value1.length];
        for(int i = 0;i<value1.length;i++){
            sub[i] = value1[i]-value2[i];
        }
        return sub;
    }
    public static double[] div(double[] value1, double[] value2) throws ArraysSizeDifferents {
        if(value1.length != value2.length){
            throw new ArraysSizeDifferents();
        }
        double[] div = new double[value1.length];
        for(int i = 0;i<value1.length;i++){
            div[i] = value1[i]/value2[i];
        }
        return div;
    }

    public static double[] concatenate(double[] first, double[] second){
        double[] out = new double[first.length+second.length];
        for(int i = 0;i<first.length;i++){
            out[i] = first[i];
        }
        for(int i = 0;i<second.length;i++){
            out[i+first.length] = second[i];
        }
        return out;
    }

    public static double maxInArray(double[] array){
        double max = Double.MIN_VALUE;
        for(int i = 0;i<array.length;i++) {
            if (max < array[i]) max = array[i];
        }

        return max;
    }
    public static double minInArray(double[] array){
        double max = Double.MIN_VALUE;
        for(int i = 0;i<array.length;i++) {
            if (max > array[i]) max = array[i];
        }

        return max;
    }
    public static <T>int indexOf(T[] array, T value){ ;
        for(int i = 0;i<array.length;i++) {
            if (value == array[i]) return i;
        }
        return -1;
    }

    public static int indexOf(double[] array, double value){ ;
        for(int i = 0;i<array.length;i++) {
            if (value == array[i]) return i;
        }
        return -1;
    }

    public static int indexOf(int[] array, int value){ ;
        for(int i = 0;i<array.length;i++) {
            if (value == array[i]) return i;
        }
        return -1;
    }

    public static double[] matricialProduct(double[] a, double[][] b){
        double[] c = new double[a.length];
        // b[x].length constant
        //a.length == b[x].length

        for(int i = 0;i<a.length;i++){
            for(int j = 0;j<b[0].length;j++){
                c[j] += a[i] * b[i][j];
            }
        }
        return c;
    }

    public static double[][] inversionGaussJordan(double[][] a){
        double[][] b = a.clone();
        int r = 0;
        for(int j = 1;j<a.length;j++){
            int k = indexOf(a[j],maxInArray(a[j]));
            if(a[j][k] != 0){
                r++;
                for(int i = 0;i<a[j].length;i++){
                    b[j][i] /= a[j][k];
                }
                double c = b[j][r];
                b[j][r] = b[j][k];
                b[j][k] = c;
                for(int i = 0;i<a[j].length;i++){
                    if(i!=r){
                        for(int l = 0;l<a[j].length;l++){
                            b[j][l] -= b[j][r] / a[j][j];
                        }
                    }
                }
            }
        }
        return b;
    }

    public static double[] imgToArray(BufferedImage in){
        int width = in.getWidth();
        int height = in.getHeight();
        double[] out = new double[width * height];
        //System.out.print("it's going to be returned : [");//#43
        for (int x = 0; x < width; x++) {
            for (int y = 0; y < height; y++) {
                Color color = new Color(in.getRGB(x, y));
                out[x * height + y] = color.getRed() / 255.0;//because greyscale r = g = b
                //System.out.print(", "+(double) (color.getRed()/255.0));//#43
            }
        }
        //System.out.println("]");//#43
        return out;
    }

    public static double[] imgRVBToArray(BufferedImage img){
        int height = img.getHeight();
        int width = img.getWidth();
        double[] out = new double[height*width*3];
        for(int x = 0;x<width;x++){
            for(int y = 0;y<height;y++){
                Color color = new Color(img.getRGB(x,y));
                out[x*height+y] = color.getRed()/255.0;
                out[x*height+y] = color.getGreen()/255.0;
                out[x*height+y] = color.getBlue()/255.0;
            }
        }
        return out;
    }

    public static double[] getRandomArray(int size){
        double[] out = new double[size];
        for(int i = 0;i<size;i++){
            out[i] = random();
        }
        return out;
    }

    public static double[] sigmoid(double[] x){
        double[] out = new double[x.length];
        for(int i = 0;i<out.length;i++){
            out[i] = sigmoid(x[i]);
        }
        return out;
    }
    public static double[] reciprocalSigmoid(double[] x){
        double[] out = new double[x.length];
        for(int i = 0;i<out.length;i++){
            out[i] = reciprocalSigmoid(x[i]);
        }
        return out;
    }

    public static double sigmoid(double x){
        return 1d/(1+Math.exp(-x));
    }
    public static double derivativeSigmoid(double x){
        return sigmoid(x)*(1-sigmoid(x));
    }
    public static double reciprocalSigmoid(double x){
        return Math.log(x/1-x);
    }

    public static double tanh(double x){
        return Math.tanh(x);
    }

    public static double reLU(double x){
        return (x>0) ? x : 0;
    }
    public static double derivativeReLU(double x){
        return (x>0) ? 1 : 0;
    }

    public static double random(){
        return random.nextDouble()*2-1;
    }

    public static double summedAbsoluteCost(double[] input, double[] output){
        double cost = 0;
        for(int i = 0;i<input.length;i++){
            cost += Math.abs(input[i]-output[i]);
        }
        return cost;
    }

    public static double quadraticCost(double input, double output){
        return Math.pow(input-output,2);
    }
    public static double derivativeQuadraticCost(double input, double output){
        return 2*(input-output);
    }

}

