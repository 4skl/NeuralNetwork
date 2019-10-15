package Temp;

import com.medi.MachineLearning.Functions;

import java.util.ArrayList;

public class ConvolutionalNN {

    public static double[][] convolutionAnReLU(double[][] input,double[][] piece){
        if(piece.length>input.length) return null;
        double[][] convolved = new double[input.length][];
        for(int x = 0;x<input.length;x++){
            convolved[x] = new double[input[x].length];
            for(int y = 0;y<input[x].length;y++){
                double moy = 0;
                int divMoy = 0;
                for(int xp = 0;xp<piece.length && xp+x<input.length;xp++){
                    for(int yp = 0;yp<piece[xp].length && yp+y<input[xp].length;yp++){
                        moy+=piece[xp][yp]*input[xp+x][yp+y];
                        divMoy++;
                    }
                }
                convolved[x][y] = Functions.reLU((divMoy>0) ? moy/divMoy : 0);
            }
        }
        return convolved;
    }

    public static double[][] maxPooling(double[][] input,int sizeX,int sizeY){
        double[][] maxPool = new double[(int) Math.ceil(((double) input.length)/sizeX)][];
        for(int x = 0;x<input.length;x+=sizeX) {
            maxPool[x/sizeX] = new double[(int) Math.ceil(((double) input[x].length)/sizeY)];
            for (int y = 0; y < input[x].length; y+=sizeY){

                double max = 0;
                for(int xp = x; xp<sizeX+x && xp+x<input.length;xp++){
                    for(int yp = y; yp<sizeY+y && yp+y<input[xp].length;yp++){
                       if(max<input[x][y]) max = input[x][y];
                    }
                }
                maxPool[x/sizeX][y/sizeX] = max;
            }
        }
        return maxPool;
    }

    public static double[][][] getFormOfThis(double[][] input,int xSize,int ySize){
        ArrayList<Double[][]> forms = new ArrayList<>();
        for(int x = 0;x<input.length-xSize;x++) {
            for (int y = 0; y < input[x].length-ySize; y++){
                Double[][] form = new Double[xSize][ySize];
                for(int xp = x; xp<xSize+x;xp++){
                    for(int yp = y; yp<ySize+y;yp++){
                        form[xp-x][yp-y] = input[xp][yp];
                    }
                }
                boolean dontExist = true;
                for(int i = 0;i<forms.size();i++){
                    if(Functions.isSame2DArray(form,forms.get(i))){
                        dontExist = false;
                        break;
                    }
                }
                if(dontExist) forms.add(form);
            }
        }
        double[][][] formsArray = new double[forms.size()][][];
        for(int x = 0;x<forms.size();x++) {
            formsArray[x] = Functions.double2DToPrimitive(forms.get(x));
        }
        return formsArray;
    }

}
