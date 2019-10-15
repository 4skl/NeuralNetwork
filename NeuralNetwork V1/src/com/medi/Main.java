package com.medi;



import Extern.Exceptions.BadIdxType;
import Extern.Exceptions.ItemsOutOfIndex;
import com.medi.MachineLearning.Exceptions.ArraysSizeDifferents;
import com.medi.MachineLearning.Functions;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.Arrays;

public class Main {

    public static void main(String[] args) throws ItemsOutOfIndex, BadIdxType, ArraysSizeDifferents, InterruptedException, IOException {
        Tester.passTest();
        System.exit(0);
        BufferedImage input = ImageIO.read(new File("lena.png"));

        int[] arraySize = {input.getWidth(),input.getHeight()};
        /*double[] array = {0.0,0.2,0.1,0.3,0.6,0.5,
                          0.1,0.1,0.5,0.4,0.3,0.7,
                          0.8,0.9,0.1,0.5,0.7,0.5,
                          0.1,0.7,0.4,0.3,0.8,0.2};
        /*=>
        *-2, 5,-3,-4, 5,-1
        * 1,
        *
        */
        double[] array = Functions.imgToArray(input);

        int[] filterSize = {13,0};
        double[] filter =  {0,0,0,0,0,0,0,0,0,0,0,0,1};

        double[] result = convolution(arraySize,array,filterSize,filter);

        BufferedImage output = Functions.arrayToImg(result,input.getWidth());
        ImageIO.write(output,"png",new File("lena-bord.png"));

    }
    static double[] convolution(int[] arraySize, double[] array, int[] filterSize, double[] filter){

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

}
