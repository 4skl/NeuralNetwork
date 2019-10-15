package Extern;
import Extern.Exceptions.BadIdxType;
import Extern.Exceptions.ItemsOutOfIndex;

import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.FileInputStream;
import java.io.IOException;

public class IdxFormat {

    static FileInputStream namesInputStream;
    static FileInputStream imagesInputStream;

    public static MNISTFeature getIdx(String fileNames, String fileImages, int index) throws IOException, BadIdxType, ItemsOutOfIndex {
        //long t1 = System.nanoTime();
        namesInputStream = new FileInputStream(fileNames);
        imagesInputStream = new FileInputStream(fileImages);
        BufferedImage img = getIdxImage(imagesInputStream,index);
        int label = getIdxLabel(namesInputStream,index);
        if(img == null)
            System.err.println("error img is null");
        MNISTFeature mnistFeature = new MNISTFeature(label,img);
        //System.out.println("Reading time : " + (System.nanoTime()-t1)/1e6 + "ms");
        return mnistFeature;
    }
    public static MNISTFeature[] getIdx(String fileNames, String fileImages, int start, int end) throws IOException, BadIdxType, ItemsOutOfIndex {
        //long t1 = System.nanoTime();
        namesInputStream = new FileInputStream(fileNames);
        imagesInputStream = new FileInputStream(fileImages);
        BufferedImage img[] = getIdxImage(imagesInputStream,start,end);
        int label[] = getIdxLabel(namesInputStream,start,end);
        MNISTFeature[] mnistFeature = new MNISTFeature[end-start];
        for(int i = 0;i<img.length;i++)
        mnistFeature[i] = new MNISTFeature(label[i],img[i]);
        //System.out.println("Reading time : " + (System.nanoTime()-t1)/1e6 + "ms");
        return mnistFeature;
    }

    public static int getIdxLabel(FileInputStream labelIs, int index) throws IOException, BadIdxType, ItemsOutOfIndex {
        int type = 0;
        //read 4 first bytes
        for(int i = 3;i>=0;i--){
        int tmp = labelIs.read();
        type += tmp*Math.pow(256,i);
        }
        if(type == 2049) {//2049 idx label type
            int numberOfItems = 0;
            //read 4 seconds bytes
            for(int i = 3;i>=0;i--){
                int tmp = labelIs.read();
                numberOfItems += tmp*Math.pow(256,i);
            }
            if(numberOfItems > index){
                labelIs.skip(index);
                int label = labelIs.read();
                return label;
            }else throw new ItemsOutOfIndex(index);//add throw error
        }else throw new BadIdxType(2049);
    }

    public static int[] getIdxLabel(FileInputStream labelIs, int start, int end) throws IOException, BadIdxType, ItemsOutOfIndex {
        int type = 0;
        //read 4 first bytes
        for(int i = 3;i>=0;i--){
            int tmp = labelIs.read();
            type += tmp*Math.pow(256,i);
        }
        if(type == 2049) {//2049 idx label type
            int numberOfItems = 0;
            //read 4 seconds bytes
            for(int i = 3;i>=0;i--){
                int tmp = labelIs.read();
                numberOfItems += tmp*Math.pow(256,i);
            }
            if(numberOfItems >= end){
                labelIs.skip(start);
                int label[] = new int[end-start];
                for(int i = 0;i<end-start;i++)
                label[i] = labelIs.read();
                return label;
            }else throw new ItemsOutOfIndex(end);//add throw error
        }else throw new BadIdxType(2049);
    }

    public static BufferedImage getIdxImage(FileInputStream imageIs,int index) throws IOException, BadIdxType, ItemsOutOfIndex {
        int type = 0;
        //read 4 first bytes
        for(int i = 3;i>=0;i--){
            int tmp = imageIs.read();
            type += tmp*Math.pow(256,i);
        }
        if(type == 2051){ //2051 idx image type

            int numberOfItems = 0;
            //read 4 seconds bytes
            for(int i = 3;i>=0;i--){
                int tmp = imageIs.read();
                numberOfItems += tmp*Math.pow(256,i);
            }
            int imageWidth = 0;
            //read 4 seconds bytes
            for(int i = 3;i>=0;i--){
                int tmp = imageIs.read();
                imageWidth += tmp*Math.pow(256,i);
            }
            int imageHeight = 0;
            //read 4 seconds bytes
            for(int i = 3;i>=0;i--){
                int tmp = imageIs.read();
                imageHeight += tmp*Math.pow(256,i);
            }
            BufferedImage image = new BufferedImage(imageWidth,imageHeight,BufferedImage.TYPE_USHORT_GRAY);
            if(numberOfItems > index){
                imageIs.skip(index*imageWidth*imageHeight);
                for(int y = 0; y < imageHeight;y++){
                    for(int x = 0; x < imageWidth;x++){
                        int greyscale = imageIs.read();
                        image.setRGB(x,y,new Color(greyscale,greyscale,greyscale).getRGB());
                    }
                }
                return image;
            }else {
                throw new ItemsOutOfIndex(index);//add throw error
            }
        }else {
            throw new BadIdxType(2051);
        }
    }

    public static BufferedImage[] getIdxImage(FileInputStream imageIs,int start, int end) throws IOException, BadIdxType, ItemsOutOfIndex {
        int type = 0;
        //read 4 first bytes
        for(int i = 3;i>=0;i--){
            int tmp = imageIs.read();
            type += tmp*Math.pow(256,i);
        }
        if(type == 2051){ //2051 idx image type

            int numberOfItems = 0;
            //read 4 seconds bytes
            for(int i = 3;i>=0;i--){
                int tmp = imageIs.read();
                numberOfItems += tmp*Math.pow(256,i);
            }
            int imageWidth = 0;
            //read 4 seconds bytes
            for(int i = 3;i>=0;i--){
                int tmp = imageIs.read();
                imageWidth += tmp*Math.pow(256,i);
            }
            int imageHeight = 0;
            //read 4 seconds bytes
            for(int i = 3;i>=0;i--){
                int tmp = imageIs.read();
                imageHeight += tmp*Math.pow(256,i);
            }
            BufferedImage image[] = new BufferedImage[end-start];
            if(numberOfItems >= end){
                imageIs.skip(start*imageWidth*imageHeight);
                for(int i = 0;i<end-start;i++) {
                    image[i] = new BufferedImage(imageWidth,imageHeight,BufferedImage.TYPE_USHORT_GRAY);
                    for (int y = 0; y < imageHeight; y++) {
                        for (int x = 0; x < imageWidth; x++) {
                            int greyscale = imageIs.read();
                            image[i].setRGB(x, y, new Color(greyscale, greyscale, greyscale).getRGB());
                        }
                    }
                }
                return image;
            }else {
                throw new ItemsOutOfIndex(end);//add throw error
            }
        }else {
            throw new BadIdxType(2051);
        }
    }

}