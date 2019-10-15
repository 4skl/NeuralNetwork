package NeuralNetworks;

import java.io.Serializable;

public class Tensor implements Serializable {
    int dimensions[];//x,y,z,...
    double data[];//

    public Tensor(int dimensions[]) {
        this.dimensions = dimensions;
        int dimension = dimensions[0];
        for(int i = 1; i< dimensions.length; i++) dimension*= dimensions[i];
        this.data = new double[dimension];
    }

    public void setData(double[] data) {
        this.data = data;
    }

    public double[] getData(){
        return data;
    }

    public void setDimensions(int[] dimensions) {
        this.dimensions = dimensions;//to modify for making dimension good size for data
    }

    public void setDataAt(double value, int pos){
        data[pos] = value;
    }

    public double getDataAt(int pos){
        return data[pos];
    }



}
