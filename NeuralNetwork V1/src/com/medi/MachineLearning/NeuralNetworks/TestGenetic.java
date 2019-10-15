package com.medi.MachineLearning.NeuralNetworks;

public class TestGenetic implements Genetic {
    double value;
    double need;

    public TestGenetic(double need){
        this.need = need;
    }

    public void setValue(double value){
        this.value = value;
    }

    public double getValue() {
        return value;
    }

    @Override
    public double getScore() {
        return (need-value);
    }

    @Override
    public TestGenetic[] getChilds(int childCount, double dispersion) {
        TestGenetic[] childs = new TestGenetic[childCount];
        for(int i = 0;i<childCount;i++){
            childs[i] = new TestGenetic(need);
            childs[i].setValue(value+(Math.random()*2-1)*dispersion);
        }
        return childs;
    }
}
