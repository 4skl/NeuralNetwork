package com.medi.MachineLearning.NeuralNetworks;

public interface Genetic {
    double getScore();
    <GeneticType extends Genetic> GeneticType[] getChilds(int childCount, double dispersion);
}
