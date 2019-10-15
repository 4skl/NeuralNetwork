package com.medi.MachineLearning.NeuralNetworks;

import com.medi.MachineLearning.Functions;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;

public class Genetical <GeneticType extends Genetic>{
    ArrayList<GeneticType> generation;
    ArrayList<Double> scores;
    public Genetical(GeneticType[] generation){
        this.generation = new ArrayList<>();
        this.scores = new ArrayList<>();
        this.generation.addAll(Arrays.asList(generation));
        scores.addAll(Arrays.asList(new Double[generation.length]));
    }
    public void scoring(){//Parallelize
        scores = new ArrayList<Double>();
        for(int i = 0;i<generation.size();i++){
            scores.add(generation.get(i).getScore());
        }
    }

    public void killAndBorn(int numberToKill, int numberToBorn, int numberPerBorn, double mutationRate){

        int[] bestIndex = new int[numberToBorn];
        double[] best = new double[numberToBorn];

        int[] badestIndex = new int[numberToKill];
        double[] badest = new double[numberToKill];

        //kill
        ArrayList<Double> scoresSorted = (ArrayList<Double>) scores.clone();
        scoresSorted.sort(new Comparator<Double>() {
            @Override
            public int compare(Double o1, Double o2) {
                if(o1<o2){
                    return 1;
                }
                else if(o1>o2){
                    return  -1;
                }else{
                    return 0;
                }
            }
        });
        for(int i = 0;i<numberToKill;i++){
            int index = scores.indexOf(scoresSorted.get(scores.size()-1));
            generation.remove(index);
            scores.remove(index);
        }

        //born
        ArrayList<GeneticType> newGeneration = new ArrayList<>();
        for(int i = 0;i<numberToBorn;i++){
            GeneticType[] newGenerated = generation.get(scores.indexOf(scoresSorted.get(i))).getChilds(numberPerBorn,mutationRate);
            newGeneration.addAll(Arrays.asList(newGenerated));
        }
        generation.addAll(newGeneration);

    }

    public GeneticType getBest(){//need scoring() before
        double score = 0;
        int indexOfBest = 0;
        for(int i = 0;i<generation.size();i++){
            if(score<scores.get(i)){
                score = scores.get(i);
                indexOfBest = i;
            }
        }
        return generation.get(indexOfBest);
    }

    public ArrayList<GeneticType> getGeneration(){
        return generation;
    }
    public void setGeneration(ArrayList<GeneticType> generation){
        this.generation = generation;
    }

}
