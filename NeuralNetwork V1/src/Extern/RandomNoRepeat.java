package Extern;

import java.util.ArrayList;
import java.util.Random;

public class RandomNoRepeat {
    private ArrayList<Integer> randoms;
    private Random rnd;
    RandomNoRepeat(int seed){
        rnd = new Random(seed);
        randoms = new ArrayList<Integer>();
    }
    public RandomNoRepeat(){
        rnd = new Random();
        randoms = new ArrayList<Integer>();
    }
    public int next(int max){
        int number = rnd.nextInt(max);
        while(randoms.contains(number)){
            number = rnd.nextInt(max);
        }
        randoms.add(number);
        return number;
    }
}
