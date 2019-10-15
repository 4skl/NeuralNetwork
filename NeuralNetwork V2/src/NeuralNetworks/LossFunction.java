package NeuralNetworks;

public class LossFunction {
    LossType lossType;

    public double delta = 1;//to determine

    public LossFunction(LossType lossType){
        this.lossType = lossType;
    }

    public double loss(double[] input, double[] target) throws UnknownLossType {
        double loss = 0;
        switch (lossType){
            case Quadratic:
                for(int i = 0;i<input.length;i++) loss += Math.pow(input[i]-target[i],2);
                break;
            case Absolute:
                for(int i = 0;i<input.length;i++) loss += Math.abs(input[i]-target[i]);
                break;
            case Relative:
                for(int i = 0;i<input.length;i++) loss += input[i]-target[i];
                break;
            case Log_Cosh:
                for(int i = 0;i<input.length;i++) loss += Math.log(Math.cosh(input[i]-target[i]));
                break;
            default:
                throw new UnknownLossType();
        }
        return loss;
    }
}

class UnknownLossType extends Exception {

}
