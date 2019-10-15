package Extern.Exceptions;

public class BadIdxType extends Exception{
    public BadIdxType(int type){
        super("BadIdxType : need " + type);
    }
}