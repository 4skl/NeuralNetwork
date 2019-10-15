package Extern.Exceptions;

public class ItemsOutOfIndex extends Exception{
    public ItemsOutOfIndex(int index){
        super("ItemsOutOfIndex : " + index);
    }
}