package com.medi.GUI;

import javax.swing.*;
import java.awt.*;

public class GUIMain extends JFrame{
    public JTextArea textArea;
    BoxLayout layout;
    public JTree tree;
    public GUIMain(){
        this.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);

        BoxLayout boxLayout = new BoxLayout(new Container(),BoxLayout.LINE_AXIS);

        textArea = new JTextArea("Text");
        tree = new JTree();
        this.setLayout(boxLayout);
        this.setSize(200,150);
        this.setVisible(true);
    }
}
