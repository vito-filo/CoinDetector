package com.example.coindetector.customview;

import java.util.List;
import com.example.coindetector.tflite.DetectorClassifier.Recognition;

public interface ResultsView {
    public void setResults(final List<Recognition> results);
}