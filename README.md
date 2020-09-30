# CoinDetector
Android application for coin detection and classification using Tensorflow lite.

This application is made on top of Tensorflow's demo apps **[Object detection](https://github.com/tensorflow/examples/blob/master/lite/examples/object_detection/android/README.md)** and **[Image classification](https://github.com/tensorflow/examples/blob/master/lite/examples/image_classification/android/README.md)**.

The app is basically divided into 2 steps:
1. Obgect detection: in order to recognize coins and thir bounding boxes.
2. Image classification: takes single coin images and classifies them.

Note: usually this task can be performed by a single neural network model, unfortunatelly due to Tensorflow lite limitation and/or low device power i was forced to devide the workflow in 2 steps.

# Object detection


