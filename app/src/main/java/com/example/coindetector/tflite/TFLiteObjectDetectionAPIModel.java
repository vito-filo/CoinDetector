package com.example.coindetector.tflite;

import android.app.Activity;
import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.RectF;
import android.os.Debug;
import android.os.Trace;
import android.renderscript.Allocation;
import android.renderscript.Element;
import android.renderscript.RenderScript;
import android.renderscript.ScriptIntrinsicBlur;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Vector;

import org.tensorflow.lite.Interpreter;
import com.example.coindetector.env.Logger;
import org.tensorflow.lite.support.image.TensorImage;

/**
 * Wrapper for frozen detection models trained using the Tensorflow Object Detection API:
 * - https://github.com/tensorflow/models/tree/master/research/object_detection
 * where you can find the training code.
 *
 * To use pretrained models in the API or convert to TF Lite models, please see docs for details:
 * - https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md
 * - https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/running_on_mobile_tensorflowlite.md#running-our-model-on-android
 */
public class TFLiteObjectDetectionAPIModel implements DetectorClassifier {
    private static final Logger LOGGER = new Logger();

    // Only return this many results.
    private static final int CROP_WIDTH = 480;
    private static final int CROP_HEIGHT = 640;
    private static final int IMG_WIDTH = 480;
    private static final int IMG_HEIGHT = 640;

    private static final int NUM_DETECTIONS = 10;
    // Float model
    private static final float IMAGE_MEAN = 127.5f;
    private static final float IMAGE_STD = 127.5f;
    // Number of threads in the java app
    private static final int NUM_THREADS = 4;

    private boolean isModelQuantized;
    // Config values.
    private int inputSize;
    // Pre-allocated buffers.
    private Vector<String> labels = new Vector<String>();
    private int[] intValues;
    // outputLocations: array of shape [Batchsize, NUM_DETECTIONS,4]
    // contains the location of detected boxes
    private float[][][] outputLocations;
    // outputClasses: array of shape [Batchsize, NUM_DETECTIONS]
    // contains the classes of detected boxes
    private float[][] outputClasses;
    // outputScores: array of shape [Batchsize, NUM_DETECTIONS]
    // contains the scores of detected boxes
    private float[][] outputScores;
    // numDetections: array of shape [Batchsize]
    // contains the number of detected boxes
    private float[] numDetections;

    private ByteBuffer imgData;

    private Interpreter tfLite;
    private static Activity detectorActivity;

    private static Vector<RectF> boxes;
    private static final String TF_IC_MODEL_FILE = "model.tflite";
    private static final String TF_IC_LABEL_FILE = "labels_coins.txt";
    private static CoinClassifier coinClassifier;

    /** Input image TensorBuffer. */
    private TensorImage inputImageBuffer;

    private TFLiteObjectDetectionAPIModel() {}

    /** Memory-map the model file in Assets. */
    private static MappedByteBuffer loadModelFile(AssetManager assets, String modelFilename)
            throws IOException {
        AssetFileDescriptor fileDescriptor = assets.openFd(modelFilename);
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }

    /**
     * Initializes a native TensorFlow session for classifying images.
     *
     * @param assetManager The asset manager to be used to load assets.
     * @param modelFilename The filepath of the model GraphDef protocol buffer.
     * @param labelFilename The filepath of label file for classes.
     * @param inputSize The size of image input
     * @param isQuantized Boolean representing model is quantized or not
     */
    public static DetectorClassifier create(
            final AssetManager assetManager,
            final String modelFilename,
            final String labelFilename,
            final int inputSize,
            final boolean isQuantized,
            Activity activity)
            throws IOException {
        final TFLiteObjectDetectionAPIModel d = new TFLiteObjectDetectionAPIModel();
        detectorActivity = activity;
        // ## Labelmap for object detection
        String actualFilename = labelFilename.split("file:///android_asset/")[1];
        InputStream labelsInput = assetManager.open(actualFilename);
        BufferedReader br = new BufferedReader(new InputStreamReader(labelsInput));
        String line;
        while ((line = br.readLine()) != null) {
            LOGGER.w(line);
            d.labels.add(line);
        }
        br.close();

        // ### Labelmap for object classification
        labelsInput = assetManager.open(TF_IC_LABEL_FILE);
        br = new BufferedReader(new InputStreamReader(labelsInput));
        while ((line = br.readLine()) != null) {
            LOGGER.w(line);
            d.labels.add(line);
        }
        br.close();
        // ########

        d.inputSize = inputSize;

        try {
            d.tfLite = new Interpreter(loadModelFile(assetManager, modelFilename));
            coinClassifier = new CoinClassifier(activity);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }

        d.isModelQuantized = isQuantized;
        // Pre-allocate buffers.
        int numBytesPerChannel;
        if (isQuantized) {
            numBytesPerChannel = 1; // Quantized
        } else {
            numBytesPerChannel = 4; // Floating point
        }
        d.imgData = ByteBuffer.allocateDirect(1 * CROP_WIDTH * CROP_HEIGHT * 3 * numBytesPerChannel);
        d.imgData.order(ByteOrder.nativeOrder());
        d.intValues = new int[CROP_WIDTH * CROP_HEIGHT];

        d.tfLite.setNumThreads(NUM_THREADS);
        d.outputLocations = new float[1][NUM_DETECTIONS][4];
        d.outputClasses = new float[1][NUM_DETECTIONS];
        d.outputScores = new float[1][NUM_DETECTIONS];
        d.numDetections = new float[1];

        boxes = new Vector<RectF>();
        return d;
    }

    @Override
    public List<Recognition> recognizeImage(final Bitmap bitmap) {
        // Log this method so that it can be analyzed with systrace.
        Trace.beginSection("recognizeImage");

        Trace.beginSection("preprocessBitmap");
        // Preprocess the image data from 0-255 int to normalized float based
        // on the provided parameters.
        bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());

        imgData.rewind();
        int c=0;
        for (int i = 0; i < CROP_HEIGHT; ++i) {
            for (int j = 0; j < CROP_WIDTH; ++j) {
                int pixelValue = intValues[c];
                c++;
                if (isModelQuantized) {
                    // Quantized model
                    imgData.put((byte) ((pixelValue >> 16) & 0xFF));
                    imgData.put((byte) ((pixelValue >> 8) & 0xFF));
                    imgData.put((byte) (pixelValue & 0xFF));
                } else { // Float model
                    imgData.putFloat((((pixelValue >> 16) & 0xFF) - IMAGE_MEAN) / IMAGE_STD);
                    imgData.putFloat((((pixelValue >> 8) & 0xFF) - IMAGE_MEAN) / IMAGE_STD);
                    imgData.putFloat(((pixelValue & 0xFF) - IMAGE_MEAN) / IMAGE_STD);
                }
            }
        }
        Trace.endSection(); // preprocessBitmap

        // Copy the input data into TensorFlow.
        Trace.beginSection("feed");
        outputLocations = new float[1][NUM_DETECTIONS][4];
        outputClasses = new float[1][NUM_DETECTIONS];
        outputScores = new float[1][NUM_DETECTIONS];
        numDetections = new float[1];

        Object[] inputArray = {imgData};
        Map<Integer, Object> outputMap = new HashMap<>();
        outputMap.put(0, outputLocations);
        outputMap.put(1, outputClasses);
        outputMap.put(2, outputScores);
        outputMap.put(3, numDetections);
        Trace.endSection();

        // Run the inference call.
        Trace.beginSection("run");
        tfLite.runForMultipleInputsOutputs(inputArray, outputMap);
        Trace.endSection();

        // Show the best detections.
        // after scaling them back to the input size.

        // You need to use the number of detections from the output and not the NUM_DETECTONS variable declared on top
        // because on some models, they don't always output the same total number of detections
        // For example, your model's NUM_DETECTIONS = 20, but sometimes it only outputs 16 predictions
        // If you don't use the output's numDetections, you'll get nonsensical data
        int numDetectionsOutput = Math.min(NUM_DETECTIONS, (int) numDetections[0]); // cast from float to integer, use min for safety

        final ArrayList<Recognition> recognitions = new ArrayList<>(numDetectionsOutput);
        for (int i = 0; i < numDetectionsOutput; ++i) {
            if(outputScores[0][i] > 0.9) {
                float left = outputLocations[0][i][1] * CROP_WIDTH;
                float top = outputLocations[0][i][0] * CROP_HEIGHT;
                float right = outputLocations[0][i][3] * CROP_WIDTH;
                float bottom = outputLocations[0][i][2] * CROP_HEIGHT;

                if (!overlapping((int) left, (int) top, (int) right, (int) bottom)){
                    boxes.add(new RectF(left - 250, top - 250, right + 250, bottom + 250));
                    final RectF detection =
                            new RectF(
                                    left,
                                    top,
                                    right,
                                    bottom);
                    // SSD Mobilenet V1 Model assumes class 0 is background class
                    // in label file and class labels start from 1 to number_of_classes+1,
                    // while outputClasses correspond to class index from 0 to number_of_classes
                    int labelOffset = 1;

                    // TODO image classification
                    if (left < 0) left = 0;
                    if (top < 0) top = 0;
                    if (right > IMG_WIDTH) right = IMG_WIDTH;
                    if (bottom > IMG_HEIGHT) bottom = IMG_HEIGHT;
                    //Debug.waitForDebugger();
                    Bitmap coin = Bitmap.createBitmap(bitmap, (int) left, (int) top, (int) (right - left), (int) (bottom - top));
                    //Debug.waitForDebugger();
                    final List<CoinClassifier.Recognition> results = coinClassifier.recognizeImage(coin, 90);
                    // ########
                    recognitions.add(
                            new Recognition(
                                    "" + i,
                                    //labels.get((int) outputClasses[0][i] + labelOffset), // localization label (coin)
                                    results.get(0).getId(), //classification label
                                    //outputScores[0][i], // localization conf
                                    results.get(0).getConfidence(), // classification conf
                                    detection));
                }
            }
        }
        boxes.clear();
        Trace.endSection(); // "recognizeImage"
        return recognitions;
    }


    @Override
    public void enableStatLogging(final boolean logStats) {}

    @Override
    public String getStatString() {
        return "";
    }

    @Override
    public void close() {}

    public void setNumThreads(int num_threads) {
        if (tfLite != null) tfLite.setNumThreads(num_threads);
    }

    @Override
    public void setUseNNAPI(boolean isChecked) {
        if (tfLite != null) tfLite.setUseNNAPI(isChecked);
    }

    public boolean overlapping(int left, int top, int right, int bottom){
        boolean overlap = false;
        RectF box;
        if(boxes != null) {
            for (int i = 0; i < boxes.size(); i++) {
                box = boxes.get(i);
                if (left > box.left && top > box.top && right < box.right && bottom < box.right)
                    overlap = true;

            }
        }
        return overlap;
    }
}
