/*
 * Copyright 2019 The TensorFlow Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.example.coindetector;

import android.app.Application;
import android.graphics.Bitmap;
import android.graphics.Bitmap.Config;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.graphics.Paint.Style;
import android.graphics.RectF;
import android.graphics.Typeface;
import android.media.ImageReader.OnImageAvailableListener;
import android.os.Debug;
import android.os.SystemClock;
import android.util.Size;
import android.util.TypedValue;
import android.widget.Toast;
import java.io.IOException;
import java.util.LinkedList;
import java.util.List;
import com.example.coindetector.customview.OverlayView;
import com.example.coindetector.customview.OverlayView.DrawCallback;
import com.example.coindetector.env.BorderedText;
import  com.example.coindetector.env.ImageUtils;
import com.example.coindetector.env.Logger;
import com.example.coindetector.openCV.OpencvObjectDetection;
import com.example.coindetector.tflite.DetectorClassifier;
import com.example.coindetector.tflite.TFLiteObjectDetectionAPIModel;
import com.example.coindetector.tracking.MultiBoxTracker;

/**
 * An activity that uses a TensorFlowMultiBoxDetector and ObjectTracker to detect and then track
 * objects.
 */
public class DetectorActivity extends CameraActivity implements OnImageAvailableListener {
    private static final Logger LOGGER = new Logger();

    // Configuration values for the prepackaged SSD model.
    private static final int TF_OD_API_INPUT_SIZE = 300;
    private static final int CROP_WIDTH = 300;
    private static final int CROP_HEIGHT = 300;
    private static final int IMG_WIDTH = 960;
    private static final int IMG_HEIGHT = 1280;
    private static final boolean TF_OD_API_IS_QUANTIZED = true;
    private static final String TF_OD_API_MODEL_FILE = "detect.tflite";
    private static final String TF_OD_API_LABELS_FILE = "file:///android_asset/labelmap.txt";
    private static final DetectorMode MODE = DetectorMode.TF_OD_API;
    // Minimum detection confidence to track a detection.
    private static final float MINIMUM_CONFIDENCE_TF_OD_API = 0.5f;
    private static final boolean MAINTAIN_ASPECT = false;
    private static final Size DESIRED_PREVIEW_SIZE = new Size(960, 1280);
    private static final boolean SAVE_PREVIEW_BITMAP = false;
    private static final float TEXT_SIZE_DIP = 10;
    OverlayView trackingOverlay;
    private Integer sensorOrientation;

    private DetectorClassifier tfliteDetector;
    private DetectorClassifier openCVDetector;

    private long lastProcessingTimeMs;
    private Bitmap rgbFrameBitmap = null;
    private Bitmap rgbVertival = null;
    private Bitmap croppedBitmap = null;
    private Bitmap cropCopyBitmap = null;

    private boolean computingDetection = false;

    private long timestamp = 0;

    private Matrix frameToCropTransform;
    private Matrix frameRotation;
    private Matrix cropToFrameTransform;

    private MultiBoxTracker tracker;

    private BorderedText borderedText;

    @Override
    public void onPreviewSizeChosen(final Size size, final int rotation) {
        final float textSizePx =
                TypedValue.applyDimension(
                        TypedValue.COMPLEX_UNIT_DIP, TEXT_SIZE_DIP, getResources().getDisplayMetrics());
        borderedText = new BorderedText(textSizePx);
        borderedText.setTypeface(Typeface.MONOSPACE);

        tracker = new MultiBoxTracker(this);


        try {
            tfliteDetector =
                    TFLiteObjectDetectionAPIModel.create(
                            getAssets(),
                            TF_OD_API_MODEL_FILE,
                            TF_OD_API_LABELS_FILE,
                            TF_OD_API_INPUT_SIZE,
                            TF_OD_API_IS_QUANTIZED,
                            this);
            openCVDetector = OpencvObjectDetection.create(this);

        } catch (final IOException e) {
            e.printStackTrace();
            LOGGER.e(e, "Exception initializing classifier!");
            Toast toast =
                    Toast.makeText(
                            getApplicationContext(), "Classifier could not be initialized", Toast.LENGTH_SHORT);
            toast.show();
            finish();
        }

        previewWidth = size.getWidth();
        previewHeight = size.getHeight();

        sensorOrientation = rotation - getScreenOrientation();
        LOGGER.i("Camera orientation relative to screen canvas: %d", sensorOrientation);

        LOGGER.i("Initializing at size %dx%d", previewWidth, previewHeight);
        rgbFrameBitmap = Bitmap.createBitmap(previewWidth, previewHeight, Config.ARGB_8888);
        rgbVertival = Bitmap.createBitmap(IMG_WIDTH, IMG_HEIGHT, Config.ARGB_8888);
        croppedBitmap = Bitmap.createBitmap(CROP_WIDTH, CROP_HEIGHT, Config.ARGB_8888);

        // crop image
        frameToCropTransform =
                ImageUtils.getTransformationMatrix(
                        previewWidth, previewHeight,
                        CROP_WIDTH, CROP_HEIGHT,
                        sensorOrientation, MAINTAIN_ASPECT);

        // rotate image in landscape
        frameRotation = ImageUtils.getTransformationMatrix(
                previewWidth, previewHeight,
                IMG_WIDTH, IMG_HEIGHT,
                sensorOrientation, MAINTAIN_ASPECT);



        cropToFrameTransform = new Matrix();
        frameToCropTransform.invert(cropToFrameTransform);

        trackingOverlay = (OverlayView) findViewById(R.id.tracking_overlay);
        trackingOverlay.addCallback(
                new DrawCallback() {
                    @Override
                    public void drawCallback(final Canvas canvas) {
                        tracker.draw(canvas);
                        if (isDebug()) {
                            tracker.drawDebug(canvas);
                        }
                    }
                });

        tracker.setFrameConfiguration(previewWidth, previewHeight, sensorOrientation);
    }

    @Override
    // get bitmap an perform inference
    protected void processImage(String detectionMode) {
        ++timestamp;
        final long currTimestamp = timestamp;
        trackingOverlay.postInvalidate();

        // No mutex needed as this method is not reentrant.
        if (computingDetection) {
            readyForNextImage();
            return;
        }
        computingDetection = true;
        LOGGER.i("Preparing image " + currTimestamp + " for detection in bg thread.");

        rgbFrameBitmap.setPixels(getRgbBytes(), 0, previewWidth, 0, 0, previewWidth, previewHeight);

        readyForNextImage();

        final Canvas canvas = new Canvas(croppedBitmap);
        canvas.drawBitmap(rgbFrameBitmap, frameToCropTransform, null);
        final Canvas rotCanvas = new Canvas(rgbVertival);
        rotCanvas.drawBitmap(rgbFrameBitmap, frameRotation, null);


        // For examining the actual TF input.
        if (SAVE_PREVIEW_BITMAP) {
            ImageUtils.saveBitmap(croppedBitmap);
            ImageUtils.saveBitmap(rgbVertival);
        }

        runInBackground(
                new Runnable() {
                    @Override
                    public void run() {
                        LOGGER.i("Running detection on image " + currTimestamp);
                        final long startTime = SystemClock.uptimeMillis();
                        List<DetectorClassifier.Recognition> results = null;
                        if(detectionMode == "tflite") {
                            results = tfliteDetector.recognizeImage(croppedBitmap, rgbVertival);
                        } else if(detectionMode == "opencv"){
                            results = openCVDetector.recognizeImage(croppedBitmap,rgbVertival);
                            computingDetection = false;
                        }
                        lastProcessingTimeMs = SystemClock.uptimeMillis() - startTime;

                        // TODO togliere if dopo implementazione di opencv
                        if(results != null) {
                            cropCopyBitmap = Bitmap.createBitmap(croppedBitmap);
                            final Canvas canvas = new Canvas(cropCopyBitmap);
                            final Paint paint = new Paint();
                            paint.setColor(Color.RED);
                            paint.setStyle(Style.STROKE);
                            paint.setStrokeWidth(2.0f);

                            float minimumConfidence = MINIMUM_CONFIDENCE_TF_OD_API;
                            switch (MODE) {
                                case TF_OD_API:
                                    minimumConfidence = MINIMUM_CONFIDENCE_TF_OD_API;
                                    break;
                            }

                            final List<DetectorClassifier.Recognition> mappedRecognitions =
                                    new LinkedList<DetectorClassifier.Recognition>();

                            for (final DetectorClassifier.Recognition result : results) {
                                final RectF location = result.getLocation();
                                if (location != null && result.getConfidence() >= minimumConfidence) {
                                    canvas.drawRect(location, paint);

                                    cropToFrameTransform.mapRect(location);

                                    result.setLocation(location);
                                    mappedRecognitions.add(result);
                                }
                            }

                            tracker.trackResults(mappedRecognitions, currTimestamp);
                            trackingOverlay.postInvalidate();

                            computingDetection = false;

                            runOnUiThread(
                                    new Runnable() {
                                        @Override
                                        public void run() {
                                            showFrameInfo(previewWidth + "x" + previewHeight);
                                            showCropInfo(cropCopyBitmap.getWidth() + "x" + cropCopyBitmap.getHeight());
                                            showInference(lastProcessingTimeMs + "ms");
                                        }
                                    });
                        }
                    }
                });
    }

    @Override
    protected int getLayoutId() {
        return R.layout.tfe_od_camera_connection_fragment_tracking;
    }

    @Override
    protected Size getDesiredPreviewFrameSize() {
        return DESIRED_PREVIEW_SIZE;
    }

    // Which detection model to use: by default uses Tensorflow Object Detection API frozen
    // checkpoints.
    private enum DetectorMode {
        TF_OD_API;
    }

    @Override
    protected void setUseNNAPI(final boolean isChecked) {
        runInBackground(() -> tfliteDetector.setUseNNAPI(isChecked));
        runInBackground(() -> openCVDetector.setUseNNAPI(isChecked));

    }

    @Override
    protected void setNumThreads(final int numThreads) {
        runInBackground(() -> tfliteDetector.setNumThreads(numThreads));
        runInBackground(() -> openCVDetector.setNumThreads(numThreads));
    }
}
