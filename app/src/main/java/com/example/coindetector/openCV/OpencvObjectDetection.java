package com.example.coindetector.openCV;
import android.graphics.Bitmap;
import android.graphics.RectF;
import android.os.Debug;
import android.util.Log;
import android.view.MenuItem;
import android.widget.Toast;

import com.example.coindetector.env.ImageUtils;
import com.example.coindetector.tflite.DetectorClassifier;

import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfFloat;
import org.opencv.core.MatOfInt;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.ListIterator;

public class OpencvObjectDetection implements DetectorClassifier {

    private static final int CROP_WIDTH = 300;
    private static final int CROP_HEIGHT = 300;
    private static final String TAG = "CoinDetector::Activity";
    private CameraBridgeViewBase mOpenCvCameraView;
    private boolean              mIsJavaCamera = true;
    private MenuItem mItemSwitchCamera = null;
    Mat mGray;
    Mat mColor;
    Mat mEqual;
    Mat circles;
    Mat square;
    Classifier classifier;
    long timer;
    List<Coin> coins;
    ListIterator<Coin> coinIterator;
    RectF detection;

    public static DetectorClassifier create(){
        return new OpencvObjectDetection();
    }

    public OpencvObjectDetection(){
        mGray = new Mat();
        mColor = new  Mat();
        mEqual = new  Mat();
        circles = new  Mat();
        //classifier = new Classifier(getApplicationContext());
        timer = System.currentTimeMillis();
        coins = new ArrayList<Coin>();
        coinIterator = coins.listIterator();
    }


    @Override
    public List<Recognition> recognizeImage(Bitmap bitmap, Bitmap fullBitmap) {

        Mat mColor = new Mat(fullBitmap.getHeight(), fullBitmap.getWidth(), CvType.CV_8UC3);
        Mat mGray = new Mat(mColor.rows(), mColor.cols(), CvType.CV_8UC1);
        Utils.bitmapToMat(fullBitmap,mColor);
        Bitmap grayBitmap = Bitmap.createBitmap(mGray.cols(), mGray.rows(), Bitmap.Config.ARGB_8888);
        Bitmap  colorBitmap = Bitmap.createBitmap(mColor.cols(), mColor.rows(), Bitmap.Config.ARGB_8888);

        Imgproc.cvtColor(mColor, mColor, Imgproc.COLOR_BGR2RGB);
        Imgproc.cvtColor(mColor, mGray, Imgproc.COLOR_RGB2GRAY);


        // ----------- HISTOGRAM ----------------
        int mHistSizeNum = 256;
        MatOfInt mHistSize = new MatOfInt(mHistSizeNum);
        Mat hist = new Mat();
        float []mBuff = new float[mHistSizeNum];

        MatOfFloat histogramRanges = new MatOfFloat(0f, 256f);
        Size imgSize = mGray.size();
        Imgproc.calcHist(Arrays.asList(mGray), new MatOfInt(0), new Mat(), hist, mHistSize, histogramRanges);
        hist.get(0,0,mBuff);

        float max = 0;
        int indexMax = 0;
        for(int i=0; i<mBuff.length; i++)
            if(mBuff[i] > max) {
                max = mBuff[i];
                indexMax = i;
            }
        // --------------------------------------

        //Imgproc.threshold(mGray, mGray, indexMax+30, 255, Imgproc.THRESH_BINARY);
        Imgproc.adaptiveThreshold(mGray, mGray, 255, Imgproc.ADAPTIVE_THRESH_MEAN_C, Imgproc.THRESH_BINARY_INV, 51, 20);
        Imgproc.Canny(mGray, mGray, 200, 255);
        Utils.matToBitmap(mGray,grayBitmap);


        // TODO immagine già scalata
        //mColor = classifier.scaleImage(mColor, 50);
        //mGray = classifier.scaleImage(mGray,50); // scale mGray for faster circle detection

        int thickness = 10;
        int minRadius = (int) mGray.cols()/15;
        int maxRadius = (int) mGray.cols()/3;
        double minDist = 3*minRadius;
        double cannyHighThreshold = 300;
        double accumlatorThreshold = 35;

        int BegX, BegY, EndX, EndY, X, Y, Rad;
        long timer = 0;
        Rect roi, bigroi;

        Imgproc.HoughCircles(mGray, circles, Imgproc.CV_HOUGH_GRADIENT, 1, minDist, cannyHighThreshold, accumlatorThreshold, minRadius, maxRadius);
        final ArrayList<Recognition> recognitions = new ArrayList<>(circles.cols());

        for (int i = 0; i < circles.cols(); i++) {
            double[] circle = circles.get(0, i);

            // coordinates of circle in scaled image
            double centerX = circle[0], centerY = circle[1], radius = circle[2];
            double scalBegX = centerX-radius, scalBegY = centerY-radius, scalEndX = scalBegX+(2*radius), scalEndY = scalBegY+(2*radius);
            roi = new Rect((int) scalBegX,(int) scalBegY ,(int) (2*radius),(int) (2*radius));

            // coordinates of circle in original image
//            BegX = (int) scalBegX * mColor.cols() /mGray.cols();
//            BegY = (int) scalBegY * mColor.rows() /mGray.rows();
//            EndX = (int) (scalEndX +1) * mColor.cols() /mGray.cols();
//            EndY = (int) (scalEndY +1) * mColor.rows() /mGray.rows();
            BegX = (int) scalBegX * CROP_WIDTH /mColor.cols();
            BegY = (int) scalBegY * CROP_HEIGHT /mColor.rows();
            EndX = (int) (scalEndX +1) * CROP_WIDTH /mColor.cols();
            EndY = (int) (scalEndY +1) * CROP_HEIGHT /mColor.rows();
            Rad  = (int) ((EndX - BegX)/2);
            X = BegX + Rad;
            Y = BegY + Rad;
            bigroi = new Rect ( (int) BegX, (int) BegY , (int) (2*Rad), (int) (2*Rad) );

            detection = new RectF( BegX, BegY, EndX, EndY);
            Imgproc.circle(mColor, new Point(X, Y), (int) radius , new Scalar(0, 0, 255), thickness);

//            if(coins.size() >= circles.cols()){
//                // one or more coins disappeared from the scene
//                coinIterator = coins.listIterator();
//                while (coinIterator.hasNext()){
//                    Coin coin = coinIterator.next();
//                    if (coin.X > BegX && coin.X < EndX && coin.Y > BegY && coin.Y < EndY ) {
//                        //coin detected before, update center coordinates
//                        coin.X = (int) (coin.X + X)/2;
//                        coin.Y = (int) (coin.Y + Y)/2;
//                        coin.roi = bigroi;
//                        coin.isPresent = true;
//                    }
//                }
//            } else if(coins.size() < circles.cols()) {
//                // new coin detected, add to coins list
//                boolean newCoin = true;
//                coinIterator = coins.listIterator();
//                while (coinIterator.hasNext()){
//                    Coin coin = coinIterator.next();
//                    // check if is a coin already detected due to hough transform errors
//                    if (coin.X > BegX && coin.X < EndX && coin.Y > BegY && coin.Y < EndY ) {
//                        //coin detected before, update center coordinates
//                        coin.X = X;
//                        coin.Y = Y;
//                        coin.roi = bigroi;
//                        coin.isPresent = true;
//                        newCoin = false;
//                    }
//                }
//                if(newCoin) {
//                    // definitely is a new coin
//                    if(checkBorder(bigroi, mColor.rows(), mColor.cols())) {
//                        square = mColor.submat(bigroi);
//                        Coin coin = new Coin(X, Y, Rad, bigroi, "null");
//                        //coin = classifier.getClass2(coin, square);
//                        coins.add(coin);
//                    }
//                }
//            }
//
//            Log.i(TAG, "detected "+coins.size()+" coins");
            recognitions.add(new Recognition("" + i, "title", (float) 1.0, detection ));

        }

//        coinIterator = coins.listIterator();
//        while (coinIterator.hasNext()){
//            Coin coin = coinIterator.next();
//            if (!coin.isPresent || coin.gap <= 0.0) {
//                coinIterator.remove();
//            }
//            if(coin.gap < 100 || coin.times < 3){
//                coin.times++;
//                if(checkBorder(coin.roi, mColor.rows(), mColor.cols())) {
//                    square = mColor.submat(coin.roi);
//                    //coin = classifier.getClass2(coin, square);
//                }
//            }
//            Imgproc.circle(mColor, new Point(coin.X, coin.Y), coin.R, new Scalar(0, 0, 255), thickness);
//            Imgproc.putText(mColor, coin.classe, new org.opencv.core.Point(coin.X - coin.R, coin.Y - coin.R), Core.FONT_HERSHEY_PLAIN, 5, new Scalar(255, 0, 0), thickness);
//            coin.isPresent = false; // we check if the coin is still present in the next iteration
//        }



        /*coinIterator = coins.listIterator();
        while (coinIterator.hasNext()){
            coinIterator.next().isPresent = false;
        }*/

        Utils.matToBitmap(mColor, colorBitmap);
        return recognitions;
    }

    @Override
    public void enableStatLogging(boolean debug) {

    }

    @Override
    public String getStatString() {
        return null;
    }

    @Override
    public void close() {

    }

    @Override
    public void setNumThreads(int num_threads) {

    }

    @Override
    public void setUseNNAPI(boolean isChecked) {

    }

    public boolean checkBorder(Rect roi, int rows, int cols){
        if(roi.x > 0 && roi.y > 0 && roi.x+roi.width < cols && roi.y+roi.height < rows)
            return true;
        else
            return false;
    }
}

class Coin{
    int X;
    int Y;
    int R;
    Rect roi;
    String classe;
    boolean isPresent;
    long timer;
    int times;
    float gap;

    Coin(int X, int Y, int R, Rect roi, String classe){
        this.X = X;
        this.Y = Y;
        this.R = R;
        this.roi = roi;
        this.classe = classe;
        this.isPresent = true;
        this.timer = System.currentTimeMillis();
        this.gap = 10000;
        this.times = 0;
    }
}
