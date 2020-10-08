package com.example.coindetector.openCV;

import android.content.Context;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Environment;
import android.util.Log;

import org.opencv.android.Utils;
import org.opencv.core.CvException;
import org.opencv.core.DMatch;
import org.opencv.core.Mat;
import org.opencv.core.MatOfDMatch;
import org.opencv.core.MatOfKeyPoint;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Size;
import org.opencv.features2d.DescriptorExtractor;
import org.opencv.features2d.DescriptorMatcher;
import org.opencv.features2d.FeatureDetector;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.lang.reflect.Array;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;

public class Classifier {

    protected Map<String, Mat> data, dataMask;
    protected Map<String, Float> avgs;
    protected  FeatureDetector orb;
    protected DescriptorExtractor extractor;

    public Classifier(Context context) {

        Mat tmp = new Mat();
        Mat img001 = new Mat();
        Mat img002 = new Mat();
        Mat img005 = new Mat();
        Mat img010 = new Mat();
        Mat img020 = new Mat();
        Mat img050 = new Mat();
        Mat img1 = new Mat();
        Mat img2 = new Mat();
        data = new HashMap<>();
        dataMask = new HashMap<>();
        avgs = new HashMap<>();
        Bitmap btm, btmM;
        BitmapFactory.Options opt = new BitmapFactory.Options();
        opt.inPreferredConfig = Bitmap.Config.ALPHA_8;

        List<String> coins = Arrays.asList("0.01", "0.02", "0.05", "0.10", "0.20", "0.50", "1", "2");
        orb = FeatureDetector.create(FeatureDetector.ORB);
        extractor = DescriptorExtractor.create(DescriptorExtractor.ORB);
        try {

            //extract descriptors from images
            /*for(String coin : coins) {
                MatOfKeyPoint Trainkps = new MatOfKeyPoint();
                Mat Traindes = new Mat();
                String[] getImages = context.getAssets().list(coin);
                for (String imgName : getImages) {
                    InputStream is = context.getAssets().open(coin+"/"+imgName);
                    btm = BitmapFactory.decodeStream(is);
                    Utils.bitmapToMat(btm, img001);
                    if(img001.cols() > 300 && img001.rows() > 300 )
                        img001 = this.getMask(img001);
                    orb.detect(img001,Trainkps);
                    Mat Newdes = new Mat();
                    extractor.compute(img001, Trainkps, Newdes);
                    Traindes.push_back(Newdes);
                }
                data.put(coin,Traindes);
                File path = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DCIM + "/Camera");
                String filename = coin + "AndroidTrained.png";
                File file = new File(path, filename);
                filename = file.toString();
                final String finalFilename = filename;
                //Imgproc.cvtColor(Traindes,Traindes, Imgproc.COLOR_BGR2GRAY);
                Imgcodecs.imwrite(finalFilename,Traindes);
            }*/
            //get descriptors already extracted
            /*for(String coin : coins) {
                InputStream is = context.getAssets().open(coin+"trainingSamples.png");
                btm = BitmapFactory.decodeStream(is);
                Utils.bitmapToMat(btm, tmp);
                Imgproc.cvtColor(tmp, tmp, Imgproc.COLOR_RGB2GRAY);
                data.put(coin,tmp);
                tmp = new Mat();
                InputStream isM = context.getAssets().open(coin+"trainingSamplesMask.png");
                btmM = BitmapFactory.decodeStream(isM);
                Utils.bitmapToMat(btmM, tmp);
                Imgproc.cvtColor(tmp, tmp, Imgproc.COLOR_RGB2GRAY);
                dataMask.put(coin,tmp);
            }*/

            //get descriptors already extracted
            for(String coin : coins) {
                tmp = new Mat();
                InputStream is = context.getAssets().open(coin+"trainingSamples.png");
                btm = BitmapFactory.decodeStream(is);
                Utils.bitmapToMat(btm, tmp);
                Imgproc.cvtColor(tmp, tmp, Imgproc.COLOR_RGB2GRAY);
                data.put(coin,tmp);
            }

        } catch (IOException e) {
            e.printStackTrace();
        }


    }

    public  List<Map.Entry<String,Float>> BFmatcher(Mat img) throws IllegalArgumentException{

        if(img.cols()>0 && img.rows()>0) {
            Bitmap bmp1 = Bitmap.createBitmap(img.cols(), img.rows(), Bitmap.Config.ARGB_8888);
            Utils.matToBitmap(img, bmp1);

            MatOfKeyPoint kps = new MatOfKeyPoint();
            Mat des = new Mat();
            List<DMatch> matches;

            orb.detect(img, kps);
            extractor.compute(img, kps, des);
            for (String k : data.keySet()) {
                try {
                    matches = findMatches(des, (Mat) data.get(k));
                    avgs.put(k, getAverage(matches));
                } catch (CvException e) {
                    e.printStackTrace();
                    Log.e("Memory exceptions", "exceptions" + e);
                }
            }
            List<Map.Entry<String, Float>> listAvgs = new LinkedList<Map.Entry<String, Float>>(avgs.entrySet());
            Collections.sort(listAvgs, new Comparator<Map.Entry<String, Float>>() {
                @Override
                public int compare(Map.Entry<String, Float> o1, Map.Entry<String, Float> o2) {
                    return Float.compare(o1.getValue(), o2.getValue());
                }
            });
            return listAvgs;
        } else{
            return  null;
        }
    }


    public String getClass(Mat img) {


        Bitmap bmp1 = Bitmap.createBitmap(img.cols(), img.rows(), Bitmap.Config.ARGB_8888);
        Utils.matToBitmap(img, bmp1);

        List<Map.Entry<String,Float>> listAvgs = new LinkedList<Map.Entry<String,Float>>(avgs.entrySet());
        Mat mask = new Mat();
        if(img.cols() > 300 && img.rows() > 300) {
            mask = this.getMask(img);
            listAvgs = BFmatcher(mask); // use the mask for classification improve performance but decrease accuracy
        } else {
            //if (listAvgs.get(1).getValue() - listAvgs.get(0).getValue() < 100){
            // the classification is uncertain, repeat on the full coin
            listAvgs = BFmatcher(img);
            //}
        }
        return listAvgs.get(0).getKey();

        /*Bitmap bmp1 = Bitmap.createBitmap(img.cols(), img.rows(), Bitmap.Config.ARGB_8888);
        Utils.matToBitmap(img, bmp1);

        MatOfKeyPoint kps = new MatOfKeyPoint();
        Mat des = new Mat();
        List<DMatch> matches;

        orb.detect(img,kps);
        extractor.compute(img, kps, des);
        for(String k : data.keySet()){
            try {
                matches = findMatches(des, (Mat) data.get(k));
                avgs.put(k, getAverage(matches));
            } catch (CvException e){
                e.printStackTrace();
                Log.e("Memory exceptions","exceptions"+e);
            }
        }
        List<Map.Entry<String,Float>> listAvgs = new LinkedList<Map.Entry<String,Float>>(avgs.entrySet());
        Collections.sort(listAvgs, new Comparator<Map.Entry<String, Float>>() {
            @Override
            public int compare(Map.Entry<String, Float> o1, Map.Entry<String, Float> o2) {
                return Float.compare(o1.getValue(), o2.getValue());
            }
        });
        return listAvgs.get(0).getKey();*/

    }

    public Coin getClass2(Coin coin, Mat img) {

        Bitmap bmp1 = Bitmap.createBitmap(img.cols(), img.rows(), Bitmap.Config.ARGB_8888);
        Utils.matToBitmap(img, bmp1);

        List<Map.Entry<String,Float>> listAvgs = new LinkedList<Map.Entry<String,Float>>(avgs.entrySet());
        Mat mask = new Mat();
        if(img.cols() > 300 && img.rows() > 300) {
            mask = this.getMask(img);
            listAvgs = BFmatcher(mask); // use the mask for classification improve performance but decrease accuracy
        } else {
            //if (listAvgs.get(1).getValue() - listAvgs.get(0).getValue() < 100){
            // the classification is uncertain, repeat on the full coin
            listAvgs = BFmatcher(img);
            //}
        }
        float gap = listAvgs.get(1).getValue() - listAvgs.get(0).getValue();
        if(gap != 0.0 && gap > 20) {
            coin.gap = gap;
            coin.classe = listAvgs.get(0).getKey();
        }else  if(gap == 0){
            coin.gap = -1;
            coin.classe = "";
        }else {
            coin.gap = gap;
            coin.classe = "";
        }
        return coin;
    }

    private List<DMatch> findMatches(Mat des1, Mat des2){
        DescriptorMatcher matcher = DescriptorMatcher.create(DescriptorMatcher.BRUTEFORCE_L1);
        MatOfDMatch matches = new MatOfDMatch();
        matcher.match(des1, des2, matches);

        List<DMatch> allMatches=matches.toList();

        // Ordino i match per distanza
        Collections.sort(allMatches, new Comparator<DMatch>() {
            @Override
            public int compare(DMatch match2, DMatch match1)
            {
                return (int) (match2.distance - match1.distance);
            }
        });

        return allMatches;
    }

    private float getAverage(List<DMatch> matches) throws IndexOutOfBoundsException {
        float avg = 0;
        /*for(DMatch i : matches){
            avg += i.distance;
        }*/
        for(int i=0; i<10; i++){
            try{
                avg += matches.get(i).distance;
            }catch(IndexOutOfBoundsException e){
                e.printStackTrace();
            }
        }
        return avg/10;
    }

    public Mat scaleImage(Mat img, int percent){

        int width = img.cols() * percent / 100;
        int height = img.rows() * percent / 100;
        Size dim = new Size(width, height);
        Imgproc.resize(img, img, dim);
        return img;
    }

    public  Mat getMask(Mat square){
        Rect roi = new Rect(new Point( (square.cols()/3) -40 ,(square.rows()/3) -40 ), new Point( (square.cols()*2/3) +40,(square.rows()*2/3) +40 ) );
        Mat baba = new Mat();
        baba = square.submat(roi);
        Bitmap bmp1 = Bitmap.createBitmap(square.cols(), square.rows(), Bitmap.Config.ARGB_8888);
        Utils.matToBitmap(square, bmp1);
        Bitmap bmp2 = Bitmap.createBitmap(baba.cols(), baba.rows(), Bitmap.Config.ARGB_8888);
        Utils.matToBitmap(baba, bmp2);
        return square.submat(roi);
    }
}