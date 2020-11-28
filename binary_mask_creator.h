#ifndef BINARY_MASK_CREATOR_H
#define BINARY_MASK_CREATOR_H

#include <iostream>
#include <sstream>
#include <vector>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgproc/types_c.h>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/video.hpp>
#include "hand_roi.h"

#define _USE_MATH_DEFINES
#include <math.h>
#include <limits>

class binary_mask_creator
{
    public : 
        binary_mask_creator();
        //binary_mask_creator(String classifierPath1, String classifierPath2);
        Mat computeBinaryMask(std::vector<float> range, Mat I, int lowThreshold, int highThreshold);
        std::vector<Hand_ROI> createMaskOverlay(Mat I_BGR, Mat I_HSV);
        Mat removeFacesFromMask(Mat frame, Mat binary_blur_uc, CascadeClassifier faceCascadeClassifier, CascadeClassifier profileCascadeClassifier);
        std::vector<float> computeAverageHSV(std::vector<Hand_ROI> roi);
};

#endif
