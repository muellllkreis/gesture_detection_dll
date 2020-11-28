// just for debugging of dll
#include <windows.h>
#include <cstdio>

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
#include "binary_mask_creator.h"
#include "gesture_detector.h"
#define _USE_MATH_DEFINES
#include <math.h>
#include <limits>

using namespace cv;
using namespace std;

VideoCapture cap;
vector<Hand_ROI> overlay(4, Hand_ROI());
vector<float> range(4, 0.f);
Mat I1, I_BGR, I_HSV, binary_mask, mask_nf;
binary_mask_creator BMC;
gesture_detector GD;
Point handCenter;

CascadeClassifier faceCascadeClassifier, profileCascadeClassifier;

// holds min and max thresholds for binary mask
struct Thresholds
{
    Thresholds(int low_threshold, int high_threshold) : lowThreshold(low_threshold), highThreshold(high_threshold) {}
    int lowThreshold, highThreshold;
};

// holds results of range calculation of HSV input image
struct HSVRange
{
    HSVRange(float min_h, float min_s, float max_h, float max_s) : minH(min_h), minS(min_s), maxH(max_h), maxS(max_s) {}
    float minH, minS, maxH, maxS;
};

// can hold a pixel position on the feed
struct Position
{
    Position(int x, int y) : X(x), Y(y) {}
    int X, Y;
};

// opens VideoCapture feed (call only once in Awake()!) TODO: pass cam parameter from unity (cap.open(camNumber) instead of hardcoded)
extern "C" int __declspec(dllexport) __stdcall openCam(int& outCameraWidth, int& outCameraHeight)
{
    cap.open(1);
    if (!cap.isOpened()) {
        return -1;
    }

    handCenter = Point(0, 0);

    // Unity (xmls have to be in root of working dir):
    String faceClassifierFileName = "haarcascade_frontalface_alt.xml";
    String profileClassifierFileName = "haarcascade_profileface.xml";

    if (!faceCascadeClassifier.load(faceClassifierFileName))
        throw runtime_error("can't load file " + faceClassifierFileName);

    if (!profileCascadeClassifier.load(profileClassifierFileName))
        throw runtime_error("can't load file " + profileClassifierFileName);

    // this is needed to get access to cout from the dll. allocates a console and prints cout to it
    AllocConsole();
    FILE* fDummy;
    freopen_s(&fDummy, "CONIN$", "r", stdin);
    freopen_s(&fDummy, "CONOUT$", "w", stderr);
    freopen_s(&fDummy, "CONOUT$", "w", stdout);
    std::cout << "Opened" << std::endl;

    outCameraHeight = cap.get(CAP_PROP_FRAME_HEIGHT);
    outCameraWidth = cap.get(CAP_PROP_FRAME_WIDTH);
    return 0;
}

// closes cam, should be called in onApplicationExit()
extern "C" void __declspec(dllexport) __stdcall closeCam() {
    cap.release();
}

// provides unaltered webcam feed, can be called periodically in Update()
extern "C" void __declspec(dllexport) __stdcall capture() {
    Mat frame;
    cap >> frame;

    if (frame.empty()) {
        return;
    }

    imshow("image", frame);
}

// provides webcam feed with overlay of region of interest for getting the binary mask, can be called periodically in Update()
extern "C" void __declspec(dllexport) __stdcall showOverlayFeed() {
    Mat frame;
    cap >> frame;

    if (frame.empty()) {
        return;
    }

    I1 = frame;

    // create bgr and hsv version of image
    frame.convertTo(I_BGR, CV_32F, 1.0 / 255.0, 0.);
    cvtColor(I_BGR, I_HSV, COLOR_BGR2HSV);
    overlay = BMC.createMaskOverlay(I_BGR, I_HSV);

    imshow("feed", I_BGR);
}

// computes new average HSV values, updates range variable. is called upon keypress in Unity
extern "C" HSVRange __declspec(dllexport) __stdcall getMaskRange() {
    range = BMC.computeAverageHSV(overlay);
    return HSVRange(range[0], range[1], range[2], range[3]);
}

// shows current binary mask using passed range and thresholds from Unity, can be called periodically in Update()
extern "C" void __declspec(dllexport) __stdcall showBinaryFeed(HSVRange hsv_range, Thresholds thresholds) {
    vector<float> range;
    range.push_back(hsv_range.minH);
    range.push_back(hsv_range.minS);
    range.push_back(hsv_range.maxH);
    range.push_back(hsv_range.maxS);
    binary_mask = BMC.computeBinaryMask(range, I_HSV, thresholds.lowThreshold, thresholds.highThreshold);
    mask_nf = BMC.removeFacesFromMask(I1, binary_mask, faceCascadeClassifier, profileCascadeClassifier);
    imshow("mask", binary_mask);
}

// provides webcam feed with hand contour overlay. should be called after correct binary mask has been established, can be called periodically in Update()
// IMPORTANT: either call this function or showOverlayFeed() but not both at the same time (since they update the same window)
extern "C" void __declspec(dllexport) __stdcall drawHandContour() {
    Mat frame;
    cap >> frame;

    if (frame.empty()) {
        handCenter = Point(0, 0);
    }

    I1 = frame;

    // create bgr and hsv version of image
    frame.convertTo(I_BGR, CV_32F, 1.0 / 255.0, 0.);
    cvtColor(I_BGR, I_HSV, COLOR_BGR2HSV);

    vector<Point> handContour;
    GD.getHandContour(mask_nf, handContour);
    if (!handContour.empty()) {
        handCenter = GD.drawHull(handContour, I_BGR);
    }
    else {
        handCenter = Point(0, 0);
    }
    imshow("feed", I_BGR);
}

// returns the current center of the identified hand's bounding box. can be called periodically when drawHandContour() is also called periodically (drawHandContour updates the handCenter)
extern "C" void __declspec(dllexport) __stdcall getHandCenter(Position& handPos) {
    handPos = Position(handCenter.x, handCenter.y);
}

// main has to be commented out for .dll creation. can be used for testing with a console application, though.

//int main(int argc, char* argv[])
//{
//    cap.open(1);
//    if (!cap.isOpened()) {
//        return -1;
//    }
//    Mat frame;
//    cap >> frame;
//
//    //Create binary Mask (+ remove face)
//    binary_mask_creator BMC;
//
//    Mat I_BGR;
//    Mat I_HSV;
//
//    // create bgr and hsv version of image
//    frame.convertTo(I_BGR, CV_32F, 1.0 / 255.0, 0.);
//    cvtColor(I_BGR, I_HSV, COLOR_BGR2HSV);
//
//    overlay = BMC.createMaskOverlay(I_BGR, I_HSV);
//    range = BMC.computeAverageHSV(overlay);
//
//    Mat binary_mask = BMC.computeBinaryMask(range, I_HSV, 0, 30);
//
//    //String faceClassifierFileName = "face_classifier/haarcascade_frontalface_alt.xml";
//    //String profileClassifierFileName = "face_classifier/haarcascade_profileface.xml";
//
//    //if (!faceCascadeClassifier.load(faceClassifierFileName)) {
//    //throw runtime_error("can't load file " + faceClassifierFileName);
//    //}
//
//    //if (!profileCascadeClassifier.load(profileClassifierFileName)) {
//    //throw runtime_error("can't load file " + profileClassifierFileName);
//    //}
//
//    imshow("image", I_BGR);
//    imshow("mask", binary_mask);
//    waitKey(0);
//}