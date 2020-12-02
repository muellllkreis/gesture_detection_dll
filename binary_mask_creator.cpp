#include "binary_mask_creator.h"

using namespace cv;
using namespace std;

binary_mask_creator::binary_mask_creator() {
}

Mat binary_mask_creator::computeBinaryMask(vector<float> range, Mat I_HSV, int lowThreshold, int highThreshold)
{
	Mat binary_mask;
	inRange(I_HSV, Scalar(range[0] - lowThreshold, range[1] - lowThreshold * 0.01f, 0), Scalar(range[2] + highThreshold, range[3] + highThreshold * 0.01f, 1), binary_mask);

	// Blur binary image to smoothen it
	Mat binary_blur(I_HSV.size(), CV_8UC1);
	Mat binary_blur_uc;	
	medianBlur(binary_mask, binary_blur, 5);
	binary_blur.convertTo(binary_blur_uc, CV_8UC1);	

	return binary_blur_uc;
}

// returns vector of form vector<float> = [min_h, min_s, max_h, max_s]
std::vector<float> binary_mask_creator::computeAverageHSV(vector<Hand_ROI> roi) {
	// create binary threshold image per ROI
	float min_h = std::numeric_limits<float>::infinity();
	float max_h = 0.f;
	float min_s = std::numeric_limits<float>::infinity();
	float max_s = 0.f;
	for (Hand_ROI r : roi) {
		if (r.roi_mean[0] < min_h) {
			min_h = r.roi_mean[0];
		}
		if (r.roi_mean[0] > max_h) {
			max_h = r.roi_mean[0];
		}
		if (r.roi_mean[1] < min_s) {
			min_s = r.roi_mean[1];
		}
		if (r.roi_mean[1] > max_s) {
			max_s = r.roi_mean[1];
		}
	}

	cout << "min h: " << min_h << endl;
	cout << "max h: " << max_h << endl;
	cout << "min s: " << min_s << endl;
	cout << "max s: " << max_s << endl;

	vector<float> range;
	range.push_back(min_h);
	range.push_back(min_s);
	range.push_back(max_h);
	range.push_back(max_s);
	return range;
}

Mat binary_mask_creator::removeFacesFromMask(Mat frame, Mat binary_mask, CascadeClassifier faceCascadeClassifier, CascadeClassifier profileCascadeClassifier)
{
	Mat frameGray;
	cvtColor(frame, frameGray, CV_BGR2GRAY);
	cout << "type: " << frameGray.type() << endl;
	equalizeHist(frameGray, frameGray);

	vector<Rect> faces;
	vector<Rect> profile;

	faceCascadeClassifier.detectMultiScale(frameGray, faces, 1.1, 2, 0 | CASCADE_SCALE_IMAGE, Size(120, 120));
	profileCascadeClassifier.detectMultiScale(frameGray, profile, 1.1, 2, 0 | CASCADE_SCALE_IMAGE, Size(120, 120));

	faces.insert(faces.end(), profile.begin(), profile.end());

	for (Rect f : faces) {
		cout << "found face" << endl;
		// note that we are removing the face from binary_blur_uc as it is the Mat used in the next step for contour detection
		rectangle(binary_mask, f, Scalar(0, 0, 0), -1);
	}
	return binary_mask;
}

vector<Hand_ROI> binary_mask_creator::createMaskOverlay(Mat I_BGR, Mat I_HSV) {
	vector<Hand_ROI> roi;

	//flip(I1, I1, 1);
	double col_offset = I_BGR.cols * 0.05;
	double row_offset = I_BGR.rows * 0.05;

	Point hand_center = Point(3 * (int)(I_BGR.cols / 4), (int)(I_BGR.rows / 2));

	roi.push_back(Hand_ROI(Point(hand_center.x + col_offset, hand_center.y - row_offset), I_HSV));
	roi.push_back(Hand_ROI(Point(hand_center.x - col_offset, hand_center.y + row_offset), I_HSV));
	roi.push_back(Hand_ROI(Point(hand_center.x + col_offset, hand_center.y + row_offset), I_HSV));
	roi.push_back(Hand_ROI(Point(hand_center.x - col_offset, hand_center.y - row_offset), I_HSV));

	for (Hand_ROI r : roi) {
		r.draw_rectangle(I_BGR);
	}

	return roi;
}

Mat binary_mask_creator::removeBackGround(Mat input)
{
	//get foreground
	Mat foregroundMask;
	if (!calibrated)
	{
		return input;
	}
	else
	{
		//remove background
		cvtColor(input, foregroundMask, CV_BGR2GRAY);
		for (int i = 0; i < foregroundMask.rows; i++)
		{
			for (int j = 0; j < foregroundMask.cols; j++)
			{
				uchar framePixel = foregroundMask.at<uchar>(i, j);
				uchar backgroundPixel = backgroundReference.at<uchar>(i, j);
				if (framePixel >= backgroundPixel - backGroundThresholdOffset && framePixel <= backgroundPixel + backGroundThresholdOffset)
				{
					foregroundMask.at<uchar>(i, j) = 0;
				}
				else
				{
					foregroundMask.at<uchar>(i, j) = 255;
				}
			}
		}
		//return foreground mask
		Mat foreground;
		input.copyTo(foreground, foregroundMask);
		return foreground;
	}
}

void binary_mask_creator::calibrateBackground(Mat inputFrame)
{
	//calibrate background for future background removal
	cvtColor(inputFrame, backgroundReference, CV_BGR2GRAY);
	calibrated = true;
}