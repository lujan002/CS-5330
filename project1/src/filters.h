#pragma once

#include <opencv2/opencv.hpp>

int greyscale_filter(const cv::Mat &src, cv::Mat &dst);
int alt_greyscale_filter(const cv::Mat &src, cv::Mat &dst);
int sepia_filter(const cv::Mat &src, cv::Mat &dst);
int blur5x5_1(const cv::Mat &src, cv::Mat &dst);
int blur5x5_2(const cv::Mat &src, cv::Mat &dst);
int sobelX3x3(const cv::Mat &src, cv::Mat &dst );
int sobelY3x3(const cv::Mat &src, cv::Mat &dst );
int magnitude(const cv::Mat &sx, const cv::Mat &sy, cv::Mat &dst );
int convertToUnsigned(const cv::Mat &short_image, cv::Mat &dst);
int blurQuantize(const cv::Mat &src, cv::Mat &dst, int levels );
int depthInvert(const cv::Mat &color, const cv::Mat &depth, cv::Mat &dst);
int halo(const cv::Mat &color, const std::vector<cv::Rect> &faces, const cv::Mat &depth, cv::Mat &dst);
int faceEmbossQuarter(const cv::Mat &color, const std::vector<cv::Rect> &faces, cv::Mat &dst);
int blue(const cv::Mat &src, cv::Mat &dst);
int emboss(const cv::Mat &src, cv::Mat &dst);
int gaborFilter(const cv::Mat &src, cv::Mat &dst, int theta);