// Luke Jansen
// Shared card-network input contract (mirrors card_match/common.py).
// Used by CardMatcher and OrientationClassifier — keep in sync with Python.

#ifndef CARD_NET_INPUT_HPP
#define CARD_NET_INPUT_HPP

#include <opencv2/core.hpp>

#include <vector>

namespace card_net {

constexpr int kInputW = 224;
constexpr int kInputH = 312;

// Resize BGR crop to kInputW x kInputH if needed, convert to RGB, ImageNet
// normalize into NCHW float32. out must hold at least 3 * kInputH * kInputW floats.
// Uses scratch mats to avoid per-frame allocation when callers reuse them.
void preprocessBgrToNchw(const cv::Mat& rectified_bgr,
                         cv::Mat& scratch_resized,
                         cv::Mat& scratch_rgb,
                         std::vector<float>& out);

}  // namespace card_net

#endif  // CARD_NET_INPUT_HPP
