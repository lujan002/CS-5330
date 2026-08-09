// Luke Jansen
// Shared ImageNet NCHW preprocess for card ONNX models.

#include "card_net_input.hpp"

#include <opencv2/imgproc.hpp>

namespace card_net {

namespace {

const float kPixelMean[3] = {0.485f, 0.456f, 0.406f};  // RGB
const float kPixelStd[3] = {0.229f, 0.224f, 0.225f};

}  // namespace

void preprocessBgrToNchw(const cv::Mat& rectified_bgr,
                         cv::Mat& scratch_resized,
                         cv::Mat& scratch_rgb,
                         std::vector<float>& out) {
    out.resize(static_cast<size_t>(3) * kInputH * kInputW);

    if (rectified_bgr.cols != kInputW || rectified_bgr.rows != kInputH) {
        const int interp =
            (rectified_bgr.cols > kInputW) ? cv::INTER_AREA : cv::INTER_LINEAR;
        cv::resize(rectified_bgr, scratch_resized, cv::Size(kInputW, kInputH), 0, 0,
                   interp);
    } else {
        scratch_resized = rectified_bgr;
    }
    cv::cvtColor(scratch_resized, scratch_rgb, cv::COLOR_BGR2RGB);

    float* data = out.data();
    const int plane = kInputH * kInputW;
    for (int y = 0; y < kInputH; y++) {
        const uchar* row = scratch_rgb.ptr<uchar>(y);
        for (int x = 0; x < kInputW; x++) {
            const int index = y * kInputW + x;
            for (int c = 0; c < 3; c++) {
                data[c * plane + index] =
                    (row[x * 3 + c] / 255.0f - kPixelMean[c]) / kPixelStd[c];
            }
        }
    }
}

}  // namespace card_net
