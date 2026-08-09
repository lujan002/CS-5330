// Luke Jansen
// ONNX upright / upside-down classifier on the same 224x312 crop as matching.

#ifndef ORIENTATION_CLASSIFIER_HPP
#define ORIENTATION_CLASSIFIER_HPP

#include <opencv2/core.hpp>

#include <memory>
#include <string>

class OrientationClassifier {
public:
    OrientationClassifier();
    ~OrientationClassifier();
    OrientationClassifier(const OrientationClassifier&) = delete;
    OrientationClassifier& operator=(const OrientationClassifier&) = delete;

    // Loads orient.onnx. On failure returns false and fills error.
    bool init(const std::string& onnx_path, std::string& error);
    bool ready() const;

    int inputWidth() const;
    int inputHeight() const;

    // Higher => more likely upright. logit_upright - logit_upside_down.
    // Returns a large negative value if the classifier is not ready or inference fails.
    float scoreUpright(const cv::Mat& rectified_bgr) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

#endif  // ORIENTATION_CLASSIFIER_HPP
