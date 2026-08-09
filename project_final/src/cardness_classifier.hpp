// Luke Jansen
// ONNX full-card vs not-a-card classifier on the same 224x312 crop as matching.

#ifndef CARDNESS_CLASSIFIER_HPP
#define CARDNESS_CLASSIFIER_HPP

#include <opencv2/core.hpp>

#include <memory>
#include <string>

class CardnessClassifier {
public:
    CardnessClassifier();
    ~CardnessClassifier();
    CardnessClassifier(const CardnessClassifier&) = delete;
    CardnessClassifier& operator=(const CardnessClassifier&) = delete;

    // Loads cardness.onnx. On failure returns false and fills error.
    bool init(const std::string& onnx_path, std::string& error);
    bool ready() const;

    int inputWidth() const;
    int inputHeight() const;

    // Higher => more likely a full card. logit_card - logit_not_card.
    // Returns a large negative value if the classifier is not ready or inference fails.
    float scoreCard(const cv::Mat& rectified_bgr) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

#endif  // CARDNESS_CLASSIFIER_HPP
