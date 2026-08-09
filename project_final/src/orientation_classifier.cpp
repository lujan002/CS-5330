// Luke Jansen
// ONNX Runtime binary orientation classifier (upright vs upside-down).

#include "orientation_classifier.hpp"

#include "card_net_input.hpp"

#include <onnxruntime_cxx_api.h>

#include <array>
#include <vector>

struct OrientationClassifier::Impl {
    Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "orientation"};
    std::unique_ptr<Ort::Session> session;
    Ort::AllocatorWithDefaultOptions allocator;
    std::string input_name;
    std::string output_name;

    mutable cv::Mat resized;
    mutable cv::Mat rgb;
    mutable std::vector<float> input_tensor;
};

OrientationClassifier::OrientationClassifier() = default;
OrientationClassifier::~OrientationClassifier() = default;

bool OrientationClassifier::init(const std::string& onnx_path, std::string& error) {
    error.clear();
    auto impl = std::make_unique<Impl>();

    try {
        Ort::SessionOptions options;
        options.SetIntraOpNumThreads(1);
        options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        impl->session =
            std::make_unique<Ort::Session>(impl->env, onnx_path.c_str(), options);

        if (impl->session->GetInputCount() != 1 || impl->session->GetOutputCount() != 1) {
            error = "expected a single-input, single-output orientation model";
            return false;
        }
        impl->input_name = impl->session->GetInputNameAllocated(0, impl->allocator).get();
        impl->output_name =
            impl->session->GetOutputNameAllocated(0, impl->allocator).get();
    } catch (const Ort::Exception& e) {
        error = std::string("onnxruntime: ") + e.what();
        return false;
    }

    impl_ = std::move(impl);

    // Warm-up catches shape mismatches before the capture loop.
    const float probe = scoreUpright(cv::Mat::zeros(card_net::kInputH, card_net::kInputW,
                                                    CV_8UC3));
    if (probe < -1e8f) {
        error = "orientation model failed its warm-up inference";
        impl_.reset();
        return false;
    }
    return true;
}

bool OrientationClassifier::ready() const {
    return impl_ && impl_->session;
}

int OrientationClassifier::inputWidth() const {
    return card_net::kInputW;
}

int OrientationClassifier::inputHeight() const {
    return card_net::kInputH;
}

float OrientationClassifier::scoreUpright(const cv::Mat& rectified_bgr) const {
    if (!impl_ || !impl_->session || rectified_bgr.empty()) {
        return -1e9f;
    }
    Impl& impl = *impl_;

    card_net::preprocessBgrToNchw(rectified_bgr, impl.resized, impl.rgb,
                                  impl.input_tensor);

    try {
        const std::array<int64_t, 4> shape = {1, 3, card_net::kInputH, card_net::kInputW};
        Ort::MemoryInfo memory =
            Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value input = Ort::Value::CreateTensor<float>(
            memory, impl.input_tensor.data(), impl.input_tensor.size(), shape.data(),
            shape.size());

        const char* input_names[] = {impl.input_name.c_str()};
        const char* output_names[] = {impl.output_name.c_str()};
        auto outputs = impl.session->Run(Ort::RunOptions{nullptr}, input_names, &input, 1,
                                         output_names, 1);
        if (outputs.empty()) {
            return -1e9f;
        }
        auto info = outputs[0].GetTensorTypeAndShapeInfo();
        const auto shape_out = info.GetShape();
        if (shape_out.size() != 2 || shape_out[1] != 2) {
            return -1e9f;
        }
        const float* logits = outputs[0].GetTensorData<float>();
        // logits[0]=upright, logits[1]=upside_down
        return logits[0] - logits[1];
    } catch (const Ort::Exception&) {
        return -1e9f;
    }
}
