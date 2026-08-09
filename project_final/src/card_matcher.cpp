// Luke Jansen
// ONNX Runtime embedder + cosine nearest neighbour over the card gallery.

#include "card_matcher.hpp"

#include "card_net_input.hpp"

#include <onnxruntime_cxx_api.h>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <numeric>

namespace {

const char kGalleryMagic[8] = {'P', 'K', 'C', 'G', 'A', 'L', '0', '1'};

bool readExact(std::ifstream& in, void* dst, std::streamsize bytes) {
    in.read(static_cast<char*>(dst), bytes);
    return in.gcount() == bytes;
}

bool readInt32(std::ifstream& in, int32_t& out) {
    return readExact(in, &out, sizeof(out));
}

bool readString(std::ifstream& in, std::string& out) {
    int32_t length = 0;
    if (!readInt32(in, length) || length < 0 || length > (1 << 20)) {
        return false;
    }
    out.assign(static_cast<size_t>(length), '\0');
    return length == 0 || readExact(in, out.data(), length);
}

}  // namespace

struct CardMatcher::Impl {
    Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "card_matcher"};
    std::unique_ptr<Ort::Session> session;
    Ort::AllocatorWithDefaultOptions allocator;
    std::string input_name;
    std::string output_name;

    // Gallery embeddings as a count x dim CV_32F matrix so the similarity sweep
    // is one gemm instead of a hand-rolled loop.
    cv::Mat gallery;
    std::vector<CardMatch> entries;

    // Scratch reused every frame; matching runs in the capture loop.
    mutable cv::Mat resized;
    mutable cv::Mat rgb;
    mutable std::vector<float> input_tensor;
    mutable cv::Mat similarity;

    bool loadGallery(const std::string& path, std::string& error);
};

bool CardMatcher::Impl::loadGallery(const std::string& path, std::string& error) {
    std::ifstream in(path, std::ios::binary);
    if (!in.is_open()) {
        error = "cannot open gallery " + path;
        return false;
    }

    char magic[8] = {};
    if (!readExact(in, magic, sizeof(magic)) ||
        std::memcmp(magic, kGalleryMagic, sizeof(magic)) != 0) {
        error = path + " is not a card gallery (bad magic)";
        return false;
    }

    int32_t count = 0;
    int32_t dim = 0;
    if (!readInt32(in, count) || !readInt32(in, dim) || count <= 0 || dim <= 0) {
        error = "gallery header is corrupt";
        return false;
    }

    gallery.create(count, dim, CV_32F);
    if (!readExact(in, gallery.ptr<float>(),
                   static_cast<std::streamsize>(count) * dim * sizeof(float))) {
        error = "gallery is truncated (embeddings)";
        return false;
    }

    entries.resize(static_cast<size_t>(count));
    for (int32_t i = 0; i < count; i++) {
        CardMatch& entry = entries[static_cast<size_t>(i)];
        int32_t dex = -1;
        if (!readInt32(in, dex) || !readString(in, entry.card_id) ||
            !readString(in, entry.name) || !readString(in, entry.set_id) ||
            !readString(in, entry.set_name)) {
            error = "gallery is truncated (metadata)";
            return false;
        }
        entry.dex = dex;
    }
    return true;
}

CardMatcher::CardMatcher() = default;
CardMatcher::~CardMatcher() = default;

bool CardMatcher::init(const std::string& onnx_path,
                       const std::string& gallery_path,
                       std::string& error) {
    error.clear();
    auto impl = std::make_unique<Impl>();

    if (!impl->loadGallery(gallery_path, error)) {
        return false;
    }

    try {
        Ort::SessionOptions options;
        options.SetIntraOpNumThreads(2);
        options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        impl->session =
            std::make_unique<Ort::Session>(impl->env, onnx_path.c_str(), options);

        if (impl->session->GetInputCount() != 1 || impl->session->GetOutputCount() != 1) {
            error = "expected a single-input, single-output embedder";
            return false;
        }
        impl->input_name = impl->session->GetInputNameAllocated(0, impl->allocator).get();
        impl->output_name = impl->session->GetOutputNameAllocated(0, impl->allocator).get();
    } catch (const Ort::Exception& e) {
        error = std::string("onnxruntime: ") + e.what();
        return false;
    }

    impl->input_tensor.resize(static_cast<size_t>(3) * card_net::kInputH * card_net::kInputW);
    impl->similarity.create(impl->gallery.rows, 1, CV_32F);
    impl_ = std::move(impl);

    // A dry run surfaces a dimension mismatch here instead of on the first frame.
    std::vector<CardMatch> probe;
    if (!match(cv::Mat::zeros(card_net::kInputH, card_net::kInputW, CV_8UC3), 1, probe)) {
        error = "embedder failed its warm-up inference";
        impl_.reset();
        return false;
    }
    if (probe.empty()) {
        error = "embedder produced no match on warm-up";
        impl_.reset();
        return false;
    }
    return true;
}

bool CardMatcher::ready() const {
    return impl_ && impl_->session;
}

int CardMatcher::inputWidth() const {
    return card_net::kInputW;
}

int CardMatcher::inputHeight() const {
    return card_net::kInputH;
}

size_t CardMatcher::gallerySize() const {
    return impl_ ? impl_->entries.size() : 0;
}

bool CardMatcher::match(const cv::Mat& rectified_bgr,
                        int top_k,
                        std::vector<CardMatch>& out) const {
    out.clear();
    if (!impl_ || !impl_->session || rectified_bgr.empty()) {
        return false;
    }
    Impl& impl = *impl_;

    // Preprocess exactly as card_match/dataset.py to_tensor() does.
    card_net::preprocessBgrToNchw(rectified_bgr, impl.resized, impl.rgb,
                                  impl.input_tensor);
    float* data = impl.input_tensor.data();

    const float* embedding = nullptr;
    int dim = 0;
    try {
        const std::array<int64_t, 4> shape = {1, 3, card_net::kInputH, card_net::kInputW};
        Ort::MemoryInfo memory =
            Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value input = Ort::Value::CreateTensor<float>(
            memory, data, impl.input_tensor.size(), shape.data(), shape.size());

        const char* input_names[] = {impl.input_name.c_str()};
        const char* output_names[] = {impl.output_name.c_str()};
        auto outputs = impl.session->Run(Ort::RunOptions{nullptr}, input_names, &input, 1,
                                         output_names, 1);
        if (outputs.empty()) {
            return false;
        }
        auto info = outputs[0].GetTensorTypeAndShapeInfo();
        const auto shape_out = info.GetShape();
        if (shape_out.size() != 2 || shape_out[1] != impl.gallery.cols) {
            return false;
        }
        dim = static_cast<int>(shape_out[1]);
        embedding = outputs[0].GetTensorData<float>();

        // The graph L2-normalizes, so a dot product against the (also normalized)
        // gallery is cosine similarity directly.
        const cv::Mat query(1, dim, CV_32F, const_cast<float*>(embedding));
        cv::gemm(impl.gallery, query, 1.0, cv::Mat(), 0.0, impl.similarity, cv::GEMM_2_T);
    } catch (const Ort::Exception&) {
        return false;
    }

    const int count = impl.similarity.rows;
    const int wanted = std::max(1, std::min(top_k, count));
    const float* scores = impl.similarity.ptr<float>();

    std::vector<int> order(static_cast<size_t>(count));
    std::iota(order.begin(), order.end(), 0);
    std::partial_sort(order.begin(), order.begin() + wanted, order.end(),
                      [scores](int a, int b) { return scores[a] > scores[b]; });

    out.reserve(static_cast<size_t>(wanted));
    for (int i = 0; i < wanted; i++) {
        CardMatch entry = impl.entries[static_cast<size_t>(order[i])];
        entry.score = scores[order[i]];
        out.push_back(std::move(entry));
    }
    return true;
}

cv::Mat rectifyCard(const cv::Mat& bgr,
                    const std::vector<cv::Point2f>& corners,
                    int width,
                    int height) {
    if (bgr.empty() || corners.size() != 4 || width <= 0 || height <= 0) {
        return cv::Mat();
    }

    const std::vector<cv::Point2f> dst = {
        cv::Point2f(0.0f, 0.0f),
        cv::Point2f(static_cast<float>(width - 1), 0.0f),
        cv::Point2f(static_cast<float>(width - 1), static_cast<float>(height - 1)),
        cv::Point2f(0.0f, static_cast<float>(height - 1)),
    };
    cv::Mat homography = cv::getPerspectiveTransform(corners, dst);
    if (homography.empty()) {
        return cv::Mat();
    }

    cv::Mat rectified;
    cv::warpPerspective(bgr, rectified, homography, cv::Size(width, height),
                        cv::INTER_LINEAR, cv::BORDER_REPLICATE);
    return rectified;
}
