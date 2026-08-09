// Luke Jansen
// Card identity lookup: rectified card crop -> nearest card in the TCG gallery.
//
// The ONNX graph and gallery.bin are produced by the offline card_match package
// into data/card_match/inference/; see project_final/card_match/README.md.
// ONNX Runtime is hidden behind a pimpl
// so main.cpp does not need its headers.

#ifndef CARD_MATCHER_HPP
#define CARD_MATCHER_HPP

#include <opencv2/core.hpp>

#include <memory>
#include <string>
#include <vector>

struct CardMatch {
    std::string card_id;   // TCGdex id, e.g. "swsh3-136"
    std::string name;      // "Furret"
    std::string set_id;    // "swsh3"
    std::string set_name;  // "Darkness Ablaze"
    int dex = -1;          // national dex number, -1 for Trainer/Energy cards
    float score = 0.0f;    // cosine similarity in [-1, 1]
};

class CardMatcher {
public:
    CardMatcher();
    ~CardMatcher();
    CardMatcher(const CardMatcher&) = delete;
    CardMatcher& operator=(const CardMatcher&) = delete;

    // Loads the embedder and the gallery. On failure returns false and fills error.
    bool init(const std::string& onnx_path,
              const std::string& gallery_path,
              std::string& error);
    bool ready() const;

    // Size the rectified crop should be handed over at; matches the training input.
    int inputWidth() const;
    int inputHeight() const;
    size_t gallerySize() const;

    // rectified_bgr must be a top-down view of one card. Results are sorted by
    // descending score; out is empty if the matcher is not ready.
    bool match(const cv::Mat& rectified_bgr, int top_k, std::vector<CardMatch>& out) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// Warp a card quad (ordered TL, TR, BR, BL) onto an upright width x height
// rectangle. This is the live equivalent of the official top-down scan, and the
// only place the single-plane homography assumption is made: pixels outside the
// card face warp to nonsense, so the quad has to be the card outline.
cv::Mat rectifyCard(const cv::Mat& bgr,
                    const std::vector<cv::Point2f>& corners,
                    int width,
                    int height);

#endif  // CARD_MATCHER_HPP
