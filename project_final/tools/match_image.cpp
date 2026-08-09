// Luke Jansen
// Run the card matcher over still images instead of a camera.
//
// Two jobs: check that the C++ preprocessing agrees with the Python side (feed
// it a gallery scan and it should retrieve itself at ~1.00), and debug a real
// photo without holding a card in front of the webcam.
//
//   ./match_image data/tcg/en/images/swsh3-136.webp
//   ./match_image photo.jpg --corners 120,80,410,95,400,520,110,505

#include "card_matcher.hpp"

#include <opencv2/imgcodecs.hpp>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static bool parseCorners(const char* text, std::vector<cv::Point2f>& corners) {
    corners.clear();
    std::vector<float> values;
    const char* cursor = text;
    while (*cursor != '\0') {
        char* end = nullptr;
        float value = std::strtof(cursor, &end);
        if (end == cursor) {
            return false;
        }
        values.push_back(value);
        cursor = end;
        while (*cursor == ',' || *cursor == ' ') {
            cursor++;
        }
    }
    if (values.size() != 8) {
        return false;
    }
    for (int i = 0; i < 4; i++) {
        corners.emplace_back(values[i * 2], values[i * 2 + 1]);
    }
    return true;
}

int main(int argc, char* argv[]) {
    const char* usage =
        "Usage: %s IMAGE [--embedder PATH] [--gallery PATH] [--top N]\n"
        "       [--corners TLx,TLy,TRx,TRy,BRx,BRy,BLx,BLy]\n"
        "Without --corners the whole image is treated as an already top-down card.\n";

    std::string image_path;
    std::string embedder = "data/card_match/embedder.onnx";
    std::string gallery = "data/card_match/gallery.bin";
    std::vector<cv::Point2f> corners;
    int top_k = 5;

    for (int i = 1; i < argc; i++) {
        if (std::strcmp(argv[i], "--embedder") == 0 && i + 1 < argc) {
            embedder = argv[++i];
        } else if (std::strcmp(argv[i], "--gallery") == 0 && i + 1 < argc) {
            gallery = argv[++i];
        } else if (std::strcmp(argv[i], "--top") == 0 && i + 1 < argc) {
            top_k = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--corners") == 0 && i + 1 < argc) {
            if (!parseCorners(argv[++i], corners)) {
                printf("--corners needs 8 comma-separated numbers (TL TR BR BL)\n");
                return -1;
            }
        } else if (argv[i][0] == '-') {
            printf(usage, argv[0]);
            return -1;
        } else {
            image_path = argv[i];
        }
    }

    if (image_path.empty()) {
        printf(usage, argv[0]);
        return -1;
    }

    cv::Mat image = cv::imread(image_path, cv::IMREAD_COLOR);
    if (image.empty()) {
        printf("could not read %s\n", image_path.c_str());
        return -1;
    }

    CardMatcher matcher;
    std::string error;
    if (!matcher.init(embedder, gallery, error)) {
        printf("matcher init failed: %s\n", error.c_str());
        return -1;
    }
    printf("gallery: %zu cards, input %dx%d\n", matcher.gallerySize(),
           matcher.inputWidth(), matcher.inputHeight());

    cv::Mat rectified = image;
    if (!corners.empty()) {
        rectified = rectifyCard(image, corners, matcher.inputWidth(),
                                matcher.inputHeight());
        if (rectified.empty()) {
            printf("rectification failed — are the corners degenerate?\n");
            return -1;
        }
    }

    std::vector<CardMatch> results;
    if (!matcher.match(rectified, top_k, results) || results.empty()) {
        printf("no match produced\n");
        return -1;
    }

    printf("\n%s\n", image_path.c_str());
    for (size_t i = 0; i < results.size(); i++) {
        const CardMatch& entry = results[i];
        printf("  %zu. %.4f  %-30s %-14s %s\n", i + 1, entry.score,
               entry.name.c_str(), entry.card_id.c_str(), entry.set_name.c_str());
    }
    return 0;
}
