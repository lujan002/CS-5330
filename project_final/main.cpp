// Luke Jansen
// 7/24/2026
// main script for AR pokemon card-model visualization

#include <opencv2/opencv.hpp>
#include <opencv2/calib3d.hpp>

#include "card_matcher.hpp"
#include "cardness_classifier.hpp"
#include "gl_renderer.hpp"
#include "model_download.hpp"
#include "model_library.hpp"
#include "obj_loader.hpp"
#include "orientation_classifier.hpp"
#include "poke3d_download.hpp"
#include "pokemon_heights.hpp"

#include <GLFW/glfw3.h>
#ifdef __linux__
#include <X11/Xlib.h>
#endif

#include <cstdio>
#include <cstdlib>
#include <cctype>
#include <deque>
#include <fstream>
#include <map>
#include <string>
#include <vector>
#include <algorithm>
#include <cmath>
#include <cstring>

// Place HighGUI windows on the monitor under the mouse (primary as fallback),
// so a multi-monitor desktop doesn't open them on a different screen.
static void placeAppWindows(bool have_card_window) {
    // Card/Match are portrait crops (~224x312), not 640x480 HighGUI windows.
    const int win_w = 640;
    const int win_h = 480;
    const int card_w = 240;
    const int margin = 40;
    const int gap = 20;

    int origin_x = 0;
    int origin_y = 0;
    int mon_w = 1920;
    int mon_h = 1080;

    if (glfwInit()) {
        int px = 0;
        int py = 0;
#ifdef __linux__
        if (Display* dpy = XOpenDisplay(nullptr)) {
            Window root = DefaultRootWindow(dpy);
            Window root_ret = 0;
            Window child = 0;
            int rx = 0;
            int ry = 0;
            int wx = 0;
            int wy = 0;
            unsigned int mask = 0;
            if (XQueryPointer(dpy, root, &root_ret, &child, &rx, &ry, &wx, &wy, &mask)) {
                px = rx;
                py = ry;
            }
            XCloseDisplay(dpy);
        }
#endif

        GLFWmonitor* monitor = glfwGetPrimaryMonitor();
        int count = 0;
        GLFWmonitor** monitors = glfwGetMonitors(&count);
        for (int i = 0; i < count; ++i) {
            int mx = 0;
            int my = 0;
            glfwGetMonitorPos(monitors[i], &mx, &my);
            const GLFWvidmode* mode = glfwGetVideoMode(monitors[i]);
            if (!mode) {
                continue;
            }
            if (px >= mx && px < mx + mode->width &&
                py >= my && py < my + mode->height) {
                monitor = monitors[i];
                break;
            }
        }

        if (monitor) {
            glfwGetMonitorPos(monitor, &origin_x, &origin_y);
            if (const GLFWvidmode* mode = glfwGetVideoMode(monitor)) {
                mon_w = mode->width;
                mon_h = mode->height;
            }
        }
        // Leave GLFW initialized; GLRenderer owns shutdown via glfwTerminate().
    }

    int video_x = origin_x + margin;
    int video_y = origin_y + margin;
    int edges_x = video_x + win_w + gap;
    int edges_y = video_y;
    // Prefer side-by-side; stack vertically if that would leave this monitor.
    if (edges_x + win_w > origin_x + mon_w - margin) {
        edges_x = video_x;
        edges_y = video_y + win_h + gap;
    }
    if (edges_y + win_h > origin_y + mon_h - margin) {
        edges_x = video_x + gap;
        edges_y = video_y + gap;
    }

    cv::moveWindow("Video", video_x, video_y);
    cv::moveWindow("Edges", edges_x, edges_y);
    if (have_card_window) {
        // Live rectified crop, then the official gallery scan of the same card.
        const int card_x = edges_x + win_w + gap;
        cv::moveWindow("Card", card_x, video_y);
        cv::moveWindow("Match", card_x + card_w + gap, video_y);
    }
}

// ---------------------------------------------------------------------------
// TODO (later): Skinned OpenGL renderer (bone palette + idle animation)
//   - Keep Assimp FBX skeleton/weights (do not PreTransformVertices)
//   - Advance idle animation time each frame
// ---------------------------------------------------------------------------

// static bool loadYoloCorners(const char* path, std::vector<cv::Point2f>& corners_out) {
//     corners_out.clear();
//     std::ifstream in(path);
//     if (!in.is_open()) return false;
//     std::string line;
//     if (!std::getline(in, line) || line == "none" || line.empty()) return false;
//     // parse 8 floats → push 4 Point2f
//     // then orderQuadCorners(corners_out);  // optional if Python already ordered
//     return corners_out.size() == 4;
// }

// Order 4 image points as TL, TR, BR, BL (top-left origin of image).
static void orderQuadCorners(std::vector<cv::Point2f>& pts) {
    if (pts.size() != 4) {
        return;
    }

    // sort by y (top pair first), then by x within each pair
    std::sort(pts.begin(), pts.end(),
              [](const cv::Point2f& a, const cv::Point2f& b) {
                  if (std::fabs(a.y - b.y) < 1.0f) {
                      return a.x < b.x;
                  }
                  return a.y < b.y;
              });

    cv::Point2f tl = pts[0].x < pts[1].x ? pts[0] : pts[1];
    cv::Point2f tr = pts[0].x < pts[1].x ? pts[1] : pts[0];
    cv::Point2f bl = pts[2].x < pts[3].x ? pts[2] : pts[3];
    cv::Point2f br = pts[2].x < pts[3].x ? pts[3] : pts[2];

    pts[0] = tl;
    pts[1] = tr;
    pts[2] = br;
    pts[3] = bl;
}

// Recover physical width/height of a planar rectangle from its image quad under
// perspective: H maps unit square -> image corners, then aspect = ||K^{-1} h1|| / ||K^{-1} h2||.
static bool recoverQuadAspect(const std::vector<cv::Point2f>& ordered,
                              const cv::Mat& K,
                              float& aspect_out,
                              bool require_ortho = true) {
    if (ordered.size() != 4 || K.empty() || K.rows != 3 || K.cols != 3) {
        return false;
    }

    std::vector<cv::Point2f> square = {
        cv::Point2f(0.0f, 0.0f),
        cv::Point2f(1.0f, 0.0f),
        cv::Point2f(1.0f, 1.0f),
        cv::Point2f(0.0f, 1.0f),
    };
    cv::Mat H = cv::getPerspectiveTransform(square, ordered);
    if (H.empty()) {
        return false;
    }

    cv::Mat Kinv = K.inv();
    cv::Mat A = Kinv * H;  // [v1 v2 t] up to scale

    cv::Vec3d v1(A.at<double>(0, 0), A.at<double>(1, 0), A.at<double>(2, 0));
    cv::Vec3d v2(A.at<double>(0, 1), A.at<double>(1, 1), A.at<double>(2, 1));
    double n1 = cv::norm(v1);
    double n2 = cv::norm(v2);
    if (n1 < 1e-9 || n2 < 1e-9) {
        return false;
    }

    if (require_ortho) {
        // Loose orthogonality: true rectangle => v1 · v2 ≈ 0
        double cos_ang = std::fabs(v1.dot(v2)) / (n1 * n2);
        if (cos_ang > 0.45) {
            return false;
        }
    }

    aspect_out = static_cast<float>(n1 / n2);
    return true;
}

// Width/height for current TL-TR / TL-BL labeling (prefer metric; fall back to image sides).
static float labelingAspect(const std::vector<cv::Point2f>& ordered,
                            const cv::Mat& K) {
    float aspect = 0.0f;
    if (recoverQuadAspect(ordered, K, aspect, /*require_ortho=*/false)) {
        return aspect;
    }
    float w = cv::norm(ordered[1] - ordered[0]);
    float h = cv::norm(ordered[3] - ordered[0]);
    if (h < 1e-6f) {
        return 1.0f;
    }
    return w / h;
}

static bool quadHasCardAspect(const std::vector<cv::Point2f>& ordered,
                              const cv::Mat& K,
                              float target = 2.5f / 3.5f,
                              float rel_tol = 0.1f) {
    float aspect = 0.0f;
    if (!recoverQuadAspect(ordered, K, aspect)) {
        return false;
    }
    float a = std::min(aspect, 1.0f / aspect);
    float t = std::min(target, 1.0f / target);
    float rel_err = std::fabs(a - t) / t;
    return rel_err <= rel_tol;
}

// Portrait only: short side is top edge (width/height ≈ 2.5/3.5), not landscape.
static bool quadHasPortraitAspect(const std::vector<cv::Point2f>& ordered,
                                  const cv::Mat& K,
                                  float target = 2.5f / 3.5f,
                                  float rel_tol = 0.25f) {
    float aspect = 0.0f;
    if (!recoverQuadAspect(ordered, K, aspect)) {
        return false;
    }
    return std::fabs(aspect - target) / target <= rel_tol;
}

// Shared Canny thresholds (detection + upright scoring).
static const double kCannyLow = 40.0;
static const double kCannyHigh = 80.0;

// Intersect two infinite lines in fitLine form (vx,vy,x0,y0). Returns false if parallel.
static bool intersectFitLines(const cv::Vec4f& a, const cv::Vec4f& b, cv::Point2f& out) {
    const float ax = a[0], ay = a[1], ax0 = a[2], ay0 = a[3];
    const float bx = b[0], by = b[1], bx0 = b[2], by0 = b[3];
    const float denom = ax * by - ay * bx;
    if (std::fabs(denom) < 1e-6f) {
        return false;
    }
    const float t = ((bx0 - ax0) * by - (by0 - ay0) * bx) / denom;
    out = cv::Point2f(ax0 + t * ax, ay0 + t * ay);
    return true;
}

// Refine a card quad by fitting a line to contour/edge pixels along each side,
// then intersecting adjacent lines. Recover corners that approxPolyDP "cuts"
// when approx_eps_frac is large.
static void refineQuadByEdgeLines(const cv::Mat& edges,
                                  std::vector<cv::Point2f>& corners,
                                  const std::vector<cv::Point>* contour = nullptr,
                                  float band_px = 4.0f,
                                  float inset_frac = 0.0f) {
    if (corners.size() != 4 || edges.empty()) {
        return;
    }

    cv::Point2f center(0.0f, 0.0f);
    for (int i = 0; i < 4; i++) {
        center += corners[i];
    }
    center *= 0.25f;

    cv::Vec4f lines[4];
    bool have_line[4] = {false, false, false, false};

    for (int i = 0; i < 4; i++) {
        const cv::Point2f& p0 = corners[i];
        const cv::Point2f& p1 = corners[(i + 1) % 4];
        cv::Point2f dir = p1 - p0;
        float len = std::sqrt(dir.x * dir.x + dir.y * dir.y);
        if (len < 1.0f) {
            continue;
        }
        dir.x /= len;
        dir.y /= len;
        cv::Point2f nrm(-dir.y, dir.x);

        std::vector<cv::Point2f> pts;
        pts.reserve(static_cast<size_t>(len) + 32);

        // Prefer the real contour: includes the white pixels that stick past
        // the cut-corner approxPolyDP vertices.
        if (contour != nullptr && !contour->empty()) {
            const float max_dist = band_px + 2.0f;
            const float extend = 0.2f * len;  // include past the cut corners
            for (const cv::Point& p : *contour) {
                cv::Point2f pf(static_cast<float>(p.x), static_cast<float>(p.y));
                float t = (pf.x - p0.x) * dir.x + (pf.y - p0.y) * dir.y;
                if (t < -extend || t > len + extend) {
                    continue;
                }
                float dist = (pf.x - p0.x) * nrm.x + (pf.y - p0.y) * nrm.y;
                if (std::fabs(dist) <= max_dist) {
                    pts.push_back(pf);
                }
            }
        }

        // Fallback / supplement: sample Canny in a band, extended past ends.
        if (pts.size() < 8) {
            const float t0 = -0.2f;
            const float t1 = 1.2f;
            const int samples = std::max(8, static_cast<int>(len * 1.4f));
            for (int s = 0; s <= samples; s++) {
                float t = t0 + (t1 - t0) * static_cast<float>(s) /
                                   static_cast<float>(samples);
                cv::Point2f base(p0.x + t * (p1.x - p0.x),
                                 p0.y + t * (p1.y - p0.y));
                for (float d = -band_px; d <= band_px + 1e-3f; d += 1.0f) {
                    int x = cvRound(base.x + d * nrm.x);
                    int y = cvRound(base.y + d * nrm.y);
                    if (x < 0 || y < 0 || x >= edges.cols || y >= edges.rows) {
                        continue;
                    }
                    if (edges.at<uchar>(y, x) != 0) {
                        pts.push_back(cv::Point2f(static_cast<float>(x),
                                                  static_cast<float>(y)));
                    }
                }
            }
        }

        if (pts.size() < 8) {
            continue;
        }
        cv::fitLine(pts, lines[i], cv::DIST_HUBER, 0, 0.01, 0.01);
        have_line[i] = true;
    }

    std::vector<cv::Point2f> refined = corners;
    for (int i = 0; i < 4; i++) {
        int prev = (i + 3) % 4;
        int next = i;
        if (!have_line[prev] || !have_line[next]) {
            continue;
        }
        cv::Point2f hit;
        if (intersectFitLines(lines[prev], lines[next], hit)) {
            float dx = hit.x - corners[i].x;
            float dy = hit.y - corners[i].y;
            // Allow larger recovery: high approx_eps can cut corners by >20px.
            if (dx * dx + dy * dy <= 40.0f * 40.0f) {
                // Prefer outward (away from center) if both are plausible — keeps
                // the quad on the outer white border, not an inner art edge.
                cv::Point2f to_hit = hit - center;
                cv::Point2f to_old = corners[i] - center;
                if (to_hit.dot(to_hit) + 1.0f >= to_old.dot(to_old)) {
                    refined[i] = hit;
                }
            }
        }
    }
    corners = refined;

    if (inset_frac > 0.0f) {
        for (int i = 0; i < 4; i++) {
            corners[i] = corners[i] + (center - corners[i]) * inset_frac;
        }
    }
}

// Higher score => more likely upright (more Canny edges in upper half = artwork).
// Fallback when orient.onnx is not loaded.
static float scoreCardUprightCanny(const cv::Mat& grey,
                                   const std::vector<cv::Point2f>& ordered) {
    const int W = 250;
    const int H = 350;
    std::vector<cv::Point2f> dst = {
        cv::Point2f(0.0f, 0.0f),
        cv::Point2f(static_cast<float>(W - 1), 0.0f),
        cv::Point2f(static_cast<float>(W - 1), static_cast<float>(H - 1)),
        cv::Point2f(0.0f, static_cast<float>(H - 1)),
    };
    cv::Mat Hmat = cv::getPerspectiveTransform(ordered, dst);
    if (Hmat.empty()) {
        return -1e9f;
    }

    cv::Mat warped;
    // cv::warpPerspective(grey, warped, Hmat, cv::Size(W, H));

    cv::Mat edges;
    // cv::Canny(warped, edges, kCannyLow, kCannyHigh);
    cv::Canny(grey, edges, kCannyLow, kCannyHigh);

    double upper = cv::mean(edges.rowRange(0, H / 2))[0];
    double lower = cv::mean(edges.rowRange(H / 2, H))[0];
    return static_cast<float>(upper - lower);
}

static void applyCyclicShift(const std::vector<cv::Point2f>& corners,
                             int shift,
                             std::vector<cv::Point2f>& out) {
    out.resize(4);
    for (int i = 0; i < 4; i++) {
        out[i] = corners[(i + shift) % 4];
    }
}

static float cornerSsd(const std::vector<cv::Point2f>& a,
                       const std::vector<cv::Point2f>& b) {
    float ssd = 0.0f;
    for (int i = 0; i < 4; i++) {
        float dx = a[i].x - b[i].x;
        float dy = a[i].y - b[i].y;
        ssd += dx * dx + dy * dy;
    }
    return ssd;
}

// Largest per-corner Euclidean distance between two ordered quads.
static float maxCornerDist(const std::vector<cv::Point2f>& a,
                           const std::vector<cv::Point2f>& b) {
    if (a.size() != 4 || b.size() != 4) {
        return 1e9f;
    }
    float max_dist = 0.0f;
    for (int i = 0; i < 4; i++) {
        float dx = a[i].x - b[i].x;
        float dy = a[i].y - b[i].y;
        max_dist = std::max(max_dist, std::sqrt(dx * dx + dy * dy));
    }
    return max_dist;
}

// True when inner is a clearly smaller panel inside outer (text box / art window).
// slack_px tolerates edge-line refine nudging a corner slightly outside the parent.
static bool quadNestedIn(const std::vector<cv::Point2f>& inner,
                         const std::vector<cv::Point2f>& outer,
                         double inner_area,
                         double outer_area,
                         float slack_px = 3.0f,
                         double max_area_ratio = 0.85) {
    if (inner.size() != 4 || outer.size() != 4 || outer_area <= 1e-6) {
        return false;
    }
    if (inner_area >= max_area_ratio * outer_area) {
        return false;
    }
    for (const cv::Point2f& p : inner) {
        // Signed distance: >0 inside, 0 on edge, <0 outside.
        if (cv::pointPolygonTest(outer, p, /*measureDist=*/true) < -slack_px) {
            return false;
        }
    }
    return true;
}

// Recently seen outer card-aspect quads. Nested panels stay suppressed until the
// parent has been missing for holdout_frames detects (outer flicker → no text-box).
struct ParentQuadMemory {
    struct Entry {
        std::vector<cv::Point2f> corners;
        double area = 0.0;
        int frames_missing = 0;
    };
    std::vector<Entry> entries;
};

static void parentMemoryObserve(ParentQuadMemory& mem,
                                const std::vector<cv::Point2f>& corners,
                                double area,
                                float match_px = 48.0f) {
    if (corners.size() != 4 || area <= 1e-6) {
        return;
    }
    int best = -1;
    float best_d = match_px;
    for (size_t i = 0; i < mem.entries.size(); i++) {
        const float d = maxCornerDist(mem.entries[i].corners, corners);
        if (d < best_d) {
            best_d = d;
            best = static_cast<int>(i);
        }
    }
    if (best >= 0) {
        mem.entries[static_cast<size_t>(best)].corners = corners;
        mem.entries[static_cast<size_t>(best)].area = area;
        mem.entries[static_cast<size_t>(best)].frames_missing = 0;
    } else {
        ParentQuadMemory::Entry e;
        e.corners = corners;
        e.area = area;
        e.frames_missing = 0;
        mem.entries.push_back(std::move(e));
    }
}

// Drop smaller memory entries nested inside a larger one (stale text-box parents).
static void parentMemoryPruneNested(ParentQuadMemory& mem) {
    if (mem.entries.size() < 2) {
        return;
    }
    std::vector<bool> drop(mem.entries.size(), false);
    for (size_t i = 0; i < mem.entries.size(); i++) {
        for (size_t j = 0; j < mem.entries.size(); j++) {
            if (i == j || drop[j]) {
                continue;
            }
            if (quadNestedIn(mem.entries[i].corners, mem.entries[j].corners,
                             mem.entries[i].area, mem.entries[j].area)) {
                drop[i] = true;
                break;
            }
        }
    }
    size_t w = 0;
    for (size_t i = 0; i < mem.entries.size(); i++) {
        if (!drop[i]) {
            if (w != i) {
                mem.entries[w] = std::move(mem.entries[i]);
            }
            w++;
        }
    }
    mem.entries.resize(w);
}

static bool parentMemorySuppresses(const ParentQuadMemory& mem,
                                   const std::vector<cv::Point2f>& corners,
                                   double area,
                                   int holdout_frames) {
    for (const ParentQuadMemory::Entry& e : mem.entries) {
        if (e.frames_missing > holdout_frames) {
            continue;
        }
        if (quadNestedIn(corners, e.corners, area, e.area)) {
            return true;
        }
    }
    return false;
}

// Rewrite corners to card-fixed TL,TR,BR,BL (portrait family only).
// With the CNN: pick the better of 0° / 180° by scoreUpright. Always succeeds —
// "is this a full card?" is cardness.onnx, not this classifier.
// Without the CNN: classical portrait + Canny / prev-frame fallback.
static bool orientCardCorners(const cv::Mat& bgr,
                              const cv::Mat& grey,
                              std::vector<cv::Point2f>& corners,
                              const cv::Mat& K,
                              const std::vector<cv::Point2f>& prev_corners,
                              const OrientationClassifier* orient) {
    if (corners.size() != 4) {
        return false;
    }

    const float target = 2.5f / 3.5f;

    // Portrait aspect hint: prefer short edge on top before 0/180.
    float best_err = 1e30f;
    int portrait_shift = 0;
    for (int s = 0; s < 4; s++) {
        std::vector<cv::Point2f> cand;
        applyCyclicShift(corners, s, cand);
        float aspect = labelingAspect(cand, K);
        float err = std::fabs(aspect - target);
        if (err < best_err) {
            best_err = err;
            portrait_shift = s;
        }
    }

    int shift_a = portrait_shift;
    int shift_b = (portrait_shift + 2) % 4;
    std::vector<cv::Point2f> cand_a, cand_b;
    applyCyclicShift(corners, shift_a, cand_a);
    applyCyclicShift(corners, shift_b, cand_b);

    int chosen = shift_a;
    if (orient && orient->ready() && !bgr.empty()) {
        cv::Mat warp_a = rectifyCard(bgr, cand_a, orient->inputWidth(),
                                     orient->inputHeight());
        cv::Mat warp_b = rectifyCard(bgr, cand_b, orient->inputWidth(),
                                     orient->inputHeight());
        const float score_a =
            warp_a.empty() ? -1e9f : orient->scoreUpright(warp_a);
        const float score_b =
            warp_b.empty() ? -1e9f : orient->scoreUpright(warp_b);
        chosen = (score_a >= score_b) ? shift_a : shift_b;
    } else {
        const float ambiguous_margin = 0.5f;
        float score_a = scoreCardUprightCanny(grey, cand_a);
        float score_b = scoreCardUprightCanny(grey, cand_b);
        chosen = (score_a >= score_b) ? shift_a : shift_b;
        const bool clear = std::fabs(score_a - score_b) >= ambiguous_margin;
        if (!clear && prev_corners.size() == 4) {
            float ssd_a = cornerSsd(cand_a, prev_corners);
            float ssd_b = cornerSsd(cand_b, prev_corners);
            chosen = (ssd_a <= ssd_b) ? shift_a : shift_b;
        }
    }

    std::vector<cv::Point2f> oriented;
    applyCyclicShift(corners, chosen, oriented);
    corners = oriented;
    return true;
}

// Full-card gate before matching. If cardness.onnx is missing, classical
// filters alone decide (aspect / area / nesting) — do not block the pipeline.
static bool quadLooksLikeCard(const cv::Mat& bgr,
                              const std::vector<cv::Point2f>& corners,
                              const CardnessClassifier* cardness,
                              float min_margin) {
    if (corners.size() != 4 || bgr.empty()) {
        return false;
    }
    if (!cardness || !cardness->ready()) {
        return true;
    }
    cv::Mat probe = rectifyCard(bgr, corners, cardness->inputWidth(),
                                cardness->inputHeight());
    if (probe.empty()) {
        return false;
    }
    return cardness->scoreCard(probe) >= min_margin;
}

static bool pointOnImage(const cv::Point2f& p, int cols, int rows,
                         float margin = 2.0f) {
    return p.x >= margin && p.y >= margin &&
           p.x < static_cast<float>(cols) - margin &&
           p.y < static_cast<float>(rows) - margin;
}

static bool pointNearImageBorder(const cv::Point2f& p, int cols, int rows,
                                 float band_px) {
    return p.x < band_px || p.y < band_px ||
           p.x >= static_cast<float>(cols) - band_px ||
           p.y >= static_cast<float>(rows) - band_px;
}

// Contour of a card clipped by the frame puts vertices on the image edge
// near where KLT still has an off-screen corner. A true full-card re-entry
// has those corners inset, not parked on the border.
static bool detectLooksBorderClipped(const std::vector<cv::Point2f>& detect,
                                     const std::vector<cv::Point2f>& track,
                                     int cols, int rows, float band_px) {
    if (detect.size() != 4 || track.size() != 4) {
        return true;
    }
    const float on_margin = 2.0f;
    for (int i = 0; i < 4; i++) {
        if (pointOnImage(track[i], cols, rows, on_margin)) {
            continue;
        }
        float best_d2 = 1e30f;
        int best_j = -1;
        for (int j = 0; j < 4; j++) {
            const float dx = detect[j].x - track[i].x;
            const float dy = detect[j].y - track[i].y;
            const float d2 = dx * dx + dy * dy;
            if (d2 < best_d2) {
                best_d2 = d2;
                best_j = j;
            }
        }
        if (best_j >= 0 &&
            pointNearImageBorder(detect[best_j], cols, rows, band_px)) {
            return true;
        }
    }
    return false;
}

static int countCornersOnImage(const std::vector<cv::Point2f>& corners,
                               int cols, int rows, float margin = 2.0f) {
    int n = 0;
    for (const cv::Point2f& p : corners) {
        if (pointOnImage(p, cols, rows, margin)) {
            n++;
        }
    }
    return n;
}

// Rigid image motion (R, t) with scale forced to 1 (2D Kabsch). Similarity
// free to scale lets foreshortened/jittered live edges shrink the quad and
// send PnP flying. Needs >= 2 point pairs.
static bool fitRigid2D(const std::vector<cv::Point2f>& src,
                       const std::vector<cv::Point2f>& dst,
                       double& ca, double& sa,
                       cv::Point2f& c_src, cv::Point2f& c_dst) {
    if (src.size() < 2 || src.size() != dst.size()) {
        return false;
    }
    c_src = cv::Point2f(0.0f, 0.0f);
    c_dst = cv::Point2f(0.0f, 0.0f);
    for (size_t i = 0; i < src.size(); i++) {
        c_src += src[i];
        c_dst += dst[i];
    }
    const float inv_n = 1.0f / static_cast<float>(src.size());
    c_src *= inv_n;
    c_dst *= inv_n;

    double sxx = 0.0;
    double sxy = 0.0;
    double syx = 0.0;
    double syy = 0.0;
    for (size_t i = 0; i < src.size(); i++) {
        const double ax = src[i].x - c_src.x;
        const double ay = src[i].y - c_src.y;
        const double bx = dst[i].x - c_dst.x;
        const double by = dst[i].y - c_dst.y;
        sxx += ax * bx;
        sxy += ax * by;
        syx += ay * bx;
        syy += ay * by;
    }
    const double ang = std::atan2(sxy - syx, sxx + syy);
    ca = std::cos(ang);
    sa = std::sin(ang);
    return true;
}

static cv::Point2f applyRigid2D(const cv::Point2f& p,
                                double ca, double sa,
                                const cv::Point2f& c_src,
                                const cv::Point2f& c_dst) {
    const double x = p.x - c_src.x;
    const double y = p.y - c_src.y;
    return cv::Point2f(static_cast<float>(ca * x - sa * y + c_dst.x),
                       static_cast<float>(sa * x + ca * y + c_dst.y));
}

// Sparse pyramidal LK on the 4 card corners with forward-backward consistency.
// With allow_off_screen: keep tracking while >= 2 corners remain live and
// track cleanly; recover the rest with a rigid (R,t) image motion — no scale.
// "Not live" covers off-screen corners and occluded ones that latch onto a
// finger/object (FB still passes, but the corner disagrees with the rigid
// motion of the others). A clean LK track that leaves the frame this step is
// kept as-is (not rigid-replaced) so perspective exits can extend fully off
// screen. Otherwise all 4 must track on-screen. live_out[i] is set when
// corner i was LK-tracked live on-screen. Recovered corners may sit outside
// the image so pose/mesh can overflow the window.
static bool trackCornersLK(const cv::Mat& prev_grey,
                           const cv::Mat& grey,
                           const std::vector<cv::Point2f>& prev_corners,
                           std::vector<cv::Point2f>& next_corners,
                           bool live_out[4],
                           bool allow_off_screen,
                           cv::Size lk_win,
                           int lk_max_level,
                           float max_fb_err_px,
                           float max_jump_px,
                           float max_rigid_residual_px) {
    next_corners.clear();
    if (live_out) {
        live_out[0] = live_out[1] = live_out[2] = live_out[3] = false;
    }
    if (prev_grey.empty() || grey.empty() || prev_corners.size() != 4) {
        return false;
    }

    const cv::TermCriteria criteria(
        cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 30, 0.01);
        // Stop after at most 30 iterations or when the error is less than 0.01
    std::vector<cv::Point2f> fwd;
    std::vector<uchar> status_fwd;
    std::vector<float> err_fwd;
    cv::calcOpticalFlowPyrLK(prev_grey, grey, prev_corners, fwd,
                             status_fwd, err_fwd, lk_win, lk_max_level, criteria);

    if (fwd.size() != 4 || status_fwd.size() != 4) {
        return false;
    }

    std::vector<cv::Point2f> back;
    std::vector<uchar> status_back;
    std::vector<float> err_back;
    cv::calcOpticalFlowPyrLK(grey, prev_grey, fwd, back,
                             status_back, err_back, lk_win, lk_max_level, criteria);

    if (back.size() != 4 || status_back.size() != 4) {
        return false;
    }

    const float max_jump_sq = max_jump_px * max_jump_px;
    const float max_fb_sq = max_fb_err_px * max_fb_err_px;
    const float margin = 2.0f;
    // Near-border corners often disagree with a rigid fit under perspective;
    // do not treat that as occlusion — let them exit (or stay live).
    const float residual_edge_band =
        std::max(24.0f, 0.5f * static_cast<float>(lk_win.width));

    bool valid[4] = {false, false, false, false};
    // LK already produced a usable off-screen position this frame — keep it.
    bool kept_exit[4] = {false, false, false, false};
    next_corners.assign(4, cv::Point2f(0.0f, 0.0f));
    int n_live = 0;

    for (int i = 0; i < 4; i++) {
        // Already off-screen last frame: LK can't see it — recover later.
        if (!pointOnImage(prev_corners[i], prev_grey.cols, prev_grey.rows,
                          margin)) {
            if (!allow_off_screen) {
                next_corners.clear();
                return false;
            }
            continue;
        }
        if (!status_fwd[i] || !status_back[i]) {
            if (!allow_off_screen) {
                next_corners.clear();
                return false;
            }
            continue;
        }
        float jx = fwd[i].x - prev_corners[i].x;
        float jy = fwd[i].y - prev_corners[i].y;
        if (jx * jx + jy * jy > max_jump_sq) {
            if (!allow_off_screen) {
                next_corners.clear();
                return false;
            }
            continue;
        }
        float fbx = back[i].x - prev_corners[i].x;
        float fby = back[i].y - prev_corners[i].y;
        if (fbx * fbx + fby * fby > max_fb_sq) {
            if (!allow_off_screen) {
                next_corners.clear();
                return false;
            }
            continue;
        }
        // Left the frame this step with a clean LK track — keep that outside
        // position (not an anchor). Rigid recovery from quieter interior
        // corners underestimates perspective exits.
        if (!pointOnImage(fwd[i], grey.cols, grey.rows, margin)) {
            if (!allow_off_screen) {
                next_corners.clear();
                return false;
            }
            next_corners[i] = fwd[i];
            kept_exit[i] = true;
            continue;
        }
        valid[i] = true;
        next_corners[i] = fwd[i];
        n_live++;
    }

    const int min_live = allow_off_screen ? 2 : 4;
    if (n_live < min_live) {
        next_corners.clear();
        return false;
    }

    if (allow_off_screen) {
        // Drop interior corners that disagree with the rigid motion of the
        // others (occlusion: LK latches onto a finger while FB still looks
        // fine). Skip near-border corners so exit motion is not punished.
        const float max_res_sq =
            max_rigid_residual_px * max_rigid_residual_px;
        while (n_live > min_live) {
            std::vector<cv::Point2f> src;
            std::vector<cv::Point2f> dst;
            src.reserve(static_cast<size_t>(n_live));
            dst.reserve(static_cast<size_t>(n_live));
            for (int i = 0; i < 4; i++) {
                if (valid[i]) {
                    src.push_back(prev_corners[i]);
                    dst.push_back(next_corners[i]);
                }
            }
            double ca = 1.0;
            double sa = 0.0;
            cv::Point2f c_src;
            cv::Point2f c_dst;
            if (!fitRigid2D(src, dst, ca, sa, c_src, c_dst)) {
                break;
            }

            int worst = -1;
            float worst_sq = max_res_sq;
            for (int i = 0; i < 4; i++) {
                if (!valid[i]) {
                    continue;
                }
                if (pointNearImageBorder(next_corners[i], grey.cols, grey.rows,
                                         residual_edge_band)) {
                    continue;
                }
                const cv::Point2f pred =
                    applyRigid2D(prev_corners[i], ca, sa, c_src, c_dst);
                const float dx = next_corners[i].x - pred.x;
                const float dy = next_corners[i].y - pred.y;
                const float r2 = dx * dx + dy * dy;
                if (r2 > worst_sq) {
                    worst_sq = r2;
                    worst = i;
                }
            }
            if (worst < 0) {
                break;
            }
            valid[worst] = false;
            n_live--;
        }

        if (n_live < 4) {
            // Recover missing corners from the live anchors. Do not overwrite
            // clean LK exits kept above. Off-screen points may sit outside the
            // frame — that is intentional (pose/mesh may overflow the window).
            std::vector<cv::Point2f> src;
            std::vector<cv::Point2f> dst;
            src.reserve(static_cast<size_t>(n_live));
            dst.reserve(static_cast<size_t>(n_live));
            for (int i = 0; i < 4; i++) {
                if (valid[i]) {
                    src.push_back(prev_corners[i]);
                    dst.push_back(next_corners[i]);
                }
            }
            double ca = 1.0;
            double sa = 0.0;
            cv::Point2f c_src;
            cv::Point2f c_dst;
            if (!fitRigid2D(src, dst, ca, sa, c_src, c_dst)) {
                next_corners.clear();
                return false;
            }
            for (int i = 0; i < 4; i++) {
                if (valid[i] || kept_exit[i]) {
                    continue;
                }
                next_corners[i] =
                    applyRigid2D(prev_corners[i], ca, sa, c_src, c_dst);
            }
        }
    }
    if (live_out) {
        for (int i = 0; i < 4; i++) {
            live_out[i] = valid[i];
        }
    }
    return true;
}

// True if every live track corner lies near some corner of the detected quad.
static bool quadSharesLiveAnchors(const std::vector<cv::Point2f>& quad,
                                  const std::vector<cv::Point2f>& track,
                                  const bool live[4],
                                  float max_dist_px) {
    if (quad.size() != 4 || track.size() != 4 || live == nullptr) {
        return false;
    }
    int need = 0;
    int hit = 0;
    const float max_dist_sq = max_dist_px * max_dist_px;
    for (int i = 0; i < 4; i++) {
        if (!live[i]) {
            continue;
        }
        need++;
        float best_sq = 1e30f;
        for (int j = 0; j < 4; j++) {
            const float dx = quad[j].x - track[i].x;
            const float dy = quad[j].y - track[i].y;
            best_sq = std::min(best_sq, dx * dx + dy * dy);
        }
        if (best_sq <= max_dist_sq) {
            hit++;
        }
    }
    return need >= 2 && hit == need;
}

// Reject exploded / collapsed KLT quads (common under fast motion when FB/jump
// thresholds are loose and corners latch onto background texture).
// Off-screen extrapolated corners are fine: checks use full-quad geometry.
static bool trackedQuadSane(const std::vector<cv::Point2f>& prev,
                            const std::vector<cv::Point2f>& next,
                            float min_area = 800.0f,
                            float area_ratio_lo = 0.5f,
                            float area_ratio_hi = 1.8f,
                            float side_ratio_hi = 1.75f) {
    if (prev.size() != 4 || next.size() != 4) {
        return false;
    }

    std::vector<cv::Point> poly(4);
    for (int i = 0; i < 4; i++) {
        poly[i] = cv::Point(cvRound(next[i].x), cvRound(next[i].y));
    }
    if (!cv::isContourConvex(poly)) {
        return false;
    }

    const float a_prev = std::fabs(static_cast<float>(cv::contourArea(prev)));
    const float a_next = std::fabs(static_cast<float>(cv::contourArea(next)));
    if (a_next < min_area || a_prev < 1.0f) {
        return false;
    }
    const float area_ratio = a_next / a_prev;
    if (area_ratio < area_ratio_lo || area_ratio > area_ratio_hi) {
        return false;
    }

    for (int i = 0; i < 4; i++) {
        const float lp = cv::norm(prev[(i + 1) % 4] - prev[i]);
        const float ln = cv::norm(next[(i + 1) % 4] - next[i]);
        if (lp < 1.0f) {
            return false;
        }
        const float r = ln / lp;
        if (r < 1.0f / side_ratio_hi || r > side_ratio_hi) {
            return false;
        }
    }
    return true;
}

// Card-aspect quads from Canny contours, largest area first. Each entry is
// ordered TL,TR,BR,BL and edge-line refined.
struct DetectedQuad {
    std::vector<cv::Point2f> corners;
    double area = 0.0;
};

static bool detectCardQuads(const cv::Mat& grey,
                            const cv::Mat& camera_mat,
                            std::vector<DetectedQuad>& quads_out,
                            bool draw_edge_quad,
                            ParentQuadMemory* parent_mem = nullptr,
                            int parent_holdout_frames = 5) {
    quads_out.clear();

    cv::Mat blurred;
    cv::GaussianBlur(grey, blurred, cv::Size(15, 15), 0);
    cv::imshow("Blurred", blurred);

    // --- tune these ---
    double min_area = 1000.0;           // reject tiny quads that aren't cards
    double approx_eps_frac = 0.04;      // approxPolyDP epsilon = frac * perimeter
    // ------------------

    cv::Mat edges;
    cv::Canny(blurred, edges, kCannyLow, kCannyHigh, 3, true);
    cv::imshow("Edges (pre-morph)", edges);

    cv::morphologyEx(edges, edges, cv::MORPH_CLOSE,
        cv::getStructuringElement(cv::MORPH_RECT, cv::Size(13, 13)));

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(edges, contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);

    struct Cand {
        std::vector<cv::Point2f> corners;
        std::vector<cv::Point> contour;
        double area;
    };
    std::vector<Cand> cands;

    for (size_t i = 0; i < contours.size(); i++) {
        double area = cv::contourArea(contours[i]);
        if (area < min_area) {
            continue;
        }

        double peri = cv::arcLength(contours[i], true);
        std::vector<cv::Point> approx;
        cv::approxPolyDP(contours[i], approx, approx_eps_frac * peri, true);
        if (approx.size() != 4 || !cv::isContourConvex(approx)) {
            continue;
        }

        std::vector<cv::Point2f> ordered;
        for (int j = 0; j < 4; j++) {
            ordered.push_back(cv::Point2f((float)approx[j].x, (float)approx[j].y));
        }
        orderQuadCorners(ordered);
        if (!quadHasCardAspect(ordered, camera_mat)) {
            continue;
        }

        Cand c;
        c.corners = ordered;
        c.contour = contours[i];
        c.area = area;
        refineQuadByEdgeLines(edges, c.corners, &c.contour);
        cands.push_back(std::move(c));
    }

    std::sort(cands.begin(), cands.end(),
              [](const Cand& a, const Cand& b) { return a.area > b.area; });

    // Largest first: keep outers, drop near-duplicates and panels nested inside a
    // kept card-aspect quad (text box / art window share ~card aspect).
    const float dup_px = 12.0f;
    for (const Cand& c : cands) {
        bool skip = false;
        for (const DetectedQuad& keep : quads_out) {
            if (maxCornerDist(c.corners, keep.corners) < dup_px ||
                quadNestedIn(c.corners, keep.corners, c.area, keep.area)) {
                skip = true;
                break;
            }
        }
        if (!skip) {
            DetectedQuad q;
            q.corners = c.corners;
            q.area = c.area;
            quads_out.push_back(std::move(q));
        }
    }

    // Temporal holdout: if a larger parent was seen recently, keep suppressing
    // nested panels for parent_holdout_frames detects after it disappears.
    if (parent_mem != nullptr) {
        for (ParentQuadMemory::Entry& e : parent_mem->entries) {
            e.frames_missing++;
        }

        std::vector<DetectedQuad> kept;
        kept.reserve(quads_out.size());
        for (const DetectedQuad& q : quads_out) {
            if (!parentMemorySuppresses(*parent_mem, q.corners, q.area,
                                        parent_holdout_frames)) {
                kept.push_back(q);
            }
        }
        quads_out.swap(kept);

        parent_mem->entries.erase(
            std::remove_if(parent_mem->entries.begin(), parent_mem->entries.end(),
                           [parent_holdout_frames](const ParentQuadMemory::Entry& e) {
                               return e.frames_missing > parent_holdout_frames;
                           }),
            parent_mem->entries.end());

        for (const DetectedQuad& q : quads_out) {
            parentMemoryObserve(*parent_mem, q.corners, q.area);
        }
        parentMemoryPruneNested(*parent_mem);
    }

    cv::Mat edges_vis;
    cv::cvtColor(edges, edges_vis, cv::COLOR_GRAY2BGR);
    if (draw_edge_quad) {
        for (size_t q = 0; q < quads_out.size(); q++) {
            const cv::Scalar color =
                (q == 0) ? cv::Scalar(0, 0, 255) : cv::Scalar(0, 128, 255);
            const int thickness = (q == 0) ? 2 : 1;
            for (int i = 0; i < 4; i++) {
                cv::line(edges_vis, quads_out[q].corners[i],
                         quads_out[q].corners[(i + 1) % 4], color, thickness);
            }
        }
    }
    cv::imshow("Edges", edges_vis);
    return !quads_out.empty();
}

// True if rectified crop looks like a full card and embeds to a gallery hit.
static bool quadMatchesCard(const cv::Mat& bgr,
                            const std::vector<cv::Point2f>& corners,
                            CardMatcher& matcher,
                            float min_score,
                            const CardnessClassifier* cardness,
                            float cardness_min_margin,
                            std::vector<CardMatch>* cands_out = nullptr) {
    if (corners.size() != 4 || bgr.empty()) {
        return false;
    }
    if (!quadLooksLikeCard(bgr, corners, cardness, cardness_min_margin)) {
        return false;
    }
    cv::Mat probe = rectifyCard(bgr, corners, matcher.inputWidth(),
                                matcher.inputHeight());
    if (probe.empty()) {
        return false;
    }
    std::vector<CardMatch> local;
    std::vector<CardMatch>& cands = cands_out ? *cands_out : local;
    return matcher.match(probe, 5, cands) && !cands.empty() &&
           cands[0].score >= min_score;
}

// Search the usual run locations (build/, project_final/, repo root) for a data file.
static std::string resolveDataPath(const std::string& relative) {
    const char* prefixes[] = {"", "../", "../../"};
    for (const char* prefix : prefixes) {
        std::string candidate = std::string(prefix) + relative;
        std::ifstream probe(candidate, std::ios::binary);
        if (probe.is_open()) {
            return candidate;
        }
    }
    return relative;
}

// Same sanitization as card_match/download_tcgdex.py safe_filename().
static std::string galleryImageStem(const std::string& card_id) {
    std::string stem;
    stem.reserve(card_id.size());
    for (unsigned char ch : card_id) {
        if (std::isalnum(ch) || ch == '-' || ch == '_') {
            stem.push_back(static_cast<char>(ch));
        } else {
            stem.push_back('_');
        }
    }
    return stem;
}

// Official TCGdex scan for a matched card id. Tries the usual extensions under
// data/tcg/en/images/; returns an empty Mat if the download is missing.
static cv::Mat loadGalleryScan(const std::string& card_id) {
    if (card_id.empty()) {
        return cv::Mat();
    }
    const std::string stem = galleryImageStem(card_id);
    const char* extensions[] = {".webp", ".png", ".jpg"};
    for (const char* extension : extensions) {
        const std::string path =
            resolveDataPath(std::string("data/tcg/en/images/") + stem + extension);
        cv::Mat image = cv::imread(path, cv::IMREAD_COLOR);
        if (!image.empty()) {
            return image;
        }
    }
    return cv::Mat();
}

// Most-frequent card id over the recent match history. One frame can be spoiled
// by glare or blur, so a single top-1 is not worth showing on its own.
static bool voteMatch(const std::deque<CardMatch>& history,
                      int min_votes,
                      CardMatch& winner) {
    if (history.empty()) {
        return false;
    }
    std::map<std::string, int> votes;
    for (const CardMatch& entry : history) {
        votes[entry.card_id]++;
    }

    std::string best_id;
    int best_votes = 0;
    for (const auto& pair : votes) {
        if (pair.second > best_votes) {
            best_votes = pair.second;
            best_id = pair.first;
        }
    }
    if (best_votes < min_votes) {
        return false;
    }

    // Report the winning id at its best observed score.
    float best_score = -2.0f;
    for (const CardMatch& entry : history) {
        if (entry.card_id == best_id && entry.score > best_score) {
            best_score = entry.score;
            winner = entry;
        }
    }
    return best_score > -2.0f;
}

static bool loadIntrinsics(const char* path,
                           cv::Mat& camera_mat,
                           cv::Mat& dist_coeffs) {
    cv::FileStorage fs(path, cv::FileStorage::READ);
    if (!fs.isOpened()) {
        return false;
    }
    fs["camera_mat"] >> camera_mat;
    fs["dist_coeffs"] >> dist_coeffs;
    fs.release();
    return !camera_mat.empty();
}

int main(int argc, char* argv[]) {
    const char* usage =
        "Usage: %s [--model pyramid|pokemon] [--models models-resource|poke-3D] "
        "[--scale MULT] [--scale-y Y] [--pokemon NAME]... [--match on|off] "
        "[--freeze-quad on|off] [--allow-off-screen-corners on|off] "
        "[--embedder PATH] [--gallery PATH] [--orient PATH] [--cardness PATH] "
        "[/dev/videoN]\n"
        "  --model     overlay type: pokemon (default) or pyramid\n"
        "  --models    3D source: models-resource (default, XY DAEs) or poke-3D\n"
        "              (Pokemon-3D-api GLBs). Needs node deps for poke-3D:\n"
        "              cd tools/poke3d_prep && npm install\n"
        "  --scale     global multiplier on Pokédex log-scale height "
        "(default %.2f)\n"
        "  --scale-y   exponent in log1p(height_m^y); lower = flatter sizes "
        "(default %.2f)\n"
        "  --pokemon   download NAME for this run only (Models Resource or\n"
        "              poke-3D depending on --models)\n"
        "  --match     identify the card against the TCG gallery (default off).\n"
        "              Drives the 3D overlay from the matched national dex;\n"
        "              skips the bundled model cycle. Needs card_match artifacts\n"
        "  --freeze-quad  after the locked quad matches, hold it / KLT and do\n"
        "              not probe alternatives (default on). off = keep hunting\n"
        "              higher-area quads that match. Still hunts while identifying\n"
        "  --allow-off-screen-corners  keep KLT while >=2 corners stay live\n"
        "              (off-screen or occluded outliers are extrapolated from the\n"
        "              live anchors; redetect only quads that share those anchors).\n"
        "              Default off = require all 4 on-screen\n"
        "  --embedder  embedder.onnx path (default data/card_match/embedder.onnx)\n"
        "  --gallery   gallery.bin path (default data/card_match/gallery.bin)\n"
        "  --orient    orient.onnx path (default data/card_match/orient.onnx).\n"
        "              Upright vs 180; falls back to Canny if missing\n"
        "  --cardness  cardness.onnx path (default data/card_match/cardness.onnx).\n"
        "              Full-card vs panel gate before matching; optional\n";

    std::string model_choice = "pokemon";
    std::string models_source = "models-resource";
    const char* device = "/dev/video0";
    float scale_multiplier = kDefaultScaleMultiplier;
    float scale_y = kDefaultScaleY;
    std::vector<std::string> requested_downloads;
    bool want_match = false;
    bool freeze_quad = true;
    bool allow_off_screen_corners = false;
    std::string embedder_path;
    std::string gallery_path;
    std::string orient_path;
    std::string cardness_path;

    auto printUsage = [&]() {
        printf(usage, argv[0], kDefaultScaleMultiplier, kDefaultScaleY);
    };

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--model") == 0) {
            if (i + 1 >= argc) {
                printUsage();
                return -1;
            }
            model_choice = argv[++i];
            if (model_choice != "pyramid" && model_choice != "pokemon") {
                printf("--model must be \"pyramid\" or \"pokemon\"\n");
                return -1;
            }
        } else if (std::strcmp(argv[i], "--models") == 0) {
            if (i + 1 >= argc) {
                printUsage();
                return -1;
            }
            models_source = argv[++i];
            if (models_source != "models-resource" && models_source != "poke-3D") {
                printf("--models must be \"models-resource\" or \"poke-3D\"\n");
                return -1;
            }
        } else if (std::strcmp(argv[i], "--scale") == 0) {
            if (i + 1 >= argc) {
                printUsage();
                return -1;
            }
            scale_multiplier = static_cast<float>(std::atof(argv[++i]));
            if (!(scale_multiplier > 0.f)) {
                printf("--scale must be a positive number\n");
                return -1;
            }
        } else if (std::strcmp(argv[i], "--scale-y") == 0) {
            if (i + 1 >= argc) {
                printUsage();
                return -1;
            }
            scale_y = static_cast<float>(std::atof(argv[++i]));
            if (!(scale_y > 0.f)) {
                printf("--scale-y must be a positive number\n");
                return -1;
            }
        } else if (std::strcmp(argv[i], "--pokemon") == 0) {
            if (i + 1 >= argc) {
                printUsage();
                return -1;
            }
            requested_downloads.push_back(argv[++i]);
        } else if (std::strcmp(argv[i], "--match") == 0) {
            if (i + 1 >= argc) {
                printUsage();
                return -1;
            }
            std::string value = argv[++i];
            if (value != "on" && value != "off") {
                printf("--match must be \"on\" or \"off\"\n");
                return -1;
            }
            want_match = (value == "on");
        } else if (std::strcmp(argv[i], "--freeze-quad") == 0) {
            if (i + 1 >= argc) {
                printUsage();
                return -1;
            }
            std::string value = argv[++i];
            if (value != "on" && value != "off") {
                printf("--freeze-quad must be \"on\" or \"off\"\n");
                return -1;
            }
            freeze_quad = (value == "on");
        } else if (std::strcmp(argv[i], "--allow-off-screen-corners") == 0) {
            if (i + 1 >= argc) {
                printUsage();
                return -1;
            }
            std::string value = argv[++i];
            if (value != "on" && value != "off") {
                printf("--allow-off-screen-corners must be \"on\" or \"off\"\n");
                return -1;
            }
            allow_off_screen_corners = (value == "on");
        } else if (std::strcmp(argv[i], "--embedder") == 0) {
            if (i + 1 >= argc) {
                printUsage();
                return -1;
            }
            embedder_path = argv[++i];
        } else if (std::strcmp(argv[i], "--gallery") == 0) {
            if (i + 1 >= argc) {
                printUsage();
                return -1;
            }
            gallery_path = argv[++i];
        } else if (std::strcmp(argv[i], "--orient") == 0) {
            if (i + 1 >= argc) {
                printUsage();
                return -1;
            }
            orient_path = argv[++i];
        } else if (std::strcmp(argv[i], "--cardness") == 0) {
            if (i + 1 >= argc) {
                printUsage();
                return -1;
            }
            cardness_path = argv[++i];
        } else if (argv[i][0] == '-') {
            printf("Unknown argument: %s\n", argv[i]);
            printUsage();
            return -1;
        } else {
            device = argv[i];
        }
    }

    const bool use_pokemon = (model_choice == "pokemon");
    const bool use_poke3d = (models_source == "poke-3D");

    // Assemble the model list before opening the camera, so a bad --pokemon name or
    // a network problem is reported straight away. Skipped entirely when --match
    // is on: the overlay is then driven by the matched card's national dex.
    std::vector<ModelEntry> model_library;
    // Owns the RAM-backed scratch dirs for --pokemon / dex downloads; they are
    // deleted when this goes out of scope at the end of main.
    ModelDownloader downloader;
    Poke3dDownloader poke3d_downloader;
    const std::vector<std::string> asset_roots = {
        "../data/assets", "data/assets", "../../data/assets"};

    if (use_pokemon && !want_match) {
        // Bundled XY DAEs only apply to the Models Resource path.
        if (!use_poke3d) {
            model_library = discoverModels(asset_roots);
        }

        for (const std::string& wanted : requested_downloads) {
            std::vector<ModelEntry> fetched;
            std::string error;
            const bool ok = use_poke3d
                                ? poke3d_downloader.download(wanted, fetched, error)
                                : downloader.download(wanted, fetched, error);
            if (ok) {
                for (ModelEntry& entry : fetched) {
                    printf("  + %s (downloaded, this session only)\n", entry.name.c_str());
                    model_library.push_back(std::move(entry));
                }
            } else {
                printf("Could not fetch \"%s\": %s\n", wanted.c_str(), error.c_str());
            }
        }
        std::sort(model_library.begin(), model_library.end(),
                  [](const ModelEntry& a, const ModelEntry& b) { return a.name < b.name; });

        printf("%zu model(s) available via %s (scale %.2f, scale-y %.2f):\n",
               model_library.size(), models_source.c_str(), scale_multiplier, scale_y);
        for (size_t i = 0; i < model_library.size(); i++) {
            printf("  [%zu] %s\n", i, model_library[i].name.c_str());
        }
        if (model_library.empty()) {
            printf("No models found — 'm' will fall back to the pyramid\n");
            if (use_poke3d) {
                printf("  tip: pass --pokemon NAME or use --match on with poke-3D\n");
            }
        }
    } else if (!requested_downloads.empty() && !want_match) {
        printf("--pokemon needs --model pokemon; ignoring the requested downloads\n");
    } else if (want_match && !requested_downloads.empty()) {
        printf("--match on drives models from dex; ignoring --pokemon / --model\n");
    }

    // Card identification against the TCG gallery. Detection stays in C++; the
    // ONNX embedder and gallery.bin come from the offline card_match package.
    CardMatcher card_matcher;
    bool match_mode = false;
    if (want_match) {
        if (embedder_path.empty()) {
            embedder_path = resolveDataPath("data/card_match/embedder.onnx");
        }
        if (gallery_path.empty()) {
            gallery_path = resolveDataPath("data/card_match/gallery.bin");
        }
        std::string error;
        if (card_matcher.init(embedder_path, gallery_path, error)) {
            match_mode = true;
            printf("Card matcher ready — %zu cards in the gallery\n",
                   card_matcher.gallerySize());
            printf("3D model source: %s\n", models_source.c_str());
            if (!use_poke3d) {
                // Prefetch the Models Resource index so the first match only pays
                // for the archive download, not the index scrape.
                std::string index_error;
                if (downloader.fetchIndex(index_error)) {
                    printf("Models Resource index ready — %zu Pokemon X/Y models\n",
                           downloader.indexSize());
                } else {
                    printf("Models Resource index unavailable: %s\n", index_error.c_str());
                    printf("  matched cards will still identify; 3D overlay needs the index\n");
                }
            }
        } else {
            printf("Card matcher disabled: %s\n", error.c_str());
            printf("  build the artifacts first: see project_final/card_match/README.md\n");
        }
    }

    // Optional learned upright / 180 classifier (same crop as the matcher).
    OrientationClassifier orient_classifier;
    if (orient_path.empty()) {
        orient_path = resolveDataPath("data/card_match/orient.onnx");
    }
    {
        std::string error;
        if (orient_classifier.init(orient_path, error)) {
            printf("Orientation classifier ready (%s)\n", orient_path.c_str());
        } else {
            printf("Orientation classifier off (%s) — using Canny upright score\n",
                   error.c_str());
        }
    }

    // Optional full-card vs panel gate before embedding / matching.
    CardnessClassifier cardness_classifier;
    if (cardness_path.empty()) {
        cardness_path = resolveDataPath("data/card_match/cardness.onnx");
    }
    {
        std::string error;
        if (cardness_classifier.init(cardness_path, error)) {
            printf("Cardness classifier ready (%s)\n", cardness_path.c_str());
        } else {
            printf("Cardness classifier off (%s) — classical quad filters only\n",
                   error.c_str());
        }
    }

    // Match mode always wants the textured Pokemon path, not the pyramid.
    const bool overlay_pokemon = use_pokemon || match_mode;

    cv::VideoCapture* capdev = new cv::VideoCapture(device, cv::CAP_V4L2);

    if (!capdev->isOpened()) {
        printf("Unable to open video device\n");
        return -1;
    }
    cv::namedWindow("Video", 1);
    cv::namedWindow("Edges", 1);
    cv::namedWindow("Blurred", 1);
    if (match_mode) {
        cv::namedWindow("Card", 1);
        cv::namedWindow("Match", 1);
    }
    placeAppWindows(match_mode);

    cv::Mat frame;
    cv::Size img_size = cv::Size(640, 480);

    // Card plane in world units (Z = 0). Aspect ~ Pokemon TCG card (2.5x3.5 in. or 63.5x88 mm).
    const float card_w = 2.5f;
    const float card_h = 3.5f;
    std::vector<cv::Point3f> card_object_pts = {
        cv::Point3f(0.0f, 0.0f, 0.0f),
        cv::Point3f(card_w, 0.0f, 0.0f),
        cv::Point3f(card_w, card_h, 0.0f),
        cv::Point3f(0.0f, card_h, 0.0f),
    };

    // initialize camera intrinsic matrix (usable default for aspect recovery / pose)
    double f = static_cast<double>(std::max(img_size.width, img_size.height));
    cv::Mat camera_mat = cv::Mat::eye(3, 3, CV_64FC1);
    camera_mat.at<double>(0, 0) = f;
    camera_mat.at<double>(1, 1) = f;
    camera_mat.at<double>(0, 2) = img_size.width / 2.0;
    camera_mat.at<double>(1, 2) = img_size.height / 2.0;
    cv::Mat dist_coeffs;
    cv::Mat rvec;
    cv::Mat tvec;
    cv::Mat last_rvec;
    cv::Mat last_tvec;
    bool have_last_pose = false;
    bool intrinsics_loaded = false;
    bool pose_mode = false;

    // project_final/camera_intrinsics.yaml (cwd is usually project_final/build/)
    const char* intrinsic_paths[] = {
        "../camera_intrinsics.yaml",
        "camera_intrinsics.yaml",
    };

    auto ensureIntrinsics = [&]() -> bool {
        if (intrinsics_loaded) {
            return true;
        }
        for (size_t i = 0; i < sizeof(intrinsic_paths) / sizeof(intrinsic_paths[0]); i++) {
            if (loadIntrinsics(intrinsic_paths[i], camera_mat, dist_coeffs)) {
                printf("Loaded intrinsics from %s\n", intrinsic_paths[i]);
                intrinsics_loaded = true;
                return true;
            }
        }
        printf("Could not open camera_intrinsics.yaml — "
               "run ./calibrate_camera from build/, or place yaml in project_final/\n");
        return false;
    };

    printf("Controls: q=quit  p=pose  m=next model  s=corners  o=overlays  c=candidates\n");
    if (match_mode) {
        printf("Match mode: 3D overlay follows the matched national dex  "
               "('m' toggles overlay on/off)\n");
    }
    if (freeze_quad) {
        printf("freeze-quad on: after a match, hold that quad (pass "
               "--freeze-quad off to keep probing)\n");
    } else {
        printf("freeze-quad off: keep probing alternatives after a match\n");
    }
    printf("Model: %s  source: %s  (pass --model / --models)\n",
           match_mode ? "dex-driven" : model_choice.c_str(),
           models_source.c_str());
    printf("Green quad=detect  Cyan quad=KLT track%s\n",
           allow_off_screen_corners
               ? "  (coasts with >=2 live corners)"
               : "");
    if (allow_off_screen_corners) {
        printf("allow-off-screen-corners on: KLT holds while >=2 corners remain "
               "live (off-screen / occluded corners are rigid-extrapolated)\n");
    }

    // Adaptive corner EMA: low alpha when still (kill flicker), high when moving (track).
    std::vector<cv::Point2f> smooth_corners;
    const float alpha_min = 0.05f;   // still: more smoothing (lower = more of the previous frame)
    const float alpha_max = 0.85f;   // moving: follow detection (higher = follow the detector more closely)
    const float speed_ref_px = 20.0f; // distance (px) that maps to alpha_max (higher = more sensitive to movement)

    // KLT optical-flow tracking (coasts through motion blur when Canny drops).
    cv::Mat prev_grey;
    std::vector<cv::Point2f> track_corners;
    bool tracking = false;
    // True after any frame where the lock had corners off-screen. Cleared when a
    // fresh full-quad detect re-acquires — used to force-reload the 3D mesh and
    // drop the coasted PnP seed (otherwise the overlay can stay wildly scaled).
    bool coasted_offscreen = false;
    const cv::Size lk_win(21, 21);
    const int lk_max_level = 3;
    const float lk_max_fb_err_px = 3.0f;   // forward-backward consistency
    const float lk_max_jump_px = 80.0f;    // reject wild per-corner jumps
    // Reject LK corners that disagree with rigid motion of the others (occlusion).
    const float lk_max_rigid_residual_px = 10.0f;
    const int redetect_every = 1;        // periodic detect while tracking (kill drift)
    const float redetect_accept_ssd = 40.0f * 40.0f * 4.0f;  // prefer detect if close to track
    // Track drifted far from a good detect (e.g. after off-screen coast) → snap.
    const float redetect_snap_ssd = 80.0f * 80.0f * 4.0f;
    // If KLT moved less than this (px), treat as still → always trust detect over drifted track.
    const float still_motion_px = 3.0f;
    // While locked, a detect far from the current quad is a "challenger" (often a
    // larger false contour). Only adopt it if it embeds to a real gallery card.
    const float challenger_gate_px = 20.0f;
    // After off-screen KLT coast the lock can sit far from the true contour;
    // allow re-acquiring within this radius (or via same-card embed).
    const float recovery_gate_px = 220.0f;
    // While coasting (off-screen / occluded), only adopt a detect that shares
    // the live KLT corners.
    const float anchor_share_px = 28.0f;
    // Reject frame-edge clipped contour vertices while coasting (px from border).
    const float coast_clip_band_px = 10.0f;
    // After a match miss, deprioritize that quad until other candidates are tried.
    std::vector<cv::Point2f> skip_match_corners;
    int frames_since_detect = 0;
    bool from_track = false;              // true => cyan quad this frame
    bool show_overlays = true;            // Video quad/labels + Edges red outline
    bool model_mode = false;
    // CNN upright lock: no 3D overlay until a quad passes cardness (and orient).
    bool orientation_confident = false;
    // Outer card-aspect quads linger here so nested text-box panels stay suppressed
    // for a few frames after the outer contour flickers out.
    ParentQuadMemory parent_quad_mem;
    const int parent_holdout_frames = 5;

    // Card matching. Embedding 640x480 -> 224x312 costs a few ms, so it runs on
    // a subset of frames and the answer is held between runs. Once an id is
    // locked, keep it (and the 3D mesh) for as long as the quad is tracked —
    // failed votes / low-angle misses must not blank the overlay.
    const int match_every = 3;
    const size_t match_history_len = 7;
    const int match_min_votes = 3;
    // Measured over the whole gallery: correct cards score 0.70 at worst and
    // 0.945 median, while crops that are not cards at all (noise, flat colour,
    // scrambled art) top out at 0.707. Sitting on that boundary is safe because
    // a spurious hit lands on a different card each frame and loses the vote,
    // whereas a threshold set high enough to be strict would drop real cards in
    // poor light.
    const float match_min_score = 0.70f;
    // logit_card - logit_not; 0 => argmax prefers "card". Raise after a full train.
    const float cardness_min_margin = 0.0f;
    std::deque<CardMatch> match_history;
    std::vector<CardMatch> match_candidates;
    CardMatch current_card;
    bool have_card_id = false;
    cv::Mat rectified_card;
    cv::Mat gallery_card;          // official scan of the current match
    std::string gallery_card_id;   // cache key so we do not reload every frame
    int frames_since_match = 0;

    // Same simple pyramid as project4 (base + apex), in card units; -Z out of card.
    const std::vector<cv::Point3f> pyramid_pts = {
        cv::Point3f(0.25f, 0.75f, -0.5f),
        cv::Point3f(2.25f, 0.75f, -0.5f),
        cv::Point3f(2.25f, 2.75f, -0.5f),
        cv::Point3f(0.25f, 2.75f, -0.5f),
        cv::Point3f(1.25f, 1.75f, -2.0f),
    };

    // OpenGL Pokemon models. Without --match they come from data/assets and are
    // cycled with 'm'. With --match they are resolved from the card's national dex.
    GLRenderer gl_renderer;
    ObjMesh pokemon_mesh;
    std::vector<cv::Point3f> pokemon_vertices;
    std::vector<std::pair<int, int>> pokemon_edges;
    bool pokemon_mesh_ready = false;
    bool pokemon_gl_ready = false;

    int model_index = -1;  // -1 = overlay off (manual cycle mode only)
    std::string current_model_name;
    int loaded_dex = -1;
    std::string loaded_model_key;  // dex + form, e.g. "479:heat-rotom"
    // Paths stay valid for the process lifetime (bundled assets or /dev/shm).
    std::map<std::string, ModelEntry> model_cache;

    auto clearPokemonModel = [&]() {
        pokemon_mesh = ObjMesh();
        pokemon_mesh_ready = false;
        pokemon_gl_ready = false;
        current_model_name.clear();
        loaded_dex = -1;
        loaded_model_key.clear();
    };

    auto heightForEntry = [&](ModelEntry entry) -> float {
        resolveModelDex(entry);
        float height_m = -1.f;
        if (entry.dex > 0) {
            height_m = PokemonHeights::heightMetresByDex(entry.dex);
        }
        if (!(height_m > 0.f)) {
            height_m = PokemonHeights::heightMetresByName(entry.name);
        }
        if (!(height_m > 0.f)) {
            printf("warning: no Pokédex height for %s (dex %d); using %.1f m\n",
                   entry.name.c_str(), entry.dex, kDefaultHeightM);
            return kDefaultHeightM;
        }
        return height_m;
    };

    auto loadModelEntry = [&](const ModelEntry& entry) -> bool {
        pokemon_mesh = ObjMesh();
        pokemon_mesh_ready = false;
        pokemon_gl_ready = false;

        const bool is_obj = entry.path.size() > 4 &&
                            entry.path.compare(entry.path.size() - 4, 4, ".obj") == 0;
        const bool loaded = is_obj ? loadObjMesh(entry.path, pokemon_mesh)
                                   : loadAssimpMesh(entry.path, pokemon_mesh);
        if (!loaded || pokemon_mesh.triangles.empty()) {
            printf("Failed to load %s (%s)\n", entry.name.c_str(), entry.path.c_str());
            return false;
        }

        ScaleParams scale_params;
        scale_params.scale_multiplier = scale_multiplier;
        scale_params.y = scale_y;
        scale_params.height_m = heightForEntry(entry);

        ModelPlacement placement =
            placeMeshOnCard(pokemon_mesh, card_w, card_h, scale_params);
        pokemon_vertices = placement.vertices;
        pokemon_edges = buildUniqueEdges(pokemon_mesh);
        pokemon_mesh_ready = true;
        current_model_name = entry.name;

        printf("Loaded %s — %zu tris, %zu materials, %.2f m → %.2f in on card\n",
               entry.name.c_str(), pokemon_mesh.triangles.size(),
               pokemon_mesh.materials.size(), placement.height_m,
               placement.height_inches);

        if (gl_renderer.init(img_size.width, img_size.height)) {
            pokemon_gl_ready =
                gl_renderer.uploadMesh(pokemon_mesh, pokemon_vertices, placement.rotation);
            if (!pokemon_gl_ready) {
                printf("OpenGL upload failed — falling back to wireframe\n");
            }
        } else {
            printf("OpenGL init failed — falling back to wireframe\n");
        }
        return true;
    };

    auto loadModelAt = [&](int index) -> bool {
        if (index < 0 || index >= static_cast<int>(model_library.size())) {
            return false;
        }
        return loadModelEntry(model_library[index]);
    };

    // Resolve a national dex (+ optional card name for forms) to a mesh.
    // Models Resource: local assets first, then scrape. poke-3D: fetch the
    // Pokemon-3D-api GLB, preferring form variants when the card name implies
    // one (Heat Rotom, Alolan Raichu, ...). Cached per dex+form key.
    auto modelCacheKey = [](int dex, const std::string& card_name) {
        std::string key = std::to_string(dex);
        if (!card_name.empty()) {
            key.push_back(':');
            for (unsigned char c : card_name) {
                if (std::isalnum(c)) {
                    key.push_back(static_cast<char>(std::tolower(c)));
                } else if (c == ' ' || c == '-' || c == '_') {
                    if (key.back() != ':') {
                        key.push_back('-');
                    }
                }
            }
        }
        return key;
    };

    auto loadModelForDex = [&](int dex, const std::string& card_name = "") -> bool {
        if (dex <= 0) {
            clearPokemonModel();
            return false;
        }
        const std::string cache_key = modelCacheKey(dex, card_name);
        if (cache_key == loaded_model_key && pokemon_mesh_ready) {
            return true;
        }

        ModelEntry entry;
        const auto cached = model_cache.find(cache_key);
        if (cached != model_cache.end()) {
            entry = cached->second;
        } else if (use_poke3d) {
            std::vector<ModelEntry> fetched;
            std::string error;
            printf("Fetching poke-3D model for dex #%d", dex);
            if (!card_name.empty()) {
                printf(" (%s)", card_name.c_str());
            }
            printf("...\n");
            if (!poke3d_downloader.downloadByDex(dex, fetched, error, card_name) ||
                fetched.empty()) {
                printf("No poke-3D model for dex #%d: %s\n", dex, error.c_str());
                clearPokemonModel();
                return false;
            }
            entry = fetched.front();
            model_cache[cache_key] = entry;
        } else {
            entry = findLocalModelByDex(dex, asset_roots);
            if (entry.path.empty()) {
                std::vector<ModelEntry> fetched;
                std::string error;
                printf("Fetching model for dex #%d...\n", dex);
                if (!downloader.downloadByDex(dex, fetched, error) || fetched.empty()) {
                    printf("No model for dex #%d: %s\n", dex, error.c_str());
                    clearPokemonModel();
                    return false;
                }
                entry = fetched.front();
            } else {
                printf("Using bundled model for dex #%d (%s)\n", dex, entry.name.c_str());
            }
            model_cache[cache_key] = entry;
        }

        // Prefer the requested national dex over whatever the archive named itself.
        entry.dex = dex;
        if (!loadModelEntry(entry)) {
            clearPokemonModel();
            return false;
        }
        loaded_dex = dex;
        loaded_model_key = cache_key;
        return true;
    };

    // 'm' walks the library and then wraps back through "off".
    auto advanceModel = [&]() {
        if (model_library.empty()) {
            model_mode = !model_mode;
            printf("model_mode = %s (pyramid)\n", model_mode ? "on" : "off");
            return;
        }
        const int count = static_cast<int>(model_library.size());
        for (int step = 0; step < count + 1; step++) {
            model_index = (model_index + 1 > count - 1) ? -1 : model_index + 1;
            if (model_index < 0) {
                model_mode = false;
                clearPokemonModel();
                printf("model_mode = off\n");
                return;
            }
            if (loadModelAt(model_index)) {
                model_mode = true;
                printf("model [%d/%d] = %s\n", model_index + 1, count, current_model_name.c_str());
                return;
            }
        }
        model_mode = false;
        printf("No loadable models\n");
    };

    for (;;) {
        *capdev >> frame;
        if (frame.empty()) {
            printf("frame is empty\n");
            break;
        }

        cv::resize(frame, frame, img_size, 0, 0, cv::INTER_AREA);

        // Convert to greyscale
        cv::Mat grey;
        cv::cvtColor(frame, grey, cv::COLOR_BGR2GRAY);

        std::vector<cv::Point2f> corner_set;
        bool card_found = false;
        from_track = false;

        // 1) Try KLT coast if we already have a lock.
        std::vector<cv::Point2f> lk_corners;
        bool lk_live[4] = {false, false, false, false};
        bool lk_ok = tracking &&
                     trackCornersLK(prev_grey, grey, track_corners, lk_corners,
                                    lk_live, allow_off_screen_corners,
                                    lk_win, lk_max_level,
                                    lk_max_fb_err_px, lk_max_jump_px,
                                    lk_max_rigid_residual_px);
        // Drop geometrically insane tracks (corners flew onto the case, etc.).
        if (lk_ok && !trackedQuadSane(track_corners, lk_corners)) {
            lk_ok = false;
            lk_corners.clear();
            lk_live[0] = lk_live[1] = lk_live[2] = lk_live[3] = false;
        }

        int lk_live_count = 0;
        for (int i = 0; i < 4; i++) {
            if (lk_live[i]) {
                lk_live_count++;
            }
        }
        // Partial coast (only with --allow-off-screen-corners on): off-screen
        // or occluded corners are rigid-extrapolated. Keep KLT lock; only adopt
        // detects that share the live anchors. Always refresh Edges while
        // detecting (do not freeze the edge image).
        const bool coasting_partial =
            allow_off_screen_corners && lk_ok && lk_live_count >= 2 &&
            lk_live_count < 4;
        if (coasting_partial) {
            coasted_offscreen = true;
        }

        bool need_detect = !lk_ok || frames_since_detect >= redetect_every ||
                           coasting_partial;

        std::vector<cv::Point2f> detected;
        bool detect_ok = false;
        if (need_detect) {
            std::vector<DetectedQuad> quads;
            if (detectCardQuads(grey, camera_mat, quads, show_overlays,
                                &parent_quad_mem, parent_holdout_frames)) {
                // Prefer quads that are not the one that just failed to match.
                std::vector<size_t> order;
                order.reserve(quads.size());
                if (skip_match_corners.size() == 4) {
                    for (size_t i = 0; i < quads.size(); i++) {
                        if (maxCornerDist(quads[i].corners, skip_match_corners) >
                            challenger_gate_px) {
                            order.push_back(i);
                        }
                    }
                }
                for (size_t i = 0; i < quads.size(); i++) {
                    if (std::find(order.begin(), order.end(), i) == order.end()) {
                        order.push_back(i);
                    }
                }

                const std::vector<cv::Point2f>* locked = nullptr;
                if (tracking) {
                    if (track_corners.size() == 4) {
                        locked = &track_corners;
                    } else if (smooth_corners.size() == 4) {
                        locked = &smooth_corners;
                    }
                }

                // Detection "score" = contour area. Alternatives must beat the
                // current lock before we bother embedding / switching to them.
                double locked_area = -1.0;
                if (locked != nullptr) {
                    for (const DetectedQuad& q : quads) {
                        if (maxCornerDist(q.corners, *locked) <=
                            challenger_gate_px) {
                            locked_area = std::max(locked_area, q.area);
                        }
                    }
                    if (locked_area < 0.0) {
                        locked_area = std::fabs(cv::contourArea(*locked));
                    }
                }

                bool card_resolved = false;
                bool tried_accept = false;
                const CardnessClassifier* cardness_ptr =
                    cardness_classifier.ready() ? &cardness_classifier : nullptr;
                auto tryAcceptCard = [&](const std::vector<cv::Point2f>& in,
                                         std::vector<cv::Point2f>& out) -> bool {
                    tried_accept = true;
                    out = in;
                    if (!orientCardCorners(frame, grey, out, camera_mat,
                                           smooth_corners,
                                           orient_classifier.ready()
                                               ? &orient_classifier
                                               : nullptr)) {
                        return false;
                    }
                    if (!quadLooksLikeCard(frame, out, cardness_ptr,
                                          cardness_min_margin)) {
                        return false;
                    }
                    card_resolved = true;
                    return true;
                };

                if (coasting_partial) {
                    // Card partially off-screen or a corner is occluded: contour
                    // find may only see a clipped/broken fragment — adopting it
                    // shrinks the quad and PnP runs the mesh away from the camera.
                    // Keep KLT's extrapolated corners for pose. Snap only when a
                    // full on-screen card re-enters near the live anchors and is
                    // not a border-clipped piece (vertices parked on the frame edge
                    // next to still-off-screen KLT corners).
                    const double coast_area =
                        std::fabs(cv::contourArea(lk_corners));
                    for (size_t idx : order) {
                        if (!quadSharesLiveAnchors(quads[idx].corners, lk_corners,
                                                   lk_live, anchor_share_px)) {
                            continue;
                        }
                        if (countCornersOnImage(quads[idx].corners, grey.cols,
                                                grey.rows) < 4) {
                            continue;
                        }
                        if (detectLooksBorderClipped(quads[idx].corners,
                                                     lk_corners, grey.cols,
                                                     grey.rows,
                                                     coast_clip_band_px)) {
                            continue;
                        }
                        if (coast_area > 1.0 &&
                            quads[idx].area < 0.75 * coast_area) {
                            continue;
                        }
                        if (tryAcceptCard(quads[idx].corners, detected)) {
                            detect_ok = true;
                            break;
                        }
                    }
                } else {
                // Hold the lock (no alternative hunt) only once this quad has
                // scored a match — unless --freeze-quad off keeps hunting.
                const bool hold_matched_lock =
                    freeze_quad && have_card_id && locked != nullptr;

                if (hold_matched_lock) {
                    // Matched lock: continuity refine only; otherwise coast on KLT.
                    for (size_t idx : order) {
                        if (maxCornerDist(quads[idx].corners, *locked) <=
                            challenger_gate_px) {
                            if (tryAcceptCard(quads[idx].corners, detected)) {
                                detect_ok = true;
                                break;
                            }
                        }
                    }
                    // Drifted lock (common after partial off-screen coast): the
                    // true card contour is outside the 20px gate. Re-acquire via
                    // same-card embed, else nearest oriented quad within a wider
                    // recovery radius / when the lock itself is incomplete.
                    if (!detect_ok) {
                        const int lock_on =
                            countCornersOnImage(*locked, grey.cols, grey.rows);
                        float best_same_d = 1e9f;
                        float best_geom_d = 1e9f;
                        std::vector<cv::Point2f> best_same;
                        std::vector<cv::Point2f> best_geom;
                        for (size_t idx : order) {
                            std::vector<cv::Point2f> cand;
                            if (!tryAcceptCard(quads[idx].corners, cand)) {
                                continue;
                            }
                            const float d = maxCornerDist(cand, *locked);
                            if (d < best_geom_d) {
                                best_geom_d = d;
                                best_geom = cand;
                            }
                            if (match_mode) {
                                std::vector<CardMatch> hits;
                                if (quadMatchesCard(frame, cand, card_matcher,
                                                    match_min_score, cardness_ptr,
                                                    cardness_min_margin, &hits) &&
                                    hits[0].card_id == current_card.card_id &&
                                    d < best_same_d) {
                                    best_same_d = d;
                                    best_same = cand;
                                }
                            }
                        }
                        if (!best_same.empty()) {
                            detected = best_same;
                            detect_ok = true;
                        } else if (!best_geom.empty() &&
                                   (lock_on < 4 ||
                                    best_geom_d <= recovery_gate_px)) {
                            detected = best_geom;
                            detect_ok = true;
                        }
                    }
                } else if (match_mode) {
                    // Identifying, or freeze-quad off: probe higher-area
                    // alternatives; switch only if one embeds to a gallery card.
                    for (size_t idx : order) {
                        const bool near_lock =
                            locked != nullptr &&
                            maxCornerDist(quads[idx].corners, *locked) <=
                                challenger_gate_px;
                        if (have_card_id && near_lock) {
                            continue;
                        }
                        if (locked != nullptr && !near_lock &&
                            quads[idx].area <= locked_area) {
                            continue;
                        }
                        std::vector<cv::Point2f> cand;
                        if (!tryAcceptCard(quads[idx].corners, cand)) {
                            continue;
                        }
                        if (quadMatchesCard(frame, cand, card_matcher,
                                            match_min_score, cardness_ptr,
                                            cardness_min_margin)) {
                            detected = cand;
                            detect_ok = true;
                            skip_match_corners.clear();
                            break;
                        }
                    }

                    // Continuity: stay on the locked quad when it did not just
                    // fail a match (or when we have an id and are holding pose).
                    if (!detect_ok && locked != nullptr) {
                        const bool locked_failed =
                            skip_match_corners.size() == 4 &&
                            maxCornerDist(*locked, skip_match_corners) <=
                                challenger_gate_px;
                        if (have_card_id || !locked_failed) {
                            for (size_t idx : order) {
                                if (maxCornerDist(quads[idx].corners, *locked) <=
                                    challenger_gate_px) {
                                    if (tryAcceptCard(quads[idx].corners, detected)) {
                                        detect_ok = true;
                                        break;
                                    }
                                }
                            }
                        }
                    }

                    // No id yet and current lock failed match: switch only to a
                    // higher-area preferred quad before retrying the failed one.
                    // Skip any candidate cardness rejects as a non-card panel.
                    if (!detect_ok && locked != nullptr && !have_card_id &&
                        !order.empty()) {
                        const bool locked_failed =
                            skip_match_corners.size() == 4 &&
                            maxCornerDist(*locked, skip_match_corners) <=
                                challenger_gate_px;
                        if (locked_failed) {
                            for (size_t idx : order) {
                                if (quads[idx].area <= locked_area) {
                                    continue;
                                }
                                if (tryAcceptCard(quads[idx].corners, detected)) {
                                    detect_ok = true;
                                    break;
                                }
                            }
                        }
                    }

                    // Cold start: first preferred quad that passes cardness.
                    if (!detect_ok && locked == nullptr && !order.empty()) {
                        for (size_t idx : order) {
                            if (tryAcceptCard(quads[idx].corners, detected)) {
                                detect_ok = true;
                                break;
                            }
                        }
                    }
                } else {
                    // No matcher: closest to lock when freezing, else largest.
                    if (locked != nullptr && freeze_quad) {
                        for (size_t idx : order) {
                            if (maxCornerDist(quads[idx].corners, *locked) <=
                                challenger_gate_px) {
                                if (tryAcceptCard(quads[idx].corners, detected)) {
                                    detect_ok = true;
                                    break;
                                }
                            }
                        }
                        if (!detect_ok) {
                            const int lock_on =
                                countCornersOnImage(*locked, grey.cols, grey.rows);
                            float best_d = 1e9f;
                            std::vector<cv::Point2f> best;
                            for (size_t idx : order) {
                                std::vector<cv::Point2f> cand;
                                if (!tryAcceptCard(quads[idx].corners, cand)) {
                                    continue;
                                }
                                const float d = maxCornerDist(cand, *locked);
                                if (d < best_d) {
                                    best_d = d;
                                    best = cand;
                                }
                            }
                            if (!best.empty() &&
                                (lock_on < 4 || best_d <= recovery_gate_px)) {
                                detected = best;
                                detect_ok = true;
                            }
                        }
                    } else {
                        // Prefer nearest lock (if any), else area order. Advance
                        // past quads that fail cardness (or upright labeling).
                        std::vector<size_t> pick_order = order;
                        if (locked != nullptr) {
                            std::sort(pick_order.begin(), pick_order.end(),
                                      [&](size_t a, size_t b) {
                                          return maxCornerDist(quads[a].corners,
                                                               *locked) <
                                                 maxCornerDist(quads[b].corners,
                                                               *locked);
                                      });
                        }
                        for (size_t idx : pick_order) {
                            if (tryAcceptCard(quads[idx].corners, detected)) {
                                detect_ok = true;
                                break;
                            }
                        }
                    }
                }
                }  // !coasting_partial

                // CNN never went super-confident upright on any candidate.
                // If KLT is still healthy (e.g. card half off-screen), keep
                // coasting — do not blank the 3D overlay.
                // Cardness never cleared the bar on any candidate.
                // If KLT is still healthy (e.g. card half off-screen), keep
                // coasting — do not blank the 3D overlay.
                if (tried_accept && !card_resolved && !lk_ok) {
                    tracking = false;
                    track_corners.clear();
                    smooth_corners.clear();
                    prev_grey.release();
                    orientation_confident = false;
                }
            }
        }

        // Fresh full-quad detect after an off-screen coast → reset pose + mesh.
        bool reacquire_after_coast = false;

        if (lk_ok && detect_ok) {
            // Both available. KLT slowly drifts even when the camera is still, so
            // "prefer track when SSD is large" causes green/cyan flip-flopping.
            // Rule: if motion is tiny (still) OR detect agrees with track → detect.
            // While coasting (off-screen / occluded), keep extrapolated KLT
            // corners unless a vetted full-card redetect exists (border-clipped
            // fragments must not replace the coast — they shrink pose depth).
            // Only keep track when moving fast, detect disagrees moderately
            // (typical blur ghost), and all 4 track corners are still live.
            float ssd = cornerSsd(detected, lk_corners);
            float motion = 0.0f;
            for (int i = 0; i < 4; i++) {
                float dx = lk_corners[i].x - track_corners[i].x;
                float dy = lk_corners[i].y - track_corners[i].y;
                motion = std::max(motion, std::sqrt(dx * dx + dy * dy));
            }
            const int lk_on = allow_off_screen_corners ? lk_live_count : 4;
            const bool coasting_incomplete = lk_on < 4;
            const bool detect_full_on =
                countCornersOnImage(detected, grey.cols, grey.rows) == 4;
            const bool take_detect =
                (!coasting_incomplete &&
                 (motion <= still_motion_px || ssd <= redetect_accept_ssd ||
                  ssd >= redetect_snap_ssd)) ||
                (coasting_incomplete && detect_full_on &&
                 !detectLooksBorderClipped(detected, lk_corners, grey.cols,
                                           grey.rows, coast_clip_band_px) &&
                 std::fabs(cv::contourArea(detected)) >=
                     0.75 * std::fabs(cv::contourArea(lk_corners)));
            if (take_detect) {
                corner_set = detected;
                from_track = false;
                frames_since_detect = 0;
                orientation_confident = true;
                if (coasted_offscreen && detect_full_on) {
                    reacquire_after_coast = true;
                }
            } else {
                corner_set = lk_corners;
                from_track = true;
                frames_since_detect++;
            }
            card_found = true;
        } else if (lk_ok) {
            corner_set = lk_corners;
            from_track = true;
            card_found = true;
            frames_since_detect++;
        } else if (detect_ok) {
            corner_set = detected;
            from_track = false;
            card_found = true;
            frames_since_detect = 0;
            orientation_confident = true;
            if (coasted_offscreen &&
                countCornersOnImage(detected, grey.cols, grey.rows) == 4) {
                reacquire_after_coast = true;
            }
        }

        if (card_found) {
            // Raw detect/LK corners for next-frame KLT — never feed EMA back in
            // (that lagged the patches and made cyan under-follow motion).
            track_corners = corner_set;
            tracking = true;
            grey.copyTo(prev_grey);

            // Keep parent memory pinned to the live lock so a brief missed outer
            // detect does not age out and free the text-box panel.
            parentMemoryObserve(parent_quad_mem, track_corners,
                                std::fabs(cv::contourArea(track_corners)));
            parentMemoryPruneNested(parent_quad_mem);

            float corner_jump = 0.0f;
                if (reacquire_after_coast || smooth_corners.size() != 4) {
                // Snap hard after off-screen re-acquire — blending with the
                // coasted EMA leaves a bloated quad / wrong PnP depth.
                smooth_corners = corner_set;
                have_last_pose = false;
                coasted_offscreen = false;
                if (model_mode) {
                    if (match_mode && have_card_id && current_card.dex > 0) {
                        // Force placeMeshOnCard + GL upload again even if same dex.
                        loaded_model_key.clear();
                        pokemon_mesh_ready = false;
                        if (ensureIntrinsics() &&
                            loadModelForDex(current_card.dex, current_card.name)) {
                            model_mode = true;
                            printf("re-acquired quad — reloaded %s\n",
                                   current_model_name.c_str());
                        }
                    } else if (!match_mode && model_index >= 0) {
                        if (loadModelAt(model_index)) {
                            printf("re-acquired quad — reloaded %s\n",
                                   current_model_name.c_str());
                        }
                    }
                }
            } else {
                for (int i = 0; i < 4; i++) {
                    float dx = corner_set[i].x - smooth_corners[i].x;
                    float dy = corner_set[i].y - smooth_corners[i].y;
                    float d = std::sqrt(dx * dx + dy * dy);
                    if (d > corner_jump) {
                        corner_jump = d;
                    }
                }
                float t = std::min(1.0f, corner_jump / speed_ref_px);
                float alpha = alpha_min + t * (alpha_max - alpha_min);
                for (int i = 0; i < 4; i++) {
                    smooth_corners[i] =
                        alpha * corner_set[i] +
                        (1.0f - alpha) * smooth_corners[i];
                }
            }
            // EMA is display/pose only.
            corner_set = smooth_corners;
            // Large corner jump while we still have an id (e.g. another card
            // snapped into the lock) — rematch immediately instead of waiting.
            if (match_mode && have_card_id && corner_jump >= 25.0f) {
                frames_since_match = match_every;
            }
        } else {
            tracking = false;
            track_corners.clear();
            smooth_corners.clear();
            prev_grey.release();
            frames_since_detect = 0;
            skip_match_corners.clear();
            orientation_confident = false;
            have_last_pose = false;
            coasted_offscreen = false;
        }

        if (match_mode) {
            if (!card_found) {
                // The next card to enter frame must not inherit these votes.
                match_history.clear();
                match_candidates.clear();
                have_card_id = false;
                rectified_card.release();
                gallery_card.release();
                gallery_card_id.clear();
                skip_match_corners.clear();
                if (loaded_dex > 0) {
                    clearPokemonModel();
                    model_mode = false;
                }
            } else {
                // Match every N frames while locked; every frame while identifying
                // so a card swap does not sit on a blank/stale overlay longer than
                // needed.
                const int match_interval = have_card_id ? match_every : 1;
                if (++frames_since_match >= match_interval) {
                    frames_since_match = 0;
                    rectified_card = rectifyCard(frame, corner_set,
                                                 card_matcher.inputWidth(),
                                                 card_matcher.inputHeight());
                    const bool looks_like_card =
                        !rectified_card.empty() &&
                        (!cardness_classifier.ready() ||
                         cardness_classifier.scoreCard(rectified_card) >=
                             cardness_min_margin);
                    if (looks_like_card &&
                        card_matcher.match(rectified_card, 5, match_candidates) &&
                        !match_candidates.empty() &&
                        match_candidates[0].score >= match_min_score) {
                        skip_match_corners.clear();
                        const CardMatch& top = match_candidates[0];
                        // Challenger top-1 while still locked: drop votes only when
                        // the challenger *id* changes (first sighting, or swap to a
                        // different card). Keep accumulating the same challenger so
                        // it can reach match_min_votes. Old id/mesh stay until then
                        // (sticky lock — low-angle flicker must not blank the overlay).
                        if (have_card_id && top.card_id != current_card.card_id &&
                            (match_history.empty() ||
                             match_history.back().card_id != top.card_id)) {
                            match_history.clear();
                        }
                        match_history.push_back(top);
                        while (match_history.size() > match_history_len) {
                            match_history.pop_front();
                        }
                        const bool had_id = have_card_id;
                        const int prev_dex = have_card_id ? current_card.dex : -1;
                        const std::string prev_card_id =
                            have_card_id ? current_card.card_id : "";
                        CardMatch voted;
                        if (voteMatch(match_history, match_min_votes, voted)) {
                            current_card = voted;
                            have_card_id = true;
                            if (current_card.card_id != gallery_card_id) {
                                gallery_card = loadGalleryScan(current_card.card_id);
                                gallery_card_id = current_card.card_id;
                                if (gallery_card.empty()) {
                                    printf("no gallery image for %s under data/tcg/en/images/\n",
                                           current_card.card_id.c_str());
                                }
                            }
                            // Swap the 3D overlay whenever the voted species or form
                            // changes (Heat Rotom vs Wash Rotom share a dex but need
                            // different GLBs).
                            const std::string want_key =
                                modelCacheKey(current_card.dex, current_card.name);
                            if (!had_id || current_card.dex != prev_dex ||
                                current_card.card_id != prev_card_id ||
                                want_key != loaded_model_key) {
                                if (current_card.dex > 0) {
                                    if (ensureIntrinsics() &&
                                        loadModelForDex(current_card.dex,
                                                        current_card.name)) {
                                        model_mode = true;
                                    } else {
                                        model_mode = false;
                                    }
                                } else {
                                    // Trainer / Energy / no-dex cards: identify, no mesh.
                                    clearPokemonModel();
                                    model_mode = false;
                                }
                            }
                        } else if (!had_id) {
                            // Still acquiring — no id yet, nothing to hold.
                        }
                        // else: vote failed but we already have an id and the quad
                        // is still locked (detect or KLT) — keep current_card + mesh.
                    } else if (!have_card_id) {
                        // No lock yet and this crop missed — try a different quad
                        // next. Once locked, a miss must not deprioritize the track.
                        skip_match_corners =
                            (track_corners.size() == 4) ? track_corners : corner_set;
                    }
                }
            }

            if (!rectified_card.empty()) {
                cv::imshow("Card", rectified_card);
            }
            if (!gallery_card.empty()) {
                // Same size as the live warp so the two windows are easy to compare.
                cv::Mat shown;
                cv::resize(gallery_card, shown,
                           cv::Size(card_matcher.inputWidth(), card_matcher.inputHeight()),
                           0, 0, cv::INTER_AREA);
                cv::imshow("Match", shown);
            }
        }

        if (match_mode && card_found) {
            char label[256];
            cv::Scalar color(0, 200, 255);
            if (have_card_id) {
                if (current_card.dex > 0) {
                    snprintf(label, sizeof(label), "%s  #%d  [%s]  %.2f",
                             current_card.name.c_str(), current_card.dex,
                             current_card.card_id.c_str(), current_card.score);
                } else {
                    // Trainers / Energy / cards whose download skipped --with-dex.
                    snprintf(label, sizeof(label), "%s  [%s]  %.2f",
                             current_card.name.c_str(), current_card.card_id.c_str(),
                             current_card.score);
                }
                color = cv::Scalar(0, 255, 0);
            } else {
                snprintf(label, sizeof(label), "identifying...");
            }
            const cv::Point at(10, 60);
            cv::putText(frame, label, at, cv::FONT_HERSHEY_SIMPLEX, 0.6,
                        cv::Scalar(0, 0, 0), 3);
            cv::putText(frame, label, at, cv::FONT_HERSHEY_SIMPLEX, 0.6, color, 1);
        }

        char key = (char)cv::waitKey(10);
        if (key == 'q') {
            break;
        }
        if (key == 'o') {
            show_overlays = !show_overlays;
            printf("overlays = %s\n", show_overlays ? "on" : "off");
        }
        // Print the ranked gallery candidates behind the displayed answer.
        if (key == 'c') {
            if (!match_mode) {
                printf("card matching is off — restart with --match on\n");
            } else if (match_candidates.empty()) {
                printf("no candidates yet — hold a card in view\n");
            } else {
                printf("top %zu candidates:\n", match_candidates.size());
                for (size_t i = 0; i < match_candidates.size(); i++) {
                    const CardMatch& entry = match_candidates[i];
                    if (entry.dex > 0) {
                        printf("  %.3f  %-28s #%d  %s (%s)\n", entry.score,
                               entry.name.c_str(), entry.dex, entry.card_id.c_str(),
                               entry.set_name.c_str());
                    } else {
                        printf("  %.3f  %-28s %s (%s)\n", entry.score, entry.name.c_str(),
                               entry.card_id.c_str(), entry.set_name.c_str());
                    }
                }
            }
        }
        // Print corners
        if (key == 's') {
            if (card_found) {
                printf("corners (TL TR BR BL):\n");
                for (int i = 0; i < 4; i++) {
                    printf("  %d: (%.2f, %.2f)\n", i,
                           corner_set[i].x, corner_set[i].y);
                }
            } else {
                printf("no card quad found\n");
            }
        }

        // Toggle pose/axes
        if (key == 'p') {
            if (ensureIntrinsics()) {
                pose_mode = !pose_mode;
                printf("pose_mode = %s\n", pose_mode ? "on" : "off");
            }
        }
        // Cycle the overlay model, or in match mode just toggle the dex-driven mesh.
        if (key == 'm') {
            if (ensureIntrinsics()) {
                if (match_mode) {
                    model_mode = !model_mode;
                    printf("model_mode = %s", model_mode ? "on" : "off");
                    if (model_mode && loaded_dex > 0) {
                        printf(" (dex #%d %s)", loaded_dex, current_model_name.c_str());
                    }
                    printf("\n");
                } else if (use_pokemon) {
                    advanceModel();
                } else {
                    model_mode = !model_mode;
                    printf("model_mode = %s (pyramid)\n", model_mode ? "on" : "off");
                }
            }
        }

        if (card_found && show_overlays) {
            // Green = fresh detect, cyan = KLT track. Off-screen corners stay in
            // corner_set for pose but are not drawn.
            const cv::Scalar quad_color = from_track ? cv::Scalar(255, 255, 0)
                                                    : cv::Scalar(0, 255, 0);
            for (int i = 0; i < 4; i++) {
                const cv::Point2f& a = corner_set[i];
                const cv::Point2f& b = corner_set[(i + 1) % 4];
                cv::line(frame, a, b, quad_color, 2);
                if (!allow_off_screen_corners ||
                    pointOnImage(a, frame.cols, frame.rows)) {
                    cv::circle(frame, a, 5, cv::Scalar(0, 0, 255), -1);
                    char label[8];
                    snprintf(label, sizeof(label), "%d", i);
                    cv::putText(frame, label, a + cv::Point2f(6, -6),
                                cv::FONT_HERSHEY_SIMPLEX, 0.5,
                                cv::Scalar(255, 255, 0), 1);
                }
            }
            cv::putText(frame, from_track ? "T" : "D", cv::Point(10, 30),
                        cv::FONT_HERSHEY_SIMPLEX, 1.0, quad_color, 2);
        }

        if ((pose_mode || model_mode) && card_found && intrinsics_loaded) {
            // Seed with the previous pose when available so ITERATIVE stays near
            // the last spatial solution instead of flipping to a far ambiguous one.
            if (have_last_pose) {
                last_rvec.copyTo(rvec);
                last_tvec.copyTo(tvec);
            }
            bool solved = cv::solvePnP(card_object_pts, corner_set,
                                       camera_mat, dist_coeffs,
                                       rvec, tvec, have_last_pose,
                                       cv::SOLVEPNP_ITERATIVE);
            if (solved) {
                // While corners are off-screen, reject depth explosions that
                // still slip through (clipped quads / bad extrapolations). Hold
                // the last good pose so the mesh stays in place; off-image
                // corners are fine — the model may draw past the frame edge.
                const bool coasting_pose =
                    allow_off_screen_corners &&
                    countCornersOnImage(corner_set, frame.cols, frame.rows) < 4;
                bool pose_ok = true;
                if (have_last_pose && coasting_pose) {
                    const double z = std::fabs(tvec.at<double>(2));
                    const double z0 = std::fabs(last_tvec.at<double>(2));
                    if (z0 > 1e-3 && (z > z0 * 1.25 || z < z0 / 1.25)) {
                        last_rvec.copyTo(rvec);
                        last_tvec.copyTo(tvec);
                        pose_ok = false;
                    }
                }
                if (pose_ok) {
                    rvec.copyTo(last_rvec);
                    tvec.copyTo(last_tvec);
                    have_last_pose = true;
                }
                if (pose_mode) {
                    cv::drawFrameAxes(frame, camera_mat, dist_coeffs, rvec, tvec, 1.0f);
                }
                // Hold the mesh until the CNN locks an upright corner order.
                if (model_mode && orientation_confident) {
                    if (overlay_pokemon && pokemon_mesh_ready) {
                        if (pokemon_gl_ready) {
                            gl_renderer.renderOverlay(camera_mat, rvec, tvec, frame);
                        } else {
                            std::vector<cv::Point2f> proj;
                            cv::projectPoints(pokemon_vertices, rvec, tvec,
                                              camera_mat, dist_coeffs, proj);
                            for (const auto& edge : pokemon_edges) {
                                cv::line(frame, proj[edge.first], proj[edge.second],
                                         cv::Scalar(0, 255, 255), 1);
                            }
                        }
                    } else {
                        std::vector<cv::Point2f> proj;
                        cv::projectPoints(pyramid_pts, rvec, tvec, camera_mat,
                                          dist_coeffs, proj);
                        const int thickness = 3;
                        for (size_t i = 0; i < proj.size(); i++) {
                            cv::circle(frame, proj[i], 5, cv::Scalar(0, 0, 255), -1);
                            for (size_t j = i + 1; j < proj.size(); j++) {
                                cv::Scalar color(
                                    (37 * static_cast<int>(i) + 17 * static_cast<int>(j)) % 256,
                                    (59 * static_cast<int>(i) + 23 * static_cast<int>(j)) % 256,
                                    (97 * static_cast<int>(i) + 31 * static_cast<int>(j)) % 256);
                                cv::line(frame, proj[i], proj[j], color, thickness);
                            }
                        }
                    }
                }
            }
        }

        if (model_mode && orientation_confident && overlay_pokemon &&
            !current_model_name.empty()) {
            const cv::Point at(10, img_size.height - 12);
            cv::putText(frame, current_model_name, at,
                        cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 0), 3);
            cv::putText(frame, current_model_name, at,
                        cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 255), 1);
        }

        cv::imshow("Video", frame);
    }

    delete capdev;
    return 0;
}
