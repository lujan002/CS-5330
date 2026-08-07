// Luke Jansen
// 7/24/2026
// main script for AR pokemon card-model visualization

#include <opencv2/opencv.hpp>
#include <opencv2/calib3d.hpp>

#include "gl_renderer.hpp"
#include "model_download.hpp"
#include "model_library.hpp"
#include "obj_loader.hpp"

#include <GLFW/glfw3.h>
#ifdef __linux__
#include <X11/Xlib.h>
#endif

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>
#include <algorithm>
#include <cmath>
#include <cstring>

// Place HighGUI windows on the monitor under the mouse (primary as fallback),
// so a multi-monitor desktop doesn't open them on a different screen.
static void placeAppWindows() {
    const int win_w = 640;
    const int win_h = 480;
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
}

// ---------------------------------------------------------------------------
// TODO (later): Skinned OpenGL renderer (bone palette + idle animation)
//   - Keep Assimp FBX skeleton/weights (do not PreTransformVertices)
//   - Advance idle animation time each frame
//
// TODO (later): OCR on card name region -> pick which Pokemon FBX to load
// TODO (later): Scrape Models Resource ZIP URLs to download models
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
        if (cos_ang > 0.35) {
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
                              float rel_tol = 0.25f) {
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
static const double kCannyLow = 20.0;
static const double kCannyHigh = 60.0;

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
static float scoreCardUpright(const cv::Mat& grey,
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
    cv::warpPerspective(grey, warped, Hmat, cv::Size(W, H));

    cv::Mat edges;
    cv::Canny(warped, edges, kCannyLow, kCannyHigh);

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

// Rewrite corners to card-fixed TL,TR,BR,BL.
// Always forces portrait (short 2.5" edge on top). 0/180 from Canny; if unclear, keep last.
static void orientCardCorners(const cv::Mat& grey,
                              std::vector<cv::Point2f>& corners,
                              const cv::Mat& K,
                              const std::vector<cv::Point2f>& prev_corners) {
    if (corners.size() != 4) {
        return;
    }

    const float target = 2.5f / 3.5f;
    const float ambiguous_margin = 0.5f;  // |score0 - score180| below this => unclear

    // Rank cyclic shifts by closeness to portrait aspect (short side = top).
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

    // Two portrait labelings: upright vs 180° (same short-top edge assignment family).
    int shift_a = portrait_shift;
    int shift_b = (portrait_shift + 2) % 4;

    std::vector<cv::Point2f> cand_a, cand_b;
    applyCyclicShift(corners, shift_a, cand_a);
    applyCyclicShift(corners, shift_b, cand_b);

    float score_a = scoreCardUpright(grey, cand_a);
    float score_b = scoreCardUpright(grey, cand_b);

    int chosen = (score_a >= score_b) ? shift_a : shift_b;
    bool clear = std::fabs(score_a - score_b) >= ambiguous_margin;

    // If uprightness is unclear, do not flip 180° — stay with last portrait labeling.
    if (!clear && prev_corners.size() == 4) {
        float ssd_a = cornerSsd(cand_a, prev_corners);
        float ssd_b = cornerSsd(cand_b, prev_corners);
        chosen = (ssd_a <= ssd_b) ? shift_a : shift_b;
    }

    std::vector<cv::Point2f> oriented;
    applyCyclicShift(corners, chosen, oriented);
    corners = oriented;
}

// Sparse pyramidal LK on the 4 card corners with forward-backward consistency.
// Returns true only if all 4 corners track cleanly into the new frame.
static bool trackCornersLK(const cv::Mat& prev_grey,
                           const cv::Mat& grey,
                           const std::vector<cv::Point2f>& prev_corners,
                           std::vector<cv::Point2f>& next_corners,
                           cv::Size lk_win,
                           int lk_max_level,
                           float max_fb_err_px,
                           float max_jump_px) {
    next_corners.clear();
    if (prev_grey.empty() || grey.empty() || prev_corners.size() != 4) {
        return false;
    }

    const cv::TermCriteria criteria(
        cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 30, 0.01);
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

    next_corners.resize(4);
    for (int i = 0; i < 4; i++) {
        if (!status_fwd[i] || !status_back[i]) {
            return false;
        }
        float jx = fwd[i].x - prev_corners[i].x;
        float jy = fwd[i].y - prev_corners[i].y;
        if (jx * jx + jy * jy > max_jump_sq) {
            return false;
        }
        float fbx = back[i].x - prev_corners[i].x;
        float fby = back[i].y - prev_corners[i].y;
        if (fbx * fbx + fby * fby > max_fb_sq) {
            return false;
        }
        if (fwd[i].x < margin || fwd[i].y < margin ||
            fwd[i].x >= grey.cols - margin || fwd[i].y >= grey.rows - margin) {
            return false;
        }
        next_corners[i] = fwd[i];
    }
    return true;
}

// Reject exploded / collapsed KLT quads (common under fast motion when FB/jump
// thresholds are loose and corners latch onto background texture).
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

// Try to find the largest 4-corner contour (the card). Tune thresholds yourself.
static bool detectCardCorners(const cv::Mat& grey,
                              const cv::Mat& camera_mat,
                              std::vector<cv::Point2f>& corners_out,
                              bool draw_edge_quad) {
    corners_out.clear();

    cv::Mat blurred;
    cv::GaussianBlur(grey, blurred, cv::Size(11, 11), 0);

    // --- tune these ---
    double min_area = 1000.0;           // reject tiny quads that aren't cards
    double approx_eps_frac = 0.04;      // approxPolyDP epsilon = frac * perimeter 
                                        // (how aggressively a wiggly contour is simplified to a polygon)
    // ------------------

    // Binary edge image
    cv::Mat edges;
    cv::Canny(blurred, edges, kCannyLow, kCannyHigh);
    // At this point, image is mostly black with thin white edge pixels
    
    // Find edge curves that are connected (filters out wood grains, glare streaks, etc.) 
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(edges, contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);
    // Contours can nest, but we dont care, so use RETR_LIST (flat list)

    // Keep the best 4-sided convex shape that matches card aspect (perspective-correct)
    double best_area = 0.0;
    std::vector<cv::Point> best_approx;
    std::vector<cv::Point> best_contour;

    // For each contour...
    for (size_t i = 0; i < contours.size(); i++) {
        double area = cv::contourArea(contours[i]);
        // Skip if too small
        if (area < min_area) {
            continue;
        }

        // Measure perimeter
        double peri = cv::arcLength(contours[i], true);
        std::vector<cv::Point> approx;
        // Fit a simpler polygon (epsilon 2% of perimeter)
        cv::approxPolyDP(contours[i], approx, approx_eps_frac * peri, true);
        // Keep it only if it has exactly 4 points and is convex
        if (approx.size() == 4 && cv::isContourConvex(approx)) {
            std::vector<cv::Point2f> ordered;
            for (int j = 0; j < 4; j++) {
                ordered.push_back(cv::Point2f((float)approx[j].x, (float)approx[j].y));
            }
            orderQuadCorners(ordered);
            if (!quadHasCardAspect(ordered, camera_mat)) {
                continue;
            }
            // Among card-aspect quads, keep the largest by area
            if (area > best_area) {
                best_area = area;
                best_approx = approx;
                best_contour = contours[i];
            }
        }
    }

    if (best_approx.size() != 4) {
        cv::Mat edges_vis;
        cv::cvtColor(edges, edges_vis, cv::COLOR_GRAY2BGR);
        cv::imshow("Edges", edges_vis);
        return false;
    }

    for (int i = 0; i < 4; i++) {
        corners_out.push_back(cv::Point2f((float)best_approx[i].x,
                                          (float)best_approx[i].y));
    }
    orderQuadCorners(corners_out);
    // Snap cut-corner approx vertices back onto white edge intersections.
    refineQuadByEdgeLines(edges, corners_out, &best_contour);

    // Display edges (white); optional red refined quad.
    cv::Mat edges_vis;
    cv::cvtColor(edges, edges_vis, cv::COLOR_GRAY2BGR);
    if (draw_edge_quad) {
        for (int i = 0; i < 4; i++) {
            cv::line(edges_vis, corners_out[i], corners_out[(i + 1) % 4],
                     cv::Scalar(0, 0, 255), 2);
        }
    }
    cv::imshow("Edges", edges_vis);
    return true;
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
        "Usage: %s [--model pyramid|pokemon] [--scale IN_PER_UNIT] "
        "[--pokemon NAME]... [/dev/videoN]\n"
        "  --scale    inches on the card per XY mesh unit (default %.2f).\n"
        "             XY models are already battle-sized; 1.0 uses that size\n"
        "             as-is. Big Pokemon overhang the card\n"
        "  --pokemon  download NAME from The Models Resource for this run only\n";

    std::string model_choice = "pyramid";
    const char* device = "/dev/video0";
    float inches_per_unit = kDefaultInchesPerUnit;
    std::vector<std::string> requested_downloads;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--model") == 0) {
            if (i + 1 >= argc) {
                printf(usage, argv[0], kDefaultInchesPerUnit);
                return -1;
            }
            model_choice = argv[++i];
            if (model_choice != "pyramid" && model_choice != "pokemon") {
                printf("--model must be \"pyramid\" or \"pokemon\"\n");
                return -1;
            }
        } else if (std::strcmp(argv[i], "--scale") == 0) {
            if (i + 1 >= argc) {
                printf(usage, argv[0], kDefaultInchesPerUnit);
                return -1;
            }
            inches_per_unit = std::atof(argv[++i]);
            if (!(inches_per_unit > 0.f)) {
                printf("--scale must be a positive number\n");
                return -1;
            }
        } else if (std::strcmp(argv[i], "--pokemon") == 0) {
            if (i + 1 >= argc) {
                printf(usage, argv[0], kDefaultInchesPerUnit);
                return -1;
            }
            requested_downloads.push_back(argv[++i]);
        } else if (argv[i][0] == '-') {
            printf("Unknown argument: %s\n", argv[i]);
            printf(usage, argv[0], kDefaultInchesPerUnit);
            return -1;
        } else {
            device = argv[i];
        }
    }

    const bool use_pokemon = (model_choice == "pokemon");

    // Assemble the model list before opening the camera, so a bad --pokemon name or
    // a network problem is reported straight away.
    std::vector<ModelEntry> model_library;
    // Owns the RAM-backed scratch dirs for --pokemon downloads; they are deleted
    // when this goes out of scope at the end of main.
    ModelDownloader downloader;

    if (use_pokemon) {
        model_library = discoverModels({"../data/assets", "data/assets", "../../data/assets"});

        for (const std::string& wanted : requested_downloads) {
            std::vector<ModelEntry> fetched;
            std::string error;
            if (downloader.download(wanted, fetched, error)) {
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

        printf("%zu model(s) available (scale %.2f in/mesh-unit):\n",
               model_library.size(), inches_per_unit);
        for (size_t i = 0; i < model_library.size(); i++) {
            printf("  [%zu] %s\n", i, model_library[i].name.c_str());
        }
        if (model_library.empty()) {
            printf("No models found — 'm' will fall back to the pyramid\n");
        }
    } else if (!requested_downloads.empty()) {
        printf("--pokemon needs --model pokemon; ignoring the requested downloads\n");
    }

    cv::VideoCapture* capdev = new cv::VideoCapture(device, cv::CAP_V4L2);

    if (!capdev->isOpened()) {
        printf("Unable to open video device\n");
        return -1;
    }
    cv::namedWindow("Video", 1);
    cv::namedWindow("Edges", 1);
    placeAppWindows();

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
    bool intrinsics_loaded = false;
    bool pose_mode = false;

    // Prefer project4 calibration if you run from project_final/build/
    const char* intrinsic_paths[] = {
        "camera_intrinsics.yaml",
        "../project4/build/camera_intrinsics.yaml",
        "../../project4/build/camera_intrinsics.yaml",
    };

    printf("Controls: q=quit  p=pose  m=next model  s=corners  o=overlays\n");
    printf("Model: %s  (pass --model pyramid|pokemon)\n", model_choice.c_str());
    printf("Green quad=detect  Cyan quad=KLT track  (tune Canny / LK knobs in main)\n");

    // Adaptive corner EMA: low alpha when still (kill flicker), high when moving (track).
    std::vector<cv::Point2f> smooth_corners;
    const float alpha_min = 0.65f;   // still: more smoothing (lower = more of the previous frame)
    const float alpha_max = 0.85f;   // moving: follow detection (higher = follow the detector more closely)
    const float speed_ref_px = 25.0f; // distance (px) that maps to alpha_max (higher = more sensitive to movement)

    // KLT optical-flow tracking (coasts through motion blur when Canny drops).
    cv::Mat prev_grey;
    std::vector<cv::Point2f> track_corners;
    bool tracking = false;
    const cv::Size lk_win(21, 21);
    const int lk_max_level = 3;
    const float lk_max_fb_err_px = 3.0f;   // forward-backward consistency
    const float lk_max_jump_px = 80.0f;    // reject wild per-corner jumps
    const int redetect_every = 1;        // periodic detect while tracking (kill drift)
    const float redetect_accept_ssd = 40.0f * 40.0f * 4.0f;  // prefer detect if close to track
    // If KLT moved less than this (px), treat as still → always trust detect over drifted track.
    const float still_motion_px = 3.0f;
    int frames_since_detect = 0;
    bool from_track = false;              // true => cyan quad this frame
    bool show_overlays = true;            // Video quad/labels + Edges red outline
    bool model_mode = false;

    // Same simple pyramid as project4 (base + apex), in card units; -Z out of card.
    const std::vector<cv::Point3f> pyramid_pts = {
        cv::Point3f(0.25f, 0.75f, -0.5f),
        cv::Point3f(2.25f, 0.75f, -0.5f),
        cv::Point3f(2.25f, 2.75f, -0.5f),
        cv::Point3f(0.25f, 2.75f, -0.5f),
        cv::Point3f(1.25f, 1.75f, -2.0f),
    };

    // OpenGL Pokemon models, discovered under data/assets and cycled with 'm'.
    GLRenderer gl_renderer;
    ObjMesh pokemon_mesh;
    std::vector<cv::Point3f> pokemon_vertices;
    std::vector<std::pair<int, int>> pokemon_edges;
    bool pokemon_mesh_ready = false;
    bool pokemon_gl_ready = false;

    int model_index = -1;  // -1 = overlay off
    std::string current_model_name;

    auto loadModelAt = [&](int index) -> bool {
        if (index < 0 || index >= static_cast<int>(model_library.size())) {
            return false;
        }
        const ModelEntry& entry = model_library[index];

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

        ModelPlacement placement =
            placeMeshOnCard(pokemon_mesh, card_w, card_h, inches_per_unit);
        pokemon_vertices = placement.vertices;
        pokemon_edges = buildUniqueEdges(pokemon_mesh);
        pokemon_mesh_ready = true;
        current_model_name = entry.name;

        printf("Loaded %s — %zu tris, %zu materials, %.2f in on card\n",
               entry.name.c_str(), pokemon_mesh.triangles.size(),
               pokemon_mesh.materials.size(), placement.height_inches);

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
                pokemon_mesh_ready = false;
                pokemon_gl_ready = false;
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
        bool lk_ok = tracking &&
                     trackCornersLK(prev_grey, grey, track_corners, lk_corners,
                                    lk_win, lk_max_level,
                                    lk_max_fb_err_px, lk_max_jump_px);
        // Drop geometrically insane tracks (corners flew onto the case, etc.).
        if (lk_ok && !trackedQuadSane(track_corners, lk_corners)) {
            lk_ok = false;
            lk_corners.clear();
        }

        bool need_detect = !lk_ok ||
                           (frames_since_detect >= redetect_every);

        std::vector<cv::Point2f> detected;
        bool detect_ok = false;
        if (need_detect) {
            detect_ok = detectCardCorners(grey, camera_mat, detected, show_overlays);
            if (detect_ok) {
                orientCardCorners(grey, detected, camera_mat, smooth_corners);
            }
        }

        if (lk_ok && detect_ok) {
            // Both available. KLT slowly drifts even when the camera is still, so
            // "prefer track when SSD is large" causes green/cyan flip-flopping.
            // Rule: if motion is tiny (still) OR detect agrees with track → detect.
            // Only keep track when moving fast AND detect disagrees (blur).
            float ssd = cornerSsd(detected, lk_corners);
            float motion = 0.0f;
            for (int i = 0; i < 4; i++) {
                float dx = lk_corners[i].x - track_corners[i].x;
                float dy = lk_corners[i].y - track_corners[i].y;
                motion = std::max(motion, std::sqrt(dx * dx + dy * dy));
            }
            if (motion <= still_motion_px || ssd <= redetect_accept_ssd) {
                corner_set = detected;
                from_track = false;
                frames_since_detect = 0;
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
        }

        if (card_found) {
            if (smooth_corners.size() != 4) {
                smooth_corners = corner_set;
            } else {
                float max_dist = 0.0f;
                for (int i = 0; i < 4; i++) {
                    float dx = corner_set[i].x - smooth_corners[i].x;
                    float dy = corner_set[i].y - smooth_corners[i].y;
                    float d = std::sqrt(dx * dx + dy * dy);
                    if (d > max_dist) {
                        max_dist = d;
                    }
                }
                float t = std::min(1.0f, max_dist / speed_ref_px);
                float alpha = alpha_min + t * (alpha_max - alpha_min);
                for (int i = 0; i < 4; i++) {
                    smooth_corners[i] =
                        alpha * corner_set[i] +
                        (1.0f - alpha) * smooth_corners[i];
                }
            }
            corner_set = smooth_corners;
            track_corners = corner_set;
            tracking = true;
            grey.copyTo(prev_grey);
        } else {
            tracking = false;
            track_corners.clear();
            smooth_corners.clear();
            prev_grey.release();
            frames_since_detect = 0;
        }

        if (card_found && show_overlays) {
            // Green = fresh detect, cyan = KLT track
            const cv::Scalar quad_color = from_track ? cv::Scalar(255, 255, 0)
                                                    : cv::Scalar(0, 255, 0);
            for (int i = 0; i < 4; i++) {
                cv::line(frame, corner_set[i], corner_set[(i + 1) % 4],
                         quad_color, 2);
                cv::circle(frame, corner_set[i], 5, cv::Scalar(0, 0, 255), -1);
                char label[8];
                snprintf(label, sizeof(label), "%d", i);
                cv::putText(frame, label, corner_set[i] + cv::Point2f(6, -6),
                            cv::FONT_HERSHEY_SIMPLEX, 0.5,
                            cv::Scalar(255, 255, 0), 1);
            }
            cv::putText(frame, from_track ? "T" : "D", cv::Point(10, 30),
                        cv::FONT_HERSHEY_SIMPLEX, 1.0, quad_color, 2);
        }

        char key = (char)cv::waitKey(10);
        if (key == 'q') {
            break;
        }
        if (key == 'o') {
            show_overlays = !show_overlays;
            printf("overlays = %s\n", show_overlays ? "on" : "off");
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
        // Load intrinsics once (shared by pose axes + pyramid).
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
                   "calibrate in project4 first, or copy yaml next to the binary\n");
            return false;
        };

        // Toggle pose/axes
        if (key == 'p') {
            if (ensureIntrinsics()) {
                pose_mode = !pose_mode;
                printf("pose_mode = %s\n", pose_mode ? "on" : "off");
            }
        }
        // Cycle the overlay model (pyramid wireframe, or each Pokemon in turn)
        if (key == 'm') {
            if (ensureIntrinsics()) {
                if (use_pokemon) {
                    advanceModel();
                } else {
                    model_mode = !model_mode;
                    printf("model_mode = %s (pyramid)\n", model_mode ? "on" : "off");
                }
            }
        }

        if ((pose_mode || model_mode) && card_found && intrinsics_loaded) {
            bool solved = cv::solvePnP(card_object_pts, corner_set,
                                       camera_mat, dist_coeffs,
                                       rvec, tvec, false, cv::SOLVEPNP_ITERATIVE);
            if (solved) {
                if (pose_mode) {
                    cv::drawFrameAxes(frame, camera_mat, dist_coeffs, rvec, tvec, 1.0f);
                }
                if (model_mode) {
                    if (use_pokemon && pokemon_mesh_ready) {
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

        if (model_mode && use_pokemon && !current_model_name.empty()) {
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
