// Asset import validation.
//
// For every discovered model this loads the mesh, reports what reached the GPU,
// and renders each material in isolation to count the pixels it contributes.
// Geometry that imports but draws nothing (a bad alpha mask, a texture that
// failed to decode, UVs pointing at empty atlas space) shows up as a FAIL here
// instead of as a missing limb in the AR view.
//
// Usage: ./validate_models [model_name ...] [--pokemon NAME]... [--models SOURCE]
//          [--scale MULT] [--scale-y Y]
//   --pokemon NAME  download NAME and validate it too
//   --models models-resource|poke-3D  (default models-resource)
//   --scale / --scale-y  same as ar_card (Pokédex log-scale)
// Writes a montage per model to build/validation/ and exits non-zero on failure.

#include "gl_renderer.hpp"
#include "model_download.hpp"
#include "model_library.hpp"
#include "obj_loader.hpp"
#include "poke3d_download.hpp"
#include "pokemon_heights.hpp"

#include <opencv2/opencv.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

const int kW = 480;
const int kH = 480;
const float kCardW = 2.5f;
const float kCardH = 3.5f;
const cv::Scalar kBackground(90, 90, 90);

cv::Mat rotX(double t) {
    return (cv::Mat_<double>(3, 3) << 1, 0, 0, 0, cos(t), -sin(t), 0, sin(t), cos(t));
}
cv::Mat rotZ(double t) {
    return (cv::Mat_<double>(3, 3) << cos(t), -sin(t), 0, sin(t), cos(t), 0, 0, 0, 1);
}

struct Camera {
    cv::Mat intrinsics;
    cv::Mat rvec;
    cv::Mat tvec;
};

// Frame the model's card-space bounding box, viewed from in front and slightly above.
Camera frameModel(const std::vector<cv::Point3f>& verts, double yaw_deg) {
    cv::Point3f lo(1e9f, 1e9f, 1e9f), hi(-1e9f, -1e9f, -1e9f);
    for (const cv::Point3f& v : verts) {
        lo.x = std::min(lo.x, v.x); hi.x = std::max(hi.x, v.x);
        lo.y = std::min(lo.y, v.y); hi.y = std::max(hi.y, v.y);
        lo.z = std::min(lo.z, v.z); hi.z = std::max(hi.z, v.z);
    }
    const cv::Point3f c(0.5f * (lo.x + hi.x), 0.5f * (lo.y + hi.y), 0.5f * (lo.z + hi.z));
    const float size = std::max(std::max(hi.x - lo.x, hi.y - lo.y), hi.z - lo.z);

    const double f = 700.0;
    Camera cam;
    cam.intrinsics = cv::Mat::eye(3, 3, CV_64F);
    cam.intrinsics.at<double>(0, 0) = f;
    cam.intrinsics.at<double>(1, 1) = f;
    cam.intrinsics.at<double>(0, 2) = kW / 2.0;
    cam.intrinsics.at<double>(1, 2) = kH / 2.0;

    // Model "up" is -Z in card space, so pitch back rather than forward.
    const cv::Mat R = rotX(-80.0 * CV_PI / 180.0) * rotZ(yaw_deg * CV_PI / 180.0);
    cv::Rodrigues(R, cam.rvec);
    const cv::Mat pc = R * (cv::Mat_<double>(3, 1) << c.x, c.y, c.z);
    const double dist = f * size / (0.85 * kH);
    cam.tvec = (cv::Mat_<double>(3, 1) <<
        -pc.at<double>(0), -pc.at<double>(1), dist - pc.at<double>(2));
    return cam;
}

int countDrawnPixels(const cv::Mat& frame) {
    cv::Mat bg(frame.size(), frame.type(), kBackground);
    cv::Mat diff;
    cv::absdiff(frame, bg, diff);
    cv::cvtColor(diff, diff, cv::COLOR_BGR2GRAY);
    return cv::countNonZero(diff > 6);
}

void label(cv::Mat& img, const std::string& text) {
    cv::putText(img, text, cv::Point(8, 20), cv::FONT_HERSHEY_SIMPLEX, 0.45,
                cv::Scalar(0, 0, 0), 3);
    cv::putText(img, text, cv::Point(8, 20), cv::FONT_HERSHEY_SIMPLEX, 0.45,
                cv::Scalar(255, 255, 255), 1);
}

struct Problem {
    std::string model;
    std::string detail;
};

}  // namespace

int main(int argc, char** argv) {
    std::vector<std::string> wanted;
    std::vector<std::string> to_download;
    std::string models_source = "models-resource";
    float scale_multiplier = kDefaultScaleMultiplier;
    float scale_y = kDefaultScaleY;
    for (int i = 1; i < argc; i++) {
        if (std::string(argv[i]) == "--pokemon" && i + 1 < argc) {
            to_download.push_back(argv[++i]);
        } else if (std::string(argv[i]) == "--models" && i + 1 < argc) {
            models_source = argv[++i];
            if (models_source != "models-resource" && models_source != "poke-3D") {
                printf("--models must be \"models-resource\" or \"poke-3D\"\n");
                return 2;
            }
        } else if (std::string(argv[i]) == "--scale" && i + 1 < argc) {
            scale_multiplier = static_cast<float>(std::atof(argv[++i]));
            if (!(scale_multiplier > 0.f)) {
                printf("--scale must be a positive number\n");
                return 2;
            }
        } else if (std::string(argv[i]) == "--scale-y" && i + 1 < argc) {
            scale_y = static_cast<float>(std::atof(argv[++i]));
            if (!(scale_y > 0.f)) {
                printf("--scale-y must be a positive number\n");
                return 2;
            }
        } else {
            wanted.push_back(argv[i]);
        }
    }

    const bool use_poke3d = (models_source == "poke-3D");
    std::vector<ModelEntry> library;
    if (!use_poke3d) {
        library = discoverModels({"../data/assets", "data/assets", "../../data/assets"});
    }

    ModelDownloader downloader;
    Poke3dDownloader poke3d_downloader;
    std::vector<Problem> problems;
    for (const std::string& name : to_download) {
        std::vector<ModelEntry> fetched;
        std::string error;
        const bool ok = use_poke3d ? poke3d_downloader.download(name, fetched, error)
                                   : downloader.download(name, fetched, error);
        if (ok) {
            for (ModelEntry& entry : fetched) {
                printf("downloaded %s\n", entry.name.c_str());
                wanted.push_back(entry.name);
                library.push_back(std::move(entry));
            }
        } else {
            printf("FAIL: could not fetch \"%s\": %s\n", name.c_str(), error.c_str());
            problems.push_back({name, "download failed: " + error});
        }
    }

    if (library.empty()) {
        if (use_poke3d) {
            printf("No poke-3D models — pass --pokemon NAME\n");
        } else {
            printf("No models discovered under data/assets\n");
        }
        return 2;
    }

    const fs::path out_dir = "validation";
    fs::create_directories(out_dir);

    GLRenderer renderer;
    if (!renderer.init(kW, kH)) {
        printf("OpenGL init failed\n");
        return 2;
    }

    int checked = 0;

    for (const ModelEntry& entry : library) {
        if (!wanted.empty() &&
            std::find(wanted.begin(), wanted.end(), entry.name) == wanted.end()) {
            continue;
        }
        checked++;
        printf("\n=== %s ===\n  %s\n", entry.name.c_str(), entry.path.c_str());

        ObjMesh mesh;
        const bool is_obj = entry.path.size() > 4 &&
                            entry.path.compare(entry.path.size() - 4, 4, ".obj") == 0;
        if (!(is_obj ? loadObjMesh(entry.path, mesh) : loadAssimpMesh(entry.path, mesh))) {
            printf("  FAIL: mesh did not load\n");
            problems.push_back({entry.name, "mesh did not load"});
            continue;
        }
        if (mesh.triangles.empty()) {
            printf("  FAIL: no triangles\n");
            problems.push_back({entry.name, "no triangles"});
            continue;
        }

        // Every vertex should carry a UV, otherwise texturing silently collapses.
        int missing_uv = 0;
        for (const ObjTriangle& tri : mesh.triangles) {
            for (int c = 0; c < 3; c++) {
                if (tri.texcoords[c] < 0 ||
                    tri.texcoords[c] >= static_cast<int>(mesh.texcoords.size())) {
                    missing_uv++;
                }
            }
        }

        const MeshOrientation up = detectOrientation(mesh);
        printf("  native bbox x[%.2f,%.2f] y[%.2f,%.2f] z[%.2f,%.2f]  "
               "up=(%.2f,%.2f,%.2f) ~%c%c native_h=%.2f\n",
               mesh.min_bounds.x, mesh.max_bounds.x,
               mesh.min_bounds.y, mesh.max_bounds.y,
               mesh.min_bounds.z, mesh.max_bounds.z,
               up.up[0], up.up[1], up.up[2],
               up.sign > 0 ? '+' : '-', up.axis == 1 ? 'Y' : 'Z', up.height);

        ModelEntry keyed = entry;
        resolveModelDex(keyed);
        ScaleParams scale_params;
        scale_params.scale_multiplier = scale_multiplier;
        scale_params.y = scale_y;
        float height_m = -1.f;
        if (keyed.dex > 0) {
            height_m = PokemonHeights::heightMetresByDex(keyed.dex);
        }
        if (!(height_m > 0.f)) {
            height_m = PokemonHeights::heightMetresByName(keyed.name);
        }
        if (!(height_m > 0.f)) {
            printf("  WARN: no Pokédex height for %s (dex %d); using %.1f m\n",
                   keyed.name.c_str(), keyed.dex, kDefaultHeightM);
            height_m = kDefaultHeightM;
        }
        scale_params.height_m = height_m;

        const ModelPlacement placement = placeMeshOnCard(mesh, kCardW, kCardH, scale_params);
        printf("  pokedex %.2f m → target %.2f in → height on card = %.2f in "
               "(scale=%.2f y=%.2f dex=%d)\n",
               placement.height_m, placement.target_inches, placement.height_inches,
               scale_multiplier, scale_y, keyed.dex);
        // Log-scale targets sit in [kMinHeightInches, kMaxHeightInches]; a miss
        // usually means orientation failed and native_h was near zero.
        if (placement.height_inches < kMinHeightInches * 0.9f ||
            placement.height_inches > kMaxHeightInches * 1.1f) {
            printf("  FAIL: implausible height on card\n");
            problems.push_back({entry.name, "implausible height on card (" +
                                                std::to_string(placement.height_inches) + " in)"});
        }
        if (!renderer.uploadMesh(mesh, placement.vertices, placement.rotation)) {
            printf("  FAIL: GPU upload failed\n");
            problems.push_back({entry.name, "GPU upload failed"});
            continue;
        }

        const Camera front = frameModel(placement.vertices, 0.0);
        const Camera side = frameModel(placement.vertices, 45.0);

        renderer.setMaterialFilter("");
        cv::Mat full(kH, kW, CV_8UC3, kBackground);
        renderer.renderOverlay(front.intrinsics, front.rvec, front.tvec, full);
        const int full_px = countDrawnPixels(full);
        label(full, entry.name + "  (all)");

        cv::Mat full_side(kH, kW, CV_8UC3, kBackground);
        renderer.renderOverlay(side.intrinsics, side.rvec, side.tvec, full_side);
        label(full_side, "45 deg");

        std::vector<cv::Mat> tiles{full, full_side};

        printf("  %zu tris, %zu materials, %d px drawn\n",
               mesh.triangles.size(), mesh.materials.size(), full_px);
        if (full_px < 500) {
            printf("  FAIL: model renders almost nothing\n");
            problems.push_back({entry.name, "model renders almost nothing"});
        }
        if (missing_uv > 0) {
            printf("  WARN: %d triangle corners without UVs\n", missing_uv);
            problems.push_back({entry.name, std::to_string(missing_uv) +
                                                " triangle corners without UVs"});
        }

        for (const MaterialReport& report : renderer.materialReports()) {
            std::string textures;
            std::vector<std::string> failed_slots;
            for (const TextureSlotReport& slot : report.textures) {
                textures += " " + slot.slot + "=" + slot.status;
                if (slot.status == "missing") {
                    failed_slots.push_back(slot.slot + " (" + slot.path + ")");
                }
            }

            int material_px = -1;
            if (report.drawn && report.triangles > 0) {
                renderer.setMaterialFilter(report.material);
                cv::Mat solo(kH, kW, CV_8UC3, kBackground);
                renderer.renderOverlay(front.intrinsics, front.rvec, front.tvec, solo);
                material_px = countDrawnPixels(solo);
                label(solo, report.material + "  " + std::to_string(material_px) + "px");
                tiles.push_back(solo);
            }

            const char* state = report.triangles == 0 ? "unused"
                                : report.drawn        ? "drawn"
                                                      : "skipped";
            printf("  %-22s tris=%-6d %s%s%s px=%s%s\n",
                   report.material.c_str(),
                   report.triangles,
                   state,
                   report.additive ? " additive" : "",
                   report.has_alpha ? " alpha" : "",
                   material_px < 0 ? "n/a" : std::to_string(material_px).c_str(),
                   textures.c_str());

            for (const std::string& slot : failed_slots) {
                problems.push_back(
                    {entry.name, report.material + ": " + slot + " failed to load"});
            }
            // Geometry that draws nothing at all is the failure mode that hid
            // Venusaur's trunk. Particle sheets are exempt: they are keyed to
            // moves the idle pose never plays, so being invisible is normal.
            if (material_px == 0) {
                if (report.additive) {
                    printf("    (inert particle sheet)\n");
                } else if (report.has_alpha) {
                    // Iris decals that lost the coverage fight, or fully
                    // transparent leftover quads -- not a missing body part.
                    printf("    (transparent; ignored)\n");
                } else {
                    problems.push_back({entry.name, report.material + ": draws 0 pixels"});
                }
            }
        }
        renderer.setMaterialFilter("");

        // Montage: 4 tiles per row.
        const int cols = 4;
        const int rows = static_cast<int>((tiles.size() + cols - 1) / cols);
        cv::Mat sheet(rows * kH, cols * kW, CV_8UC3, cv::Scalar(20, 20, 20));
        for (size_t i = 0; i < tiles.size(); i++) {
            tiles[i].copyTo(sheet(cv::Rect(static_cast<int>(i % cols) * kW,
                                           static_cast<int>(i / cols) * kH, kW, kH)));
        }
        const std::string path = (out_dir / (entry.name + ".png")).string();
        cv::imwrite(path, sheet);
        printf("  montage -> %s\n", path.c_str());
    }

    printf("\n================ summary ================\n");
    printf("checked %d model(s)\n", checked);
    if (problems.empty()) {
        printf("all assets imported cleanly\n");
        return 0;
    }
    printf("%zu problem(s):\n", problems.size());
    for (const Problem& p : problems) {
        printf("  [%s] %s\n", p.model.c_str(), p.detail.c_str());
    }
    return 1;
}
