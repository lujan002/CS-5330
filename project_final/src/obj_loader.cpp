#include "obj_loader.hpp"

#include <assimp/Importer.hpp>
#include <assimp/postprocess.h>
#include <assimp/scene.h>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <map>
#include <system_error>
#include <iostream>
#include <limits>
#include <sstream>
#include <unordered_set>
#include <vector>
#include <cstdint>

namespace fs = std::filesystem;

namespace {

std::string trim(const std::string& value) {
    const auto start = value.find_first_not_of(" \t\r\n");
    if (start == std::string::npos) {
        return "";
    }
    const auto end = value.find_last_not_of(" \t\r\n");
    return value.substr(start, end - start + 1);
}

std::string parentDirectory(const std::string& path) {
    const std::size_t slash = path.find_last_of("/\\");
    if (slash == std::string::npos) {
        return ".";
    }
    return path.substr(0, slash);
}

int parseIndex(const std::string& token, std::size_t count) {
    const int index = std::stoi(token);
    if (index > 0) {
        return index - 1;
    }
    return static_cast<int>(count) + index;
}

bool parseFaceVertex(const std::string& token, std::size_t v_count, std::size_t vt_count, std::size_t vn_count,
                     int& vi, int& ti, int& ni) {
    std::size_t first_slash = token.find('/');
    if (first_slash == std::string::npos) {
        vi = parseIndex(token, v_count);
        ti = -1;
        ni = -1;
        return vi >= 0 && vi < static_cast<int>(v_count);
    }

    vi = parseIndex(token.substr(0, first_slash), v_count);
    std::size_t second_slash = token.find('/', first_slash + 1);
    if (second_slash == std::string::npos) {
        const std::string vt_token = token.substr(first_slash + 1);
        ti = vt_token.empty() ? -1 : parseIndex(vt_token, vt_count);
        ni = -1;
    } else {
        const std::string vt_token = token.substr(first_slash + 1, second_slash - first_slash - 1);
        ti = vt_token.empty() ? -1 : parseIndex(vt_token, vt_count);
        ni = parseIndex(token.substr(second_slash + 1), vn_count);
    }

    return vi >= 0 && vi < static_cast<int>(v_count);
}

void updateBounds(ObjMesh& mesh, const cv::Point3f& point) {
    mesh.min_bounds.x = std::min(mesh.min_bounds.x, point.x);
    mesh.min_bounds.y = std::min(mesh.min_bounds.y, point.y);
    mesh.min_bounds.z = std::min(mesh.min_bounds.z, point.z);
    mesh.max_bounds.x = std::max(mesh.max_bounds.x, point.x);
    mesh.max_bounds.y = std::max(mesh.max_bounds.y, point.y);
    mesh.max_bounds.z = std::max(mesh.max_bounds.z, point.z);
}

bool fileExists(const std::string& path) {
    std::ifstream file(path);
    return file.good();
}

std::string toLowerAscii(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return value;
}

bool isGrayscaleImage(const std::string& path) {
    if (path.empty()) {
        return false;
    }
    const cv::Mat image = cv::imread(path, cv::IMREAD_UNCHANGED);
    if (image.empty()) {
        return false;
    }
    if (image.channels() < 3) {
        return true;
    }
    std::vector<cv::Mat> channels;
    cv::split(image, channels);
    // poke-3D fire sheets are authored greyscale but often encode with 1-level
    // RGB noise after WebP→PNG (Rapidash FireCoreA mean 130.82/130.81/130.82).
    // Exact channel equality misses them and they render as black lit albedo.
    cv::Mat diff01, diff12;
    cv::absdiff(channels[0], channels[1], diff01);
    cv::absdiff(channels[1], channels[2], diff12);
    double max01 = 0.0;
    double max12 = 0.0;
    cv::minMaxLoc(diff01, nullptr, &max01);
    cv::minMaxLoc(diff12, nullptr, &max12);
    return max01 <= 3.0 && max12 <= 3.0;
}

// Face parts are albedo even when the atlas happens to be colourless, so they
// must never be mistaken for particle sheets (Psyduck's "LIris" is greyscale).
bool isFacePartName(const std::string& name) {
    const std::string lower = toLowerAscii(name);
    for (const char* key : {"eye", "iris", "pupil", "mouth", "tongue", "teeth"}) {
        if (lower.find(key) != std::string::npos) {
            return true;
        }
    }
    return false;
}

// Effect overlays that the games drive with a fire/aura shader. poke-3D keeps
// them as ordinary materials, so the name is the only reliable marker when the
// mesh ships no texture at all (Rotom's "FireMask" shell).
bool isEffectPartName(const std::string& name) {
    const std::string lower = toLowerAscii(name);
    for (const char* key : {"fire", "flame", "smoke", "aura", "glow", "spark"}) {
        if (lower.find(key) != std::string::npos) {
            return true;
        }
    }
    return false;
}

// Pokemon XY convention: a diffuse "<base>1.png" is accompanied by
// "<base>Nor", "<base>Nor_Alpha", "<base>2" and "<base>Mask" siblings, and eyes
// may ship a pre-composited "<base>1_Merged". The "*1Id" maps are deliberately
// ignored: they are authored on a different UV layout than the diffuse atlas
// (512x512 vs 512x256, near-zero island overlap).
void resolvePokemonSiblingTextures(ObjMaterial& material, const std::string& base_dir) {
    if (material.diffuse_texture.empty()) {
        return;
    }

    std::string path = material.diffuse_texture;
    std::replace(path.begin(), path.end(), '\\', '/');

    const std::size_t slash = path.find_last_of('/');
    const std::string dir = (slash == std::string::npos) ? "" : path.substr(0, slash + 1);
    const std::string filename =
        (slash == std::string::npos) ? path : path.substr(slash + 1);

    const std::size_t dot = filename.find_last_of('.');
    if (dot == std::string::npos || dot == 0) {
        return;
    }
    const std::string stem = filename.substr(0, dot);
    const std::string ext = filename.substr(dot);
    if (stem.empty() || stem.back() != '1') {
        return;
    }

    auto sibling = [&](const std::string& relative) -> std::string {
        return resolveAssetPath(base_dir, relative).empty() ? std::string() : relative;
    };

    const std::string base = stem.substr(0, stem.size() - 1);  // strip trailing '1'
    material.normal_texture = sibling(dir + base + "Nor" + ext);
    material.specular_texture = sibling(dir + base + "Nor_Alpha" + ext);
    material.color2_texture = sibling(dir + base + "2" + ext);
    material.alpha_mask_texture = sibling(dir + base + "Mask" + ext);

    const std::string merged = sibling(dir + stem + "_Merged" + ext);
    if (!merged.empty()) {
        material.diffuse_texture = merged;
    }

    // An iris quad is redundant once its eye partner uses the merged texture.
    const std::size_t iris_pos = stem.find("Iris1");
    if (iris_pos != std::string::npos) {
        std::string eye_stem = stem;
        eye_stem.replace(iris_pos, 5, "Eye1_Merged");
        if (!sibling(dir + eye_stem + ext).empty()) {
            material.skip_draw = true;
        }
    }
}

void resolvePokemonTextureSet(ObjMaterial& material, const std::string& base_dir) {
    resolvePokemonSiblingTextures(material, base_dir);
    // Outside the sibling lookup, which bails on rips whose textures are named
    // "*.tga.png" -- exactly the rips whose effect meshes matter most.
    // Body parts always ship a normal map, so a colourless diffuse without one is
    // an unlit intensity map: a particle sheet, not albedo.
    material.is_effect =
        material.normal_texture.empty() &&
        isGrayscaleImage(resolveAssetPath(base_dir, material.diffuse_texture));
}

void resolveAllPokemonTextures(ObjMesh& mesh) {
    for (auto& entry : mesh.materials) {
        resolvePokemonTextureSet(entry.second, mesh.base_dir);
    }
}

// True cutout shapes have both near-transparent and near-opaque texels (iris,
// flame masks). Mid-range film opacity (Greedent's eye atlas stuck around ~150,
// Beedrill's wing Mask at ~93) is not a shape — treating it as alpha just makes
// a flat eye plate look like a darker polygon on the face.
bool diffuseTextureHasCutoutAlpha(const std::string& base_dir, const std::string& relative) {
    const std::string path = resolveAssetPath(base_dir, relative);
    if (path.empty()) {
        return false;
    }
    const cv::Mat image = cv::imread(path, cv::IMREAD_UNCHANGED);
    if (image.empty() || image.channels() != 4) {
        return false;
    }
    double min_a = 0.0;
    double max_a = 0.0;
    cv::Mat alpha;
    cv::extractChannel(image, alpha, 3);
    cv::minMaxLoc(alpha, &min_a, &max_a);
    if (min_a > 25.0 || max_a < 230.0) {
        return false;
    }
    cv::Scalar mean;
    cv::Scalar stddev;
    cv::meanStdDev(alpha, mean, stddev);
    return stddev[0] > 15.0;
}

struct MaterialAabb {
    cv::Point3f lo{
        std::numeric_limits<float>::max(),
        std::numeric_limits<float>::max(),
        std::numeric_limits<float>::max()};
    cv::Point3f hi{
        std::numeric_limits<float>::lowest(),
        std::numeric_limits<float>::lowest(),
        std::numeric_limits<float>::lowest()};
    int triangles = 0;

    void grow(const cv::Point3f& p) {
        lo.x = std::min(lo.x, p.x);
        lo.y = std::min(lo.y, p.y);
        lo.z = std::min(lo.z, p.z);
        hi.x = std::max(hi.x, p.x);
        hi.y = std::max(hi.y, p.y);
        hi.z = std::max(hi.z, p.z);
    }

    float volume() const {
        return std::max(0.f, hi.x - lo.x) * std::max(0.f, hi.y - lo.y) *
               std::max(0.f, hi.z - lo.z);
    }
};

float aabbIou(const MaterialAabb& a, const MaterialAabb& b) {
    const cv::Point3f lo(
        std::max(a.lo.x, b.lo.x), std::max(a.lo.y, b.lo.y), std::max(a.lo.z, b.lo.z));
    const cv::Point3f hi(
        std::min(a.hi.x, b.hi.x), std::min(a.hi.y, b.hi.y), std::min(a.hi.z, b.hi.z));
    const float inter = std::max(0.f, hi.x - lo.x) * std::max(0.f, hi.y - lo.y) *
                        std::max(0.f, hi.z - lo.z);
    const float uni = a.volume() + b.volume() - inter;
    return uni > 1e-12f ? inter / uni : 0.f;
}

// Split a poke-3D eye expression atlas into sclera (bright/desaturated) and
// eyelid (remaining eye paint) cutouts. Atlas field (black padding / face-tint)
// becomes transparent in both.
bool splitEyeSheetLayers(
    ObjMesh& mesh,
    const ObjMaterial& sheet_material,
    std::string& sclera_tex_out,
    std::string& lid_tex_out) {
    if (sheet_material.diffuse_texture.empty()) {
        return false;
    }
    const std::string path =
        resolveAssetPath(mesh.base_dir, sheet_material.diffuse_texture);
    if (path.empty()) {
        return false;
    }
    cv::Mat image = cv::imread(path, cv::IMREAD_UNCHANGED);
    if (image.empty()) {
        return false;
    }
    cv::Mat bgr;
    if (image.channels() == 4) {
        cv::cvtColor(image, bgr, cv::COLOR_BGRA2BGR);
    } else if (image.channels() == 3) {
        bgr = image;
    } else if (image.channels() == 1) {
        cv::cvtColor(image, bgr, cv::COLOR_GRAY2BGR);
    } else {
        return false;
    }

    constexpr int kBins = 16;
    constexpr int kBinSize = 256 / kBins;
    std::vector<int> hist(kBins * kBins * kBins, 0);
    const int total = bgr.rows * bgr.cols;
    for (int y = 0; y < bgr.rows; ++y) {
        const cv::Vec3b* row = bgr.ptr<cv::Vec3b>(y);
        for (int x = 0; x < bgr.cols; ++x) {
            const cv::Vec3b& px = row[x];
            const int bb = std::min(px[0] / kBinSize, kBins - 1);
            const int gg = std::min(px[1] / kBinSize, kBins - 1);
            const int rr = std::min(px[2] / kBinSize, kBins - 1);
            hist[(bb * kBins + gg) * kBins + rr] += 1;
        }
    }

    std::vector<int> order(hist.size());
    for (int i = 0; i < static_cast<int>(order.size()); ++i) {
        order[i] = i;
    }
    std::partial_sort(
        order.begin(),
        order.begin() + std::min(6, static_cast<int>(order.size())),
        order.end(),
        [&](int a, int b) { return hist[a] > hist[b]; });

    auto bin_center = [&](int idx) {
        const int bb = idx / (kBins * kBins);
        const int gg = (idx / kBins) % kBins;
        const int rr = idx % kBins;
        return cv::Vec3i(
            bb * kBinSize + kBinSize / 2,
            gg * kBinSize + kBinSize / 2,
            rr * kBinSize + kBinSize / 2);
    };
    auto is_sclera_like = [](int b, int g, int r) {
        const int max_c = std::max(b, std::max(g, r));
        const int min_c = std::min(b, std::min(g, r));
        const float lum = (b + g + r) / 3.f;
        const float sat = max_c > 0 ? (max_c - min_c) / static_cast<float>(max_c) : 0.f;
        return lum >= 170.f && sat <= 0.40f;
    };

    std::vector<cv::Vec3i> fields;
    for (int n = 0; n < 6 && static_cast<int>(fields.size()) < 2; ++n) {
        const int idx = order[n];
        if (hist[idx] < total / 25) {
            break;
        }
        const cv::Vec3i c = bin_center(idx);
        if (is_sclera_like(c[0], c[1], c[2])) {
            continue;
        }
        fields.push_back(c);
    }

    constexpr int kBgDistSq = 55 * 55;
    cv::Mat sclera(bgr.size(), CV_8UC4);
    cv::Mat lid(bgr.size(), CV_8UC4);
    for (int y = 0; y < bgr.rows; ++y) {
        const cv::Vec3b* row = bgr.ptr<cv::Vec3b>(y);
        cv::Vec4b* sclera_row = sclera.ptr<cv::Vec4b>(y);
        cv::Vec4b* lid_row = lid.ptr<cv::Vec4b>(y);
        for (int x = 0; x < bgr.cols; ++x) {
            const cv::Vec3b& px = row[x];
            bool is_bg = fields.empty();
            for (const cv::Vec3i& bg : fields) {
                const int db = static_cast<int>(px[0]) - bg[0];
                const int dg = static_cast<int>(px[1]) - bg[1];
                const int dr = static_cast<int>(px[2]) - bg[2];
                if (db * db + dg * dg + dr * dr <= kBgDistSq) {
                    is_bg = true;
                    break;
                }
            }
            // Fallback: near-black padding when no field was classified.
            if (fields.empty()) {
                const int lum = (px[0] + px[1] + px[2]) / 3;
                is_bg = lum <= 18;
            }
            const bool is_sclera = !is_bg && is_sclera_like(px[0], px[1], px[2]);
            sclera_row[x] = cv::Vec4b(px[0], px[1], px[2], is_sclera ? 255 : 0);
            lid_row[x] = cv::Vec4b(
                px[0], px[1], px[2], (!is_bg && !is_sclera) ? 255 : 0);
        }
    }

    sclera_tex_out = "eye_sclera_" + sheet_material.name + ".png";
    lid_tex_out = "eye_lid_" + sheet_material.name + ".png";
    if (!cv::imwrite((fs::path(mesh.base_dir) / sclera_tex_out).string(), sclera) ||
        !cv::imwrite((fs::path(mesh.base_dir) / lid_tex_out).string(), lid)) {
        return false;
    }
    return true;
}

// poke-3D ships sclera/lid on one atlas + a separate alpha iris (pupils). XY can
// drop Iris1 when Eye1_Merged already composites lids over pupils; poke-3D's
// sheet does not, so keep the iris between a sclera cutout and a lid cutout.
// No camera-ward vertex bias — that pushed eyes through the brow.
void fixPoke3dEyeLayers(ObjMesh& mesh) {
    std::map<std::string, MaterialAabb> bounds;
    for (std::size_t i = 0; i < mesh.triangles.size(); ++i) {
        const std::string& mat_name = mesh.triangle_materials[i];
        MaterialAabb& box = bounds[mat_name];
        const ObjTriangle& tri = mesh.triangles[i];
        for (int c = 0; c < 3; ++c) {
            const int vi = tri.vertices[c];
            if (vi >= 0 && vi < static_cast<int>(mesh.vertices.size())) {
                box.grow(mesh.vertices[vi]);
            }
        }
        box.triangles += 1;
    }

    std::map<std::string, bool> has_cutout;
    for (const auto& entry : mesh.materials) {
        has_cutout[entry.first] =
            diffuseTextureHasCutoutAlpha(mesh.base_dir, entry.second.diffuse_texture);
    }

    constexpr float kMinIou = 0.90f;
    std::vector<std::string> alpha_names;
    for (const auto& entry : mesh.materials) {
        if (has_cutout[entry.first]) {
            alpha_names.push_back(entry.first);
        }
    }

    for (const std::string& iris_name : alpha_names) {
        auto alpha_it = mesh.materials.find(iris_name);
        if (alpha_it == mesh.materials.end() || alpha_it->second.skip_draw) {
            continue;
        }
        const auto alpha_bounds = bounds.find(iris_name);
        if (alpha_bounds == bounds.end() || alpha_bounds->second.triangles == 0) {
            continue;
        }

        std::string sheet_name;
        for (const auto& opaque_entry : mesh.materials) {
            if (opaque_entry.first == iris_name || has_cutout[opaque_entry.first] ||
                opaque_entry.second.eye_layer != 0) {
                continue;
            }
            const auto opaque_bounds = bounds.find(opaque_entry.first);
            if (opaque_bounds == bounds.end() || opaque_bounds->second.triangles == 0) {
                continue;
            }
            if (aabbIou(alpha_bounds->second, opaque_bounds->second) < kMinIou) {
                continue;
            }
            sheet_name = opaque_entry.first;
            break;
        }
        if (sheet_name.empty()) {
            continue;
        }
        auto sheet_it = mesh.materials.find(sheet_name);
        if (sheet_it == mesh.materials.end()) {
            continue;
        }

        std::string sclera_tex;
        std::string lid_tex;
        if (!splitEyeSheetLayers(mesh, sheet_it->second, sclera_tex, lid_tex)) {
            continue;
        }

        const std::string sclera_name = sheet_name + "__sclera";
        const std::string lid_name = sheet_name + "__lid";

        ObjMaterial sclera_mat = sheet_it->second;
        sclera_mat.name = sclera_name;
        sclera_mat.diffuse_texture = sclera_tex;
        sclera_mat.eye_layer = 1;
        sclera_mat.alpha_clip = false;

        ObjMaterial lid_mat = sheet_it->second;
        lid_mat.name = lid_name;
        lid_mat.diffuse_texture = lid_tex;
        lid_mat.eye_layer = 3;
        lid_mat.alpha_clip = false;

        alpha_it->second.eye_layer = 2;
        alpha_it->second.skip_draw = false;

        mesh.materials[sclera_name] = sclera_mat;
        mesh.materials[lid_name] = lid_mat;
        mesh.materials.erase(sheet_name);

        const std::size_t tri_count = mesh.triangles.size();
        for (std::size_t ti = 0; ti < tri_count; ++ti) {
            if (mesh.triangle_materials[ti] != sheet_name) {
                continue;
            }
            mesh.triangle_materials[ti] = sclera_name;
            // Lid uses the same geometry (no outward bias).
            mesh.triangles.push_back(mesh.triangles[ti]);
            mesh.triangle_materials.push_back(lid_name);
        }

        std::cerr << "Eye fix: sclera/iris/lid '" << sheet_name << "' -> '"
                  << sclera_name << "' + '" << iris_name << "' + '" << lid_name
                  << "'\n";
    }

    // An iris promoted into the eye stack is albedo, never a particle sheet:
    // poke-3D irises are greyscale often enough to trip the effect heuristic,
    // and drawing a pupil additively erases it (Porygon2, Psyduck).
    for (auto& entry : mesh.materials) {
        if (entry.second.eye_layer != 0) {
            entry.second.is_effect = false;
        }
    }
}

// Bounds of the Pokemon proper, so that oversized particle volumes do not drive
// the up-axis detection or the on-card scale.
void computeBodyBounds(ObjMesh& mesh) {
    cv::Point3f lo(std::numeric_limits<float>::max(),
                   std::numeric_limits<float>::max(),
                   std::numeric_limits<float>::max());
    cv::Point3f hi(std::numeric_limits<float>::lowest(),
                   std::numeric_limits<float>::lowest(),
                   std::numeric_limits<float>::lowest());

    for (std::size_t i = 0; i < mesh.triangles.size(); ++i) {
        const auto material = mesh.materials.find(mesh.triangle_materials[i]);
        if (material != mesh.materials.end() && material->second.is_effect) {
            continue;
        }
        for (int corner = 0; corner < 3; ++corner) {
            const int index = mesh.triangles[i].vertices[corner];
            if (index < 0 || index >= static_cast<int>(mesh.vertices.size())) {
                continue;
            }
            const cv::Point3f& p = mesh.vertices[index];
            lo.x = std::min(lo.x, p.x); hi.x = std::max(hi.x, p.x);
            lo.y = std::min(lo.y, p.y); hi.y = std::max(hi.y, p.y);
            lo.z = std::min(lo.z, p.z); hi.z = std::max(hi.z, p.z);
        }
    }

    const bool found = hi.x >= lo.x;
    mesh.body_min_bounds = found ? lo : mesh.min_bounds;
    mesh.body_max_bounds = found ? hi : mesh.max_bounds;
}

bool quatIsAxisAligned(float x, float y, float z, float w, float tol = 0.12f) {
    const float comps[4] = {x, y, z, w};
    for (float c : comps) {
        const float a = std::fabs(c);
        if (a > tol && std::fabs(a - 1.f) > tol &&
            std::fabs(a - 0.70710678f) > tol) {
            return false;
        }
    }
    return true;
}

// True when R maps each basis vector onto a world axis (90°/180° permutations).
// Ponyta's shared root is one of these: after baking, the mesh is already Y-up
// and axis-aligned. Undoing it remaps length onto X and leaves the horse sideways.
bool matIsAxisPermutation(const aiMatrix3x3& R, float tol = 0.15f) {
    for (int col = 0; col < 3; ++col) {
        const float a0 = std::fabs(R[0][col]);
        const float a1 = std::fabs(R[1][col]);
        const float a2 = std::fabs(R[2][col]);
        const float dominant = std::max(a0, std::max(a1, a2));
        if (dominant < 1.f - tol) {
            return false;
        }
    }
    return true;
}

// Breloom / Toxicroak ship every mesh node with the same non-axis-aligned
// rotation, so the character is permanently tilted once transforms are baked.
// Undo that shared root rotation; detectOrientation can then find a real up axis.
bool undoSharedTiltedRoot(
    ObjMesh& mesh,
    const std::vector<aiMatrix4x4>& mesh_transforms) {
    if (mesh_transforms.empty()) {
        return false;
    }
    aiVector3D scale0;
    aiQuaternion quat0;
    aiVector3D pos0;
    mesh_transforms.front().Decompose(scale0, quat0, pos0);
    if (quatIsAxisAligned(quat0.x, quat0.y, quat0.z, quat0.w)) {
        return false;
    }
    // Axis permutations already yield a clean AABB after bake (Ponyta).
    if (matIsAxisPermutation(quat0.GetMatrix())) {
        return false;
    }
    for (const aiMatrix4x4& t : mesh_transforms) {
        aiVector3D scale;
        aiQuaternion quat;
        aiVector3D pos;
        t.Decompose(scale, quat, pos);
        if (std::fabs(quat.x - quat0.x) > 0.02f ||
            std::fabs(quat.y - quat0.y) > 0.02f ||
            std::fabs(quat.z - quat0.z) > 0.02f ||
            std::fabs(quat.w - quat0.w) > 0.02f) {
            return false;
        }
        if (quatIsAxisAligned(quat.x, quat.y, quat.z, quat.w) ||
            matIsAxisPermutation(quat.GetMatrix())) {
            return false;
        }
    }

    aiMatrix3x3 R = quat0.GetMatrix();
    R.Inverse();
    for (cv::Point3f& v : mesh.vertices) {
        const aiVector3D p = R * aiVector3D(v.x, v.y, v.z);
        v = {p.x, p.y, p.z};
    }
    for (cv::Point3f& n : mesh.normals) {
        aiVector3D p = R * aiVector3D(n.x, n.y, n.z);
        p.NormalizeSafe();
        n = {p.x, p.y, p.z};
    }

    // The undo leaves an axis-aligned cloud, but the feet may sit on -Z rather
    // than +Y (Breloom). Pick ±Y/±Z by requiring BOTH ends of the silhouette to
    // carry geometry (feet and head/hat). Maximising base contact alone stood
    // Breloom on its mushroom cap.
    struct Cand {
        int axis = 1;
        float sign = 1.f;
        int both_ends = -1;
        float foot_off = 1e9f;
    };
    Cand best;
    const int grid = 32;
    for (int axis = 1; axis <= 2; ++axis) {
        for (float sign : {1.f, -1.f}) {
            float lo_u = std::numeric_limits<float>::max();
            float hi_u = std::numeric_limits<float>::lowest();
            float lo0 = std::numeric_limits<float>::max();
            float hi0 = std::numeric_limits<float>::lowest();
            float lo1 = std::numeric_limits<float>::max();
            float hi1 = std::numeric_limits<float>::lowest();
            const int a0 = 0;
            const int a1 = axis == 1 ? 2 : 1;
            double sum0 = 0.0;
            double sum1 = 0.0;
            for (const cv::Point3f& v : mesh.vertices) {
                const float c[3] = {v.x, v.y, v.z};
                lo_u = std::min(lo_u, c[axis]);
                hi_u = std::max(hi_u, c[axis]);
                lo0 = std::min(lo0, c[a0]);
                hi0 = std::max(hi0, c[a0]);
                lo1 = std::min(lo1, c[a1]);
                hi1 = std::max(hi1, c[a1]);
                sum0 += c[a0];
                sum1 += c[a1];
            }
            const float span = std::max(hi_u - lo_u, 1e-6f);
            const float base_lim =
                sign > 0.f ? lo_u + 0.08f * span : hi_u - 0.08f * span;
            const float top_lim =
                sign > 0.f ? hi_u - 0.08f * span : lo_u + 0.08f * span;
            const float span0 = std::max(hi0 - lo0, 1e-6f);
            const float span1 = std::max(hi1 - lo1, 1e-6f);
            const double cen0 = sum0 / static_cast<double>(mesh.vertices.size());
            const double cen1 = sum1 / static_cast<double>(mesh.vertices.size());
            std::vector<unsigned char> base_cells(grid * grid, 0);
            std::vector<unsigned char> top_cells(grid * grid, 0);
            int base_occ = 0;
            int top_occ = 0;
            double foot0 = 0.0;
            double foot1 = 0.0;
            int foot_n = 0;
            for (const cv::Point3f& v : mesh.vertices) {
                const float c[3] = {v.x, v.y, v.z};
                const int c0 = std::clamp(
                    static_cast<int>((c[a0] - lo0) / span0 * (grid - 1)), 0, grid - 1);
                const int c1 = std::clamp(
                    static_cast<int>((c[a1] - lo1) / span1 * (grid - 1)), 0, grid - 1);
                const bool in_base =
                    sign > 0.f ? c[axis] <= base_lim : c[axis] >= base_lim;
                const bool in_top =
                    sign > 0.f ? c[axis] >= top_lim : c[axis] <= top_lim;
                if (in_base) {
                    unsigned char& cell = base_cells[c1 * grid + c0];
                    if (!cell) {
                        cell = 1;
                        base_occ++;
                    }
                    foot0 += c[a0];
                    foot1 += c[a1];
                    foot_n++;
                }
                if (in_top) {
                    unsigned char& cell = top_cells[c1 * grid + c0];
                    if (!cell) {
                        cell = 1;
                        top_occ++;
                    }
                }
            }
            const int both = std::min(base_occ, top_occ);
            const float foot_off =
                foot_n > 0
                    ? static_cast<float>(std::sqrt(
                          std::pow(foot0 / foot_n - cen0, 2) +
                          std::pow(foot1 / foot_n - cen1, 2))) /
                          span
                    : 1e9f;
            if (both > best.both_ends ||
                (both == best.both_ends && foot_off < best.foot_off)) {
                best = {axis, sign, both, foot_off};
            }
        }
    }

    if (best.both_ends > 0 && !(best.axis == 1 && best.sign > 0.f)) {
        cv::Vec3f uy(0.f, 0.f, 0.f);
        uy[best.axis] = best.sign;
        cv::Vec3f ux(1.f, 0.f, 0.f);
        cv::Vec3f uz = ux.cross(uy);
        const float len = static_cast<float>(cv::norm(uz));
        if (len > 1e-6f) {
            uz *= 1.f / len;
            ux = uy.cross(uz);
            const cv::Matx33f M(
                ux[0], ux[1], ux[2],
                uy[0], uy[1], uy[2],
                uz[0], uz[1], uz[2]);
            for (cv::Point3f& v : mesh.vertices) {
                const cv::Vec3f p(v.x, v.y, v.z);
                const cv::Vec3f q = M * p;
                v = {q[0], q[1], q[2]};
            }
            for (cv::Point3f& n : mesh.normals) {
                const cv::Vec3f p(n.x, n.y, n.z);
                cv::Vec3f q = M * p;
                const float nlen = static_cast<float>(cv::norm(q));
                if (nlen > 1e-8f) {
                    q *= 1.f / nlen;
                }
                n = {q[0], q[1], q[2]};
            }
        }
    }

    float ymin = std::numeric_limits<float>::max();
    float xmin = std::numeric_limits<float>::max();
    float xmax = std::numeric_limits<float>::lowest();
    float zmin = std::numeric_limits<float>::max();
    float zmax = std::numeric_limits<float>::lowest();
    for (const cv::Point3f& v : mesh.vertices) {
        ymin = std::min(ymin, v.y);
        xmin = std::min(xmin, v.x);
        xmax = std::max(xmax, v.x);
        zmin = std::min(zmin, v.z);
        zmax = std::max(zmax, v.z);
    }
    const cv::Point3f shift(
        -0.5f * (xmin + xmax),
        -ymin,
        -0.5f * (zmin + zmax));
    for (cv::Point3f& v : mesh.vertices) {
        v.x += shift.x;
        v.y += shift.y;
        v.z += shift.z;
    }

    mesh.min_bounds = cv::Point3f(
        std::numeric_limits<float>::max(),
        std::numeric_limits<float>::max(),
        std::numeric_limits<float>::max());
    mesh.max_bounds = cv::Point3f(
        std::numeric_limits<float>::lowest(),
        std::numeric_limits<float>::lowest(),
        std::numeric_limits<float>::lowest());
    for (const cv::Point3f& v : mesh.vertices) {
        updateBounds(mesh, v);
    }
    std::cerr << "Upright: undid shared tilted root rotation\n";
    return true;
}

struct SubMesh {
    const aiMesh* mesh = nullptr;
    aiMatrix4x4 transform;
    std::string name;
    cv::Point3f lo;
    cv::Point3f hi;
    bool skip = false;
};

void collectSubMeshes(
    const aiScene* scene,
    const aiNode* node,
    const aiMatrix4x4& parent,
    std::vector<SubMesh>& out) {
    const aiMatrix4x4 global = parent * node->mTransformation;
    for (unsigned int i = 0; i < node->mNumMeshes; ++i) {
        const aiMesh* aimesh = scene->mMeshes[node->mMeshes[i]];
        SubMesh sub;
        sub.mesh = aimesh;
        sub.transform = global;
        // Prefer the node name: poke-3D puts part labels on nodes
        // (pm0910_00_00_ChigoASkin) while the mesh is often just "Mesh.001".
        sub.name = node->mName.length > 0 ? node->mName.C_Str() : aimesh->mName.C_Str();
        sub.lo = cv::Point3f(std::numeric_limits<float>::max(),
                             std::numeric_limits<float>::max(),
                             std::numeric_limits<float>::max());
        sub.hi = cv::Point3f(std::numeric_limits<float>::lowest(),
                             std::numeric_limits<float>::lowest(),
                             std::numeric_limits<float>::lowest());
        for (unsigned int v = 0; v < aimesh->mNumVertices; ++v) {
            const aiVector3D p = global * aimesh->mVertices[v];
            sub.lo.x = std::min(sub.lo.x, p.x); sub.hi.x = std::max(sub.hi.x, p.x);
            sub.lo.y = std::min(sub.lo.y, p.y); sub.hi.y = std::max(sub.hi.y, p.y);
            sub.lo.z = std::min(sub.lo.z, p.z); sub.hi.z = std::max(sub.hi.z, p.z);
        }
        out.push_back(sub);
    }
    for (unsigned int i = 0; i < node->mNumChildren; ++i) {
        collectSubMeshes(scene, node->mChildren[i], global, out);
    }
}

// "<Pokemon>_<Part>-mesh" sub-meshes that come in lettered variants are alternate
// poses of the same part, and the game's animations choose between them. Gengar
// ships TongueA (tucked in) and TongueB (lolling out); with no animation to pick
// one, drawing both leaves a tongue hanging through its chin.
//
// A shared base name alone is not enough to condemn a part -- Blastoise's
// CannonsA/CannonsB are both real -- and neither is sticking out, since
// Greninja's single Tongue-mesh is a scarf that juts further than Gengar's
// tongue does. Only a part that is BOTH a lettered variant AND outside the
// body's silhouette is treated as a non-idle pose.
void markAlternateExpressionParts(std::vector<SubMesh>& subs) {
    auto part_name = [](const std::string& name) -> std::string {
        const std::size_t underscore = name.find('_');
        if (underscore == std::string::npos) {
            return "";
        }
        std::size_t end = name.find("-mesh", underscore);
        if (end == std::string::npos) {
            end = name.size();
        }
        return name.substr(underscore + 1, end - underscore - 1);
    };
    // "TongueA" -> "Tongue"; only a single trailing letter counts as a variant tag.
    auto variant_base = [](const std::string& part) -> std::string {
        if (part.size() < 2) {
            return "";
        }
        const char last = part.back();
        if (last < 'A' || last > 'Z') {
            return "";
        }
        return part.substr(0, part.size() - 1);
    };

    const SubMesh* body = nullptr;
    for (const SubMesh& sub : subs) {
        if (body == nullptr || sub.mesh->mNumFaces > body->mesh->mNumFaces) {
            body = &sub;
        }
    }
    if (body == nullptr) {
        return;
    }
    const cv::Point3f span = body->hi - body->lo;

    std::map<std::string, int> variant_counts;
    for (const SubMesh& sub : subs) {
        const std::string base = variant_base(part_name(sub.name));
        if (!base.empty()) {
            variant_counts[base] += 1;
        }
    }

    // A quarter of the body's extent clears Blastoise's cannons (16%) while
    // catching Gengar's tongue (42%).
    const float tolerance = 0.25f;
    for (SubMesh& sub : subs) {
        const std::string base = variant_base(part_name(sub.name));
        if (base.empty() || variant_counts[base] < 2) {
            continue;
        }
        const bool outside =
            sub.hi.x - body->hi.x > tolerance * span.x ||
            body->lo.x - sub.lo.x > tolerance * span.x ||
            sub.hi.y - body->hi.y > tolerance * span.y ||
            body->lo.y - sub.lo.y > tolerance * span.y ||
            sub.hi.z - body->hi.z > tolerance * span.z ||
            body->lo.z - sub.lo.z > tolerance * span.z;
        if (outside) {
            sub.skip = true;
        }
    }
}

// poke-3D Greedent ships held-item berry meshes (Chigo/Himeri/Nana *Skin) as
// siblings of BodySkin. With no animation they hang under the feet in a line.
// Keep anatomy skins (TailSkin, WingSkin, …); drop the rest when BodySkin is
// present.
void markHeldItemSkins(std::vector<SubMesh>& subs) {
    bool has_body_skin = false;
    for (const SubMesh& sub : subs) {
        const std::string lower = toLowerAscii(sub.name);
        if (lower.find("bodyskin") != std::string::npos) {
            has_body_skin = true;
            break;
        }
    }
    if (!has_body_skin) {
        return;
    }

    const char* keep_keys[] = {
        "body", "tail", "wing", "hair", "ear", "hand", "foot", "leg", "arm",
        "head", "eye", "mouth", "cloth", "cape", "scarf", "horn", "shell",
        "fin", "tooth", "fang", "claw", "mane", "fur"};
    for (SubMesh& sub : subs) {
        const std::string lower = toLowerAscii(sub.name);
        if (lower.find("skin") == std::string::npos) {
            continue;
        }
        bool keep = false;
        for (const char* key : keep_keys) {
            if (lower.find(key) != std::string::npos) {
                keep = true;
                break;
            }
        }
        if (!keep) {
            sub.skip = true;
        }
    }
}

}  // namespace

std::string resolveAssetPath(const std::string& base_dir, const std::string& relative) {
    if (relative.empty()) {
        return {};
    }

    std::string rel = relative;
    std::replace(rel.begin(), rel.end(), '\\', '/');
    // Some exports write the reference as if it were absolute ("/Textures/x.png").
    while (!rel.empty() && rel.front() == '/') {
        rel.erase(0, 1);
    }
    if (rel.empty()) {
        return {};
    }

    const std::string direct = base_dir + "/" + rel;
    if (fileExists(direct)) {
        return direct;
    }

    // Fall back to a case-insensitive search of the plausible directories.
    const std::filesystem::path rel_path(rel);
    const std::string wanted = toLowerAscii(rel_path.filename().string());
    const std::filesystem::path base(base_dir);
    const std::filesystem::path candidates[] = {
        base / rel_path.parent_path(),
        base,
        base / "images",
        base / "Textures",
    };

    std::error_code ec;
    for (const std::filesystem::path& dir : candidates) {
        if (!std::filesystem::is_directory(dir, ec)) {
            continue;
        }
        for (std::filesystem::directory_iterator it(dir, ec), end; it != end && !ec;
             it.increment(ec)) {
            if (!it->is_regular_file(ec)) {
                continue;
            }
            if (toLowerAscii(it->path().filename().string()) == wanted) {
                return it->path().string();
            }
        }
    }
    return {};
}

bool loadObjMesh(const std::string& obj_path, ObjMesh& mesh) {
    mesh = ObjMesh{};
    mesh.base_dir = parentDirectory(obj_path);
    mesh.source_path = obj_path;
    mesh.min_bounds = cv::Point3f(
        std::numeric_limits<float>::max(),
        std::numeric_limits<float>::max(),
        std::numeric_limits<float>::max());
    mesh.max_bounds = cv::Point3f(
        std::numeric_limits<float>::lowest(),
        std::numeric_limits<float>::lowest(),
        std::numeric_limits<float>::lowest());

    std::ifstream obj_file(obj_path);
    if (!obj_file.is_open()) {
        return false;
    }

    std::string mtl_path;
    std::string current_material = "default";
    mesh.materials[current_material] = ObjMaterial{};
    mesh.materials[current_material].name = current_material;

    std::string line;
    while (std::getline(obj_file, line)) {
        line = trim(line);
        if (line.empty() || line[0] == '#') {
            continue;
        }

        std::istringstream iss(line);
        std::string tag;
        iss >> tag;

        if (tag == "mtllib") {
            iss >> mtl_path;
            continue;
        }

        if (tag == "usemtl") {
            iss >> current_material;
            if (!mesh.materials.count(current_material)) {
                mesh.materials[current_material] = ObjMaterial{};
                mesh.materials[current_material].name = current_material;
            }
            continue;
        }

        if (tag == "v") {
            cv::Point3f point{};
            iss >> point.x >> point.y >> point.z;
            mesh.vertices.push_back(point);
            mesh.colors.emplace_back(1.f, 1.f, 1.f);
            updateBounds(mesh, point);
            continue;
        }

        if (tag == "vt") {
            cv::Point2f uv{};
            iss >> uv.x >> uv.y;
            mesh.texcoords.push_back(uv);
            continue;
        }

        if (tag == "vn") {
            cv::Point3f normal{};
            iss >> normal.x >> normal.y >> normal.z;
            mesh.normals.push_back(normal);
            continue;
        }

        if (tag == "f") {
            std::vector<std::string> tokens;
            std::string token;
            while (iss >> token) {
                tokens.push_back(token);
            }
            if (tokens.size() < 3) {
                continue;
            }

            auto add_triangle = [&](const std::string& a, const std::string& b, const std::string& c) {
                int vi0 = -1;
                int vi1 = -1;
                int vi2 = -1;
                int ti0 = -1;
                int ti1 = -1;
                int ti2 = -1;
                int ni0 = -1;
                int ni1 = -1;
                int ni2 = -1;
                if (!parseFaceVertex(a, mesh.vertices.size(), mesh.texcoords.size(), mesh.normals.size(),
                                     vi0, ti0, ni0) ||
                    !parseFaceVertex(b, mesh.vertices.size(), mesh.texcoords.size(), mesh.normals.size(),
                                     vi1, ti1, ni1) ||
                    !parseFaceVertex(c, mesh.vertices.size(), mesh.texcoords.size(), mesh.normals.size(),
                                     vi2, ti2, ni2)) {
                    return;
                }
                ObjTriangle triangle;
                triangle.vertices = cv::Vec3i(vi0, vi1, vi2);
                triangle.texcoords = cv::Vec3i(ti0, ti1, ti2);
                triangle.normals = cv::Vec3i(ni0, ni1, ni2);
                mesh.triangles.push_back(triangle);
                mesh.triangle_materials.push_back(current_material);
            };

            for (std::size_t i = 1; i + 1 < tokens.size(); ++i) {
                add_triangle(tokens[0], tokens[i], tokens[i + 1]);
            }
        }
    }

    if (!mtl_path.empty()) {
        const std::string full_mtl_path = mesh.base_dir + "/" + mtl_path;
        std::ifstream mtl_file(full_mtl_path);
        if (mtl_file.is_open()) {
            std::string mtl_line;
            std::string active_material;
            while (std::getline(mtl_file, mtl_line)) {
                mtl_line = trim(mtl_line);
                if (mtl_line.empty() || mtl_line[0] == '#') {
                    continue;
                }

                std::istringstream mtl_iss(mtl_line);
                std::string mtl_tag;
                mtl_iss >> mtl_tag;
                if (mtl_tag == "newmtl") {
                    mtl_iss >> active_material;
                    mesh.materials[active_material] = ObjMaterial{};
                    mesh.materials[active_material].name = active_material;
                    continue;
                }
                if (active_material.empty()) {
                    continue;
                }

                ObjMaterial& material = mesh.materials[active_material];
                if (mtl_tag == "Kd") {
                    mtl_iss >> material.diffuse[0] >> material.diffuse[1] >> material.diffuse[2];
                } else if (mtl_tag == "map_Kd") {
                    mtl_iss >> material.diffuse_texture;
                }
            }
        }
    }

    resolveAllPokemonTextures(mesh);
    computeBodyBounds(mesh);
    return !mesh.vertices.empty() && !mesh.triangles.empty();
}

bool isGlbPath(const std::string& path) {
    if (path.size() < 4) {
        return false;
    }
    std::string ext = toLowerAscii(path.substr(path.size() - 4));
    return ext == ".glb";
}

// Assimp names embedded GLB textures "*0", "*1", ... Write the compressed
// blob next to the mesh so gl_renderer can cv::imread it like a sidecar PNG.
std::string extractEmbeddedTexture(
    const aiScene* scene, const std::string& tex_ref, const std::string& base_dir) {
    if (tex_ref.empty() || tex_ref[0] != '*' || scene == nullptr) {
        return tex_ref;
    }
    std::size_t end = 1;
    while (end < tex_ref.size() &&
           std::isdigit(static_cast<unsigned char>(tex_ref[end]))) {
        end++;
    }
    if (end == 1) {
        return "";
    }
    const unsigned int index = static_cast<unsigned int>(std::stoul(tex_ref.substr(1, end - 1)));
    if (index >= scene->mNumTextures || scene->mTextures[index] == nullptr) {
        return "";
    }
    const aiTexture* tex = scene->mTextures[index];

    std::string ext = "png";
    if (tex->achFormatHint[0] != '\0') {
        ext.clear();
        for (int i = 0; i < 3 && tex->achFormatHint[i] != '\0'; ++i) {
            ext.push_back(static_cast<char>(std::tolower(
                static_cast<unsigned char>(tex->achFormatHint[i]))));
        }
        if (ext.empty()) {
            ext = "png";
        }
    }

    const std::string filename = "embedded_" + std::to_string(index) + "." + ext;
    const fs::path out_path = fs::path(base_dir) / filename;
    std::error_code ec;
    fs::create_directories(out_path.parent_path(), ec);

    std::ofstream out(out_path, std::ios::binary);
    if (!out) {
        return "";
    }
    if (tex->mHeight == 0) {
        // Compressed (PNG/JPEG/WebP) blob; mWidth is the byte count.
        out.write(reinterpret_cast<const char*>(tex->pcData),
                  static_cast<std::streamsize>(tex->mWidth));
    } else {
        // Uncompressed BGRA texels.
        const std::size_t bytes =
            static_cast<std::size_t>(tex->mWidth) * tex->mHeight * 4;
        out.write(reinterpret_cast<const char*>(tex->pcData),
                  static_cast<std::streamsize>(bytes));
    }
    return filename;
}

bool loadAssimpMesh(const std::string& path, ObjMesh& mesh) {
    mesh = ObjMesh{};
    mesh.base_dir = parentDirectory(path);
    mesh.source_path = path;
    mesh.min_bounds = cv::Point3f(
        std::numeric_limits<float>::max(),
        std::numeric_limits<float>::max(),
        std::numeric_limits<float>::max());
    mesh.max_bounds = cv::Point3f(
        std::numeric_limits<float>::lowest(),
        std::numeric_limits<float>::lowest(),
        std::numeric_limits<float>::lowest());

    Assimp::Importer importer;
    // No aiProcess_FlipUVs: GLRenderer already flips V when building vertices.
    // Doing both cancels out and samples the atlas upside down (red rose texels
    // land on the torso, etc.).
    // No aiProcess_PreTransformVertices: it merges sub-meshes that share a
    // material, which loses the names needed to spot alternate-expression parts
    // (Gengar's lolling tongue). Node transforms are applied by hand below.
    const unsigned int flags =
        aiProcess_Triangulate |
        aiProcess_GenSmoothNormals |
        aiProcess_JoinIdenticalVertices;

    const aiScene* scene = importer.ReadFile(path, flags);
    if (!scene || !scene->mRootNode ||
        (scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE) || scene->mNumMeshes == 0) {
        std::cerr << "Assimp failed to load '" << path << "': "
                  << importer.GetErrorString() << std::endl;
        return false;
    }

    const bool glb_source = isGlbPath(path);

    // Materials. Keys stay mat_N so triangle slots stay stable; mat.name keeps
    // the authored label (Eye, FireCoreA, green) for effect/eye heuristics.
    for (unsigned int mi = 0; mi < scene->mNumMaterials; ++mi) {
        const aiMaterial* aimat = scene->mMaterials[mi];
        const std::string key = "mat_" + std::to_string(mi);
        ObjMaterial mat;
        mat.name = key;
        aiString authored_name;
        if (aimat->Get(AI_MATKEY_NAME, authored_name) == AI_SUCCESS &&
            authored_name.length > 0) {
            mat.name = authored_name.C_Str();
        }

        aiColor4D base(0.8f, 0.8f, 0.8f, 1.f);
        if (aimat->Get(AI_MATKEY_BASE_COLOR, base) == AI_SUCCESS) {
            mat.diffuse = cv::Vec3f(base.r, base.g, base.b);
            mat.opacity = base.a;
        } else {
            aiColor3D diffuse(0.8f, 0.8f, 0.8f);
            if (aimat->Get(AI_MATKEY_COLOR_DIFFUSE, diffuse) == AI_SUCCESS) {
                mat.diffuse = cv::Vec3f(diffuse.r, diffuse.g, diffuse.b);
            }
        }
        float opacity = mat.opacity;
        if (aimat->Get(AI_MATKEY_OPACITY, opacity) == AI_SUCCESS) {
            mat.opacity = opacity;
        }

        aiString tex;
        if (aimat->GetTexture(aiTextureType_DIFFUSE, 0, &tex) == AI_SUCCESS ||
            aimat->GetTexture(aiTextureType_BASE_COLOR, 0, &tex) == AI_SUCCESS) {
            std::string tex_path = tex.C_Str();
            std::replace(tex_path.begin(), tex_path.end(), '\\', '/');
            // Drop leading "./"
            if (tex_path.rfind("./", 0) == 0) {
                tex_path = tex_path.substr(2);
            }
            if (!tex_path.empty() && tex_path[0] == '*') {
                tex_path = extractEmbeddedTexture(scene, tex_path, mesh.base_dir);
            }
            mat.diffuse_texture = tex_path;
        }
        if (!glb_source) {
            resolvePokemonTextureSet(mat, mesh.base_dir);
        } else if (!isFacePartName(mat.name)) {
            if (!mat.diffuse_texture.empty()) {
                // poke-3D fire/gas sheets are greyscale intensity maps with no
                // normal sibling (Rapidash FireCoreA/FireStenA). Same rule as XY.
                mat.is_effect =
                    isGrayscaleImage(resolveAssetPath(mesh.base_dir, mat.diffuse_texture));
            } else if (mat.opacity < 0.999f && isEffectPartName(mat.name)) {
                // Untextured overlay shell wrapping the body. Alpha-blending it
                // just tints the whole Pokemon dark (Rotom Heat).
                mat.is_effect = true;
            }
        }
        mesh.materials[key] = mat;
    }
    if (mesh.materials.empty()) {
        ObjMaterial fallback;
        fallback.name = "default";
        fallback.diffuse = cv::Vec3f(0.8f, 0.8f, 0.8f);
        mesh.materials["default"] = fallback;
    }

    auto append_mesh = [&](const aiMesh* aimesh, const aiMatrix4x4& node_transform) {
        const int v_base = static_cast<int>(mesh.vertices.size());
        const int n_base = static_cast<int>(mesh.normals.size());
        const int t_base = static_cast<int>(mesh.texcoords.size());

        aiMatrix3x3 normal_transform(node_transform);
        normal_transform.Inverse().Transpose();

        for (unsigned int i = 0; i < aimesh->mNumVertices; ++i) {
            const aiVector3D p = node_transform * aimesh->mVertices[i];
            cv::Point3f pt(p.x, p.y, p.z);
            mesh.vertices.push_back(pt);
            updateBounds(mesh, pt);

            if (aimesh->HasNormals()) {
                aiVector3D n = normal_transform * aimesh->mNormals[i];
                n.NormalizeSafe();
                mesh.normals.emplace_back(n.x, n.y, n.z);
            } else {
                mesh.normals.emplace_back(0.f, 1.f, 0.f);
            }

            if (aimesh->HasTextureCoords(0)) {
                const aiVector3D& uv = aimesh->mTextureCoords[0][i];
                mesh.texcoords.emplace_back(uv.x, uv.y);
            } else {
                mesh.texcoords.emplace_back(0.f, 0.f);
            }

            // poke-3D Baltoy-style assets colour the mesh with COLOR_0 and ship
            // no diffuse texture. Assimp exposes them as per-vertex RGBA.
            if (aimesh->HasVertexColors(0)) {
                const aiColor4D& c = aimesh->mColors[0][i];
                mesh.colors.emplace_back(c.r, c.g, c.b);
            } else {
                mesh.colors.emplace_back(1.f, 1.f, 1.f);
            }
        }

        const std::string mat_name =
            "mat_" + std::to_string(aimesh->mMaterialIndex);
        if (!mesh.materials.count(mat_name)) {
            ObjMaterial fallback;
            fallback.name = mat_name;
            fallback.diffuse = cv::Vec3f(0.8f, 0.8f, 0.8f);
            mesh.materials[mat_name] = fallback;
        }

        for (unsigned int fi = 0; fi < aimesh->mNumFaces; ++fi) {
            const aiFace& face = aimesh->mFaces[fi];
            if (face.mNumIndices != 3) {
                continue;
            }
            ObjTriangle tri;
            const int i0 = v_base + static_cast<int>(face.mIndices[0]);
            const int i1 = v_base + static_cast<int>(face.mIndices[1]);
            const int i2 = v_base + static_cast<int>(face.mIndices[2]);
            // Per-vertex normals/uvs were pushed 1:1 with vertices above.
            const int ni0 = n_base + static_cast<int>(face.mIndices[0]);
            const int ni1 = n_base + static_cast<int>(face.mIndices[1]);
            const int ni2 = n_base + static_cast<int>(face.mIndices[2]);
            const int ti0 = t_base + static_cast<int>(face.mIndices[0]);
            const int ti1 = t_base + static_cast<int>(face.mIndices[1]);
            const int ti2 = t_base + static_cast<int>(face.mIndices[2]);
            tri.vertices = cv::Vec3i(i0, i1, i2);
            tri.normals = cv::Vec3i(ni0, ni1, ni2);
            tri.texcoords = cv::Vec3i(ti0, ti1, ti2);
            mesh.triangles.push_back(tri);
            mesh.triangle_materials.push_back(mat_name);
        }
    };

    std::vector<SubMesh> sub_meshes;
    collectSubMeshes(scene, scene->mRootNode, aiMatrix4x4(), sub_meshes);
    markAlternateExpressionParts(sub_meshes);
    markHeldItemSkins(sub_meshes);

    std::vector<aiMatrix4x4> mesh_transforms;
    for (const SubMesh& sub : sub_meshes) {
        if (sub.skip) {
            std::cerr << "Skipping alternate part '" << sub.name << "' in " << path << "\n";
            continue;
        }
        append_mesh(sub.mesh, sub.transform);
        mesh_transforms.push_back(sub.transform);
    }

    // poke-3D: same as XY Eye1_Merged — drop coplanar iris, punch sheet bg.
    if (glb_source) {
        fixPoke3dEyeLayers(mesh);
        undoSharedTiltedRoot(mesh, mesh_transforms);
    }

    computeBodyBounds(mesh);
    return !mesh.vertices.empty() && !mesh.triangles.empty();
}

std::vector<cv::Point3f> transformMeshVertices(
    const ObjMesh& mesh,
    const cv::Point3f& center,
    float scale,
    const cv::Mat& rotation) {
    std::vector<cv::Point3f> transformed;
    transformed.reserve(mesh.vertices.size());

    for (const cv::Point3f& vertex : mesh.vertices) {
        cv::Mat point = (cv::Mat_<float>(3, 1) << vertex.x - center.x, vertex.y - center.y, vertex.z - center.z);
        cv::Mat rotated = rotation * point;
        transformed.emplace_back(
            rotated.at<float>(0) * scale,
            rotated.at<float>(1) * scale,
            rotated.at<float>(2) * scale);
    }

    return transformed;
}

std::vector<std::pair<int, int>> buildUniqueEdges(const ObjMesh& mesh) {
    std::vector<std::pair<int, int>> edges;
    std::unordered_set<std::uint64_t> seen;

    for (const ObjTriangle& triangle : mesh.triangles) {
        const int indices[3] = {triangle.vertices[0], triangle.vertices[1], triangle.vertices[2]};
        for (int edge = 0; edge < 3; ++edge) {
            int a = indices[edge];
            int b = indices[(edge + 1) % 3];
            if (a > b) {
                std::swap(a, b);
            }
            const std::uint64_t key =
                (static_cast<std::uint64_t>(static_cast<std::uint32_t>(a)) << 32) |
                static_cast<std::uint32_t>(b);
            if (seen.insert(key).second) {
                edges.emplace_back(a, b);
            }
        }
    }

    return edges;
}
