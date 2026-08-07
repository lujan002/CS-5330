#include "model_library.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <filesystem>
#include <limits>
#include <map>
#include <set>
#include <utility>

namespace fs = std::filesystem;

namespace {

std::string toLower(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return value;
}

bool endsWith(const std::string& value, const std::string& suffix) {
    return value.size() >= suffix.size() &&
           value.compare(value.size() - suffix.size(), suffix.size(), suffix) == 0;
}

// Gendered rips ship as a pair ("VenusaurM"/"VenusaurF", "zubat_m"/"zubat_f").
// Keep the male model and present it under the plain species name. A lone
// suffixed model is left alone: it is the only export there is, and a trailing
// "m"/"f" may just be part of the name.
void dropFemaleVariants(std::map<std::string, std::pair<int, ModelEntry>>& best) {
    auto split_gender = [](const std::string& name,
                           std::string& base,
                           char& gender) -> bool {
        if (name.size() < 2) {
            return false;
        }
        const char last = static_cast<char>(std::tolower(static_cast<unsigned char>(name.back())));
        if (last != 'm' && last != 'f') {
            return false;
        }
        std::size_t cut = name.size() - 1;
        if (cut > 0 && name[cut - 1] == '_') {
            --cut;
        }
        if (cut == 0) {
            return false;
        }
        base = name.substr(0, cut);
        gender = last;
        return true;
    };

    std::map<std::string, std::map<char, std::string>> pairs;  // base -> gender -> key
    for (const auto& entry : best) {
        std::string base;
        char gender = 0;
        if (split_gender(entry.second.second.name, base, gender)) {
            pairs[toLower(base)][gender] = entry.first;
        }
    }

    for (const auto& group : pairs) {
        const auto male = group.second.find('m');
        const auto female = group.second.find('f');
        if (male == group.second.end() || female == group.second.end()) {
            continue;
        }
        best.erase(female->second);

        auto kept = best.find(male->second);
        if (kept == best.end()) {
            continue;
        }
        std::string base;
        char gender = 0;
        split_gender(kept->second.second.name, base, gender);
        ModelEntry renamed = kept->second.second;
        renamed.name = base;
        const int priority = kept->second.first;
        best.erase(kept);
        best[toLower(base)] = {priority, std::move(renamed)};
    }
}

}  // namespace

std::vector<ModelEntry> discoverModels(const std::vector<std::string>& roots) {
    std::vector<ModelEntry> models;
    std::set<std::string> seen_names;

    for (const std::string& root : roots) {
        std::error_code ec;
        if (!fs::exists(root, ec) || !fs::is_directory(root, ec)) {
            continue;
        }

        // Several exports of the same model can sit side by side; keep the best one.
        std::map<std::string, std::pair<int, ModelEntry>> best;
        for (fs::recursive_directory_iterator it(root, fs::directory_options::skip_permission_denied, ec), end;
             it != end && !ec;
             it.increment(ec)) {
            if (!it->is_regular_file(ec)) {
                continue;
            }
            const fs::path& p = it->path();
            const std::string ext = toLower(p.extension().string());
            const std::string stem = p.stem().string();

            std::string name;
            int priority = 0;
            if (ext == ".dae") {
                if (endsWith(stem, "_ColladaMax")) {
                    name = stem.substr(0, stem.size() - std::string("_ColladaMax").size());
                    priority = 0;
                } else if (endsWith(stem, "_OpenCollada")) {
                    // Usually a duplicate of the ColladaMax export, but some rips
                    // (Gyarados) ship only this one.
                    name = stem.substr(0, stem.size() - std::string("_OpenCollada").size());
                    priority = 1;
                } else {
                    name = stem;
                    priority = 2;
                }
            } else if (ext == ".obj") {
                name = stem;
                priority = 3;
            } else {
                continue;
            }

            // Mega forms are alternate versions of a Pokemon already in the list.
            if (toLower(name).find("mega") != std::string::npos) {
                continue;
            }

            const std::string key = toLower(name);
            const auto existing = best.find(key);
            if (existing == best.end() || priority < existing->second.first) {
                best[key] = {priority, ModelEntry{name, p.string()}};
            }
        }

        dropFemaleVariants(best);

        for (auto& entry : best) {
            if (seen_names.insert(entry.first).second) {
                models.push_back(std::move(entry.second.second));
            }
        }

        if (!models.empty()) {
            break;  // first root that yields models wins
        }
    }

    std::sort(models.begin(), models.end(),
              [](const ModelEntry& a, const ModelEntry& b) { return a.name < b.name; });
    return models;
}

float nativeUnitsPerMetre(const ObjMesh& mesh) {
    const std::string path = toLower(mesh.source_path);

    // Every XY archive unpacks into a "Pokemon XY/<Name>/" folder, which is a far
    // more reliable marker than the file name: most are "*_ColladaMax.DAE" but a
    // few (Ninetales) are just "<Name>.DAE", same units either way.
    if (path.find("pokemon xy") != std::string::npos) {
        return kXyUnitsPerMetre;
    }
    // The N64 Nidoking OBJ happens to be authored at the same scale (2.6 units/m).
    if (toLower(fs::path(mesh.source_path).extension().string()) == ".obj") {
        return kXyUnitsPerMetre;
    }
    return kModernRipUnitsPerMetre;
}

MeshOrientation detectOrientation(const ObjMesh& mesh) {
    // Effect volumes are excluded: Weezing's gas cloud straddles the origin and
    // makes +Y look better grounded than the body's true -Z up axis.
    const cv::Point3f& lo = mesh.body_min_bounds;
    const cv::Point3f& hi = mesh.body_max_bounds;

    const float extent[3] = {hi.x - lo.x, hi.y - lo.y, hi.z - lo.z};
    const float low[3] = {lo.x, lo.y, lo.z};
    const float high[3] = {hi.x, hi.y, hi.z};

    MeshOrientation best;
    float best_score = std::numeric_limits<float>::max();

    // X is the bilateral symmetry axis on every Pokemon model, so up is Y or Z.
    for (int axis = 1; axis <= 2; ++axis) {
        const float span = std::max(extent[axis], 1e-6f);
        // Treating +axis as up puts the model's base at its minimum, and vice versa.
        const float score_positive = std::fabs(low[axis]) / span;
        const float score_negative = std::fabs(high[axis]) / span;

        if (score_positive < best_score) {
            best_score = score_positive;
            best = MeshOrientation{axis, 1.f, span};
        }
        if (score_negative < best_score) {
            best_score = score_negative;
            best = MeshOrientation{axis, -1.f, span};
        }
    }

    // Caterpie (and a few other XY bugs) are centred on the origin rather than
    // planted on a ground plane, so Y and Z score within a few percent of each
    // other. Prefer the longer axis in that tie -- it matches Weedle/Metapod and
    // the authored standing pose. Skip the tie-break when a base is already
    // clearly planted (Raichu's feet sit on y=0); otherwise a long tail along Z
    // steals "up" and the model tips onto its back.
    if (best_score > 0.25f) {
        const float tie = 0.15f;
        for (int axis = 1; axis <= 2; ++axis) {
            const float span = std::max(extent[axis], 1e-6f);
            for (float sign : {1.f, -1.f}) {
                const float base = sign > 0.f ? low[axis] : high[axis];
                const float score = std::fabs(base) / span;
                if (score <= best_score + tie && span > best.height * 1.25f) {
                    best_score = score;
                    best = MeshOrientation{axis, sign, span};
                }
            }
        }
    }

    // XY rips that aren't planted on a ground plane (Arbok, Ekans) often score
    // slightly better for -Y because the head sits closer to the origin than the
    // tail. That hangs them upside down. Prefer the +Y convention when neither
    // end is clearly the floor.
    if (best.axis == 1 && best_score > 0.35f) {
        best.sign = 1.f;
        best.height = extent[1];
    }
    return best;
}

ModelPlacement placeMeshOnCard(
    const ObjMesh& mesh,
    float card_w,
    float card_h,
    float inches_per_unit) {
    ModelPlacement placement;

    const MeshOrientation up = detectOrientation(mesh);

    // Build a proper rotation (determinant +1, so the model is never mirrored)
    // that sends the mesh's up axis to card -Z, which is "out of the card" -- the
    // same convention the pyramid overlay uses.
    //
    // Card X defaults to mesh X (the bilateral symmetry axis). Snakes like Ekans
    // are authored nearly planar in YZ, so keeping X as width shows only a
    // needle-thin edge; when X is much smaller than the other horizontal axis,
    // that longer axis becomes card X instead.
    cv::Vec3f up_vec(0.f, 0.f, 0.f);
    up_vec[up.axis] = up.sign;

    const float extent[3] = {
        mesh.body_max_bounds.x - mesh.body_min_bounds.x,
        mesh.body_max_bounds.y - mesh.body_min_bounds.y,
        mesh.body_max_bounds.z - mesh.body_min_bounds.z};
    // Prefer mesh X as card width (bilateral symmetry). Swap only when X is a
    // thin edge relative to the model's height -- Ekans is authored as a flat
    // ribbon in YZ, so keeping X as width showed a needle. Raichu's long tail
    // makes Z large without X being thin, so it must keep X.
    int width_axis = 0;
    const int other_horizontal = up.axis == 1 ? 2 : 1;
    if (extent[0] < up.height * 0.2f &&
        extent[0] < extent[other_horizontal] * 0.8f) {
        width_axis = other_horizontal;
    }

    cv::Vec3f row0(0.f, 0.f, 0.f);
    row0[width_axis] = 1.f;
    const cv::Vec3f row2 = -up_vec;
    const cv::Vec3f row1 = row2.cross(row0);
    placement.rotation = (cv::Mat_<float>(3, 3) <<
        row0[0], row0[1], row0[2],
        row1[0], row1[1], row1[2],
        row2[0], row2[1], row2[2]);

    const cv::Point3f& lo = mesh.body_min_bounds;
    const cv::Point3f& hi = mesh.body_max_bounds;
    const float low[3] = {lo.x, lo.y, lo.z};
    const float high[3] = {hi.x, hi.y, hi.z};

    // Pivot: centred on the two horizontal axes, resting on the base of the up axis.
    cv::Point3f pivot(0.5f * (lo.x + hi.x), 0.5f * (lo.y + hi.y), 0.5f * (lo.z + hi.z));
    const float base = up.sign > 0.f ? low[up.axis] : high[up.axis];
    if (up.axis == 1) {
        pivot.y = base;
    } else {
        pivot.z = base;
    }

    // XY Collada rips are already in battle proportions (<unit name="inch"/>).
    // Newer rips are ~40× larger; fold them into XY inches first so one --scale
    // keeps every Pokemon at its in-game relative size. Pokédex metres are NOT
    // used: they shrink bugs like Caterpie and inflate Onix far beyond battle.
    const float unit_correction = kXyUnitsPerMetre / nativeUnitsPerMetre(mesh);
    const float scale = std::max(inches_per_unit, 1e-6f) * unit_correction;
    placement.vertices = transformMeshVertices(mesh, pivot, scale, placement.rotation);
    placement.height_inches = up.height * scale;

    // Stand it in the middle of the card face.
    for (cv::Point3f& v : placement.vertices) {
        v.x += 0.5f * card_w;
        v.y += 0.5f * card_h;
    }
    return placement;
}
