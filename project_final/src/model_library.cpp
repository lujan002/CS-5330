#include "model_library.hpp"
#include "pokemon_heights.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <limits>
#include <map>
#include <set>
#include <utility>
#include <vector>

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
                ModelEntry me;
                me.name = name;
                me.path = p.string();
                best[key] = {priority, std::move(me)};
            }
        }

        dropFemaleVariants(best);

        for (auto& entry : best) {
            if (seen_names.insert(entry.first).second) {
                resolveModelDex(entry.second.second);
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

void resolveModelDex(ModelEntry& entry) {
    if (entry.dex > 0) {
        return;
    }

    // Folder names look like "... - #0006 Charizard".
    const std::string path = entry.path;
    for (std::size_t i = 0; i < path.size(); ++i) {
        if (path[i] != '#') {
            continue;
        }
        std::size_t end = i + 1;
        while (end < path.size() &&
               std::isdigit(static_cast<unsigned char>(path[end]))) {
            ++end;
        }
        if (end > i + 1) {
            try {
                const int dex = std::stoi(path.substr(i + 1, end - (i + 1)));
                if (dex > 0) {
                    entry.dex = dex;
                    return;
                }
            } catch (...) {
            }
        }
    }

    entry.dex = PokemonHeights::dexByName(entry.name);
}

float nativeUnitsPerMetre(const ObjMesh& mesh) {
    const std::string path = toLower(mesh.source_path);

    // Every XY archive unpacks into a "Pokemon XY/<Name>/" folder, which is a far
    // more reliable marker than the file name: most are "*_ColladaMax.DAE" but a
    // few (Ninetales) are just "<Name>.DAE", same units either way.
    if (path.find("pokemon xy") != std::string::npos) {
        return kXyUnitsPerMetre;
    }
    // Pokemon-3D-api GLBs are treated like XY for unit-family diagnostics only;
    // placement size comes from Pokédex log-scale, not this constant.
    if (path.size() >= 4 && path.compare(path.size() - 4, 4, ".glb") == 0) {
        return kXyUnitsPerMetre;
    }
    // The N64 Nidoking OBJ happens to be authored at the same scale (2.6 units/m).
    if (toLower(fs::path(mesh.source_path).extension().string()) == ".obj") {
        return kXyUnitsPerMetre;
    }
    return kModernRipUnitsPerMetre;
}

// Centroid of the eye geometry, or false when the rip paints the eyes into the
// body texture instead of giving them their own material.
bool eyeCentroid(const ObjMesh& mesh, cv::Point3f& centre) {
    cv::Point3f sum(0.f, 0.f, 0.f);
    long count = 0;
    for (std::size_t i = 0; i < mesh.triangles.size(); ++i) {
        const auto material = mesh.materials.find(mesh.triangle_materials[i]);
        if (material == mesh.materials.end()) {
            continue;
        }
        const std::string name = toLower(material->second.name);
        const bool is_eye = material->second.eye_layer != 0 ||
                            name.find("eye") != std::string::npos ||
                            name.find("iris") != std::string::npos ||
                            name.find("pupil") != std::string::npos;
        if (!is_eye) {
            continue;
        }
        for (int corner = 0; corner < 3; ++corner) {
            const int index = mesh.triangles[i].vertices[corner];
            if (index >= 0 && index < static_cast<int>(mesh.vertices.size())) {
                sum += mesh.vertices[index];
                count++;
            }
        }
    }
    if (count == 0) {
        return false;
    }
    centre = sum / static_cast<float>(count);
    return true;
}

MeshOrientation detectOrientation(const ObjMesh& mesh) {
    // Effect volumes are excluded: Weezing's gas cloud straddles the origin and
    // makes +Y look better grounded than the body's true -Z up axis.
    const cv::Point3f& lo = mesh.body_min_bounds;
    const cv::Point3f& hi = mesh.body_max_bounds;

    const float extent[3] = {hi.x - lo.x, hi.y - lo.y, hi.z - lo.z};
    const float low[3] = {lo.x, lo.y, lo.z};
    const float high[3] = {hi.x, hi.y, hi.z};

    cv::Point3f eye_centre;
    const bool has_eyes = eyeCentroid(mesh, eye_centre);
    const float eye[3] = {eye_centre.x, eye_centre.y, eye_centre.z};

    // Body centroid (effects skipped) — used to reject "standing on the heavy
    // end" candidates such as Articuno's long-axis tip plant.
    cv::Point3f body_sum(0.f, 0.f, 0.f);
    long body_count = 0;
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
            body_sum += mesh.vertices[index];
            body_count++;
        }
    }
    const cv::Point3f body_cen =
        body_count > 0 ? body_sum * (1.f / static_cast<float>(body_count))
                       : cv::Point3f(0.f, 0.f, 0.f);
    const float cen[3] = {body_cen.x, body_cen.y, body_cen.z};

    // Every rip we consume is bilaterally symmetric about X, so up is Y or Z;
    // only the two signs of each are worth scoring.
    struct Candidate {
        int axis;
        float sign;
        float span;
        float planted;     // |base| / span: 0 when authored on a ground plane
        float eye_height;  // 0 at the base, 1 at the top; -1 when unknown
        float cen_height;  // centroid along up: 0 at base, 1 at top
        int base_occ = 0;  // occupied cells in a footprint grid at the base
    };
    std::vector<Candidate> candidates;
    for (int axis = 1; axis <= 2; ++axis) {
        const float span = std::max(extent[axis], 1e-6f);
        for (float sign : {1.f, -1.f}) {
            Candidate c;
            c.axis = axis;
            c.sign = sign;
            c.span = span;
            c.planted = std::fabs(sign > 0.f ? low[axis] : high[axis]) / span;
            c.eye_height = -1.f;
            if (has_eyes) {
                const float fraction = (eye[axis] - low[axis]) / span;
                c.eye_height = sign > 0.f ? fraction : 1.f - fraction;
            }
            const float cen_frac = (cen[axis] - low[axis]) / span;
            c.cen_height = sign > 0.f ? cen_frac : 1.f - cen_frac;
            candidates.push_back(c);
        }
    }

    // Footprint occupancy at each candidate's base. A face that merely grazes
    // the origin (Pikachu's back tip) scores as well-planted but covers almost
    // no cells; real feet cover a meaningful slice of the horizontal AABB.
    constexpr int kGrid = 32;
    for (Candidate& c : candidates) {
        const int a0 = 0;
        const int a1 = c.axis == 1 ? 2 : 1;
        const float span0 = std::max(extent[a0], 1e-6f);
        const float span1 = std::max(extent[a1], 1e-6f);
        const float base_lim = c.sign > 0.f ? low[c.axis] + 0.08f * c.span
                                            : high[c.axis] - 0.08f * c.span;
        std::vector<unsigned char> cells(kGrid * kGrid, 0);
        int occ = 0;
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
                const float coords[3] = {p.x, p.y, p.z};
                const bool in_base = c.sign > 0.f ? coords[c.axis] <= base_lim
                                                  : coords[c.axis] >= base_lim;
                if (!in_base) {
                    continue;
                }
                const int c0 = std::clamp(
                    static_cast<int>((coords[a0] - low[a0]) / span0 * (kGrid - 1)),
                    0, kGrid - 1);
                const int c1 = std::clamp(
                    static_cast<int>((coords[a1] - low[a1]) / span1 * (kGrid - 1)),
                    0, kGrid - 1);
                unsigned char& cell = cells[c1 * kGrid + c0];
                if (!cell) {
                    cell = 1;
                    occ++;
                }
            }
        }
        c.base_occ = occ;
    }

    // Nearly every asset is authored standing on a ground plane, so the axis
    // whose base sits at the origin is the up axis. The trap is a model with an
    // unrelated face at zero -- Mewtwo's and Greedent's back planes make Z look
    // planted, which lays them face-down on the card.
    //
    // Eyes break that tie: no character stands with its eyes in the bottom
    // sixth of its own silhouette, so any candidate that would put them there is
    // a lying-down pose and is discarded before the planted test runs.
    //
    // Centroid height catches the same class of miss when there is no eye
    // material: Articuno's long-axis tip plant puts the body's mass in the
    // bottom fifth of the AABB (diving pose with the tail streamers as "up").
    //
    // Tip plants: a single vertex cluster grazing the origin (Pikachu -Z) looks
    // perfectly planted but covers almost no footprint cells.
    const float kEyeFloor = 0.15f;
    const float kCenFloor = 0.30f;
    const float kTipPlanted = 0.10f;
    const int kTipOcc = 10;
    auto is_viable = [&](const Candidate& c) {
        if (c.eye_height >= 0.f && c.eye_height < kEyeFloor) {
            return false;
        }
        if (c.cen_height < kCenFloor) {
            return false;
        }
        if (c.planted < kTipPlanted && c.base_occ < kTipOcc) {
            return false;
        }
        return true;
    };
    std::vector<const Candidate*> viable;
    for (const Candidate& c : candidates) {
        if (is_viable(c)) {
            viable.push_back(&c);
        }
    }
    if (viable.empty()) {
        // Fall back without the centroid/tip filters (keep the eye veto).
        for (const Candidate& c : candidates) {
            if (c.eye_height < 0.f || c.eye_height >= kEyeFloor) {
                viable.push_back(&c);
            }
        }
    }
    if (viable.empty()) {
        for (const Candidate& c : candidates) {
            viable.push_back(&c);
        }
    }

    const Candidate* best = viable.front();
    for (const Candidate* c : viable) {
        if (c->planted < best->planted) {
            best = c;
        }
    }

    // glTF and the XY rips both call +Y up, so keep that convention whenever the
    // alternative is not decisively better grounded (Gardevoir scores +Y 0.173
    // against -Z 0.183 and would otherwise hang upside down).
    const float kYPreference = 0.10f;
    for (const Candidate* c : viable) {
        if (c->axis == 1 && c->sign > 0.f &&
            c->planted <= best->planted + kYPreference) {
            best = c;
            break;
        }
    }

    // Charizard is authored floating, so no face is near the origin and the
    // axis-aligned winner is just the least-bad accident — fall back to +Y.
    // Articuno is the same class of miss but its body sits on a ~45° diagonal
    // between +Y and -Z; snapping to either axis leaves it pitched. When the
    // discrete answer is unplanted, search a continuous pitch around it and
    // accept a nearby angle that actually plants (centroid still sane).
    const float kUnplanted = 0.25f;
    auto make_orient = [&](float uy, float uz, float span) {
        MeshOrientation o;
        const float len = std::sqrt(uy * uy + uz * uz);
        if (len < 1e-6f) {
            uy = 1.f;
            uz = 0.f;
        } else {
            uy /= len;
            uz /= len;
        }
        o.up = cv::Vec3f(0.f, uy, uz);
        if (std::fabs(uy) >= std::fabs(uz)) {
            o.axis = 1;
            o.sign = uy >= 0.f ? 1.f : -1.f;
        } else {
            o.axis = 2;
            o.sign = uz >= 0.f ? 1.f : -1.f;
        }
        o.height = span;
        return o;
    };

    float theta = 0.f;
    if (best->axis == 1) {
        theta = best->sign > 0.f ? 0.f : static_cast<float>(CV_PI);
    } else {
        theta = best->sign > 0.f ? static_cast<float>(CV_PI / 2.0)
                                 : static_cast<float>(-CV_PI / 2.0);
    }
    // Unplanted discrete answer: prefer +Y as the search centre (glTF / XY).
    if (!has_eyes && best->planted > kUnplanted) {
        theta = 0.f;
        best = nullptr;  // span taken from continuous measure below
    }

    // Body samples for continuous scoring (same set as the centroid above).
    std::vector<cv::Point3f> body_pts;
    body_pts.reserve(static_cast<std::size_t>(std::max(body_count, 0L)));
    for (std::size_t i = 0; i < mesh.triangles.size(); ++i) {
        const auto material = mesh.materials.find(mesh.triangle_materials[i]);
        if (material != mesh.materials.end() && material->second.is_effect) {
            continue;
        }
        for (int corner = 0; corner < 3; ++corner) {
            const int index = mesh.triangles[i].vertices[corner];
            if (index >= 0 && index < static_cast<int>(mesh.vertices.size())) {
                body_pts.push_back(mesh.vertices[index]);
            }
        }
    }

    auto score_theta = [&](float th, float& planted, float& cen_h, float& span,
                           int& base_occ, float& eye_h) {
        const float cy = std::cos(th);
        const float sy = std::sin(th);
        float lo_u = std::numeric_limits<float>::max();
        float hi_u = std::numeric_limits<float>::lowest();
        double sum_u = 0.0;
        float lo_v = std::numeric_limits<float>::max();
        float hi_v = std::numeric_limits<float>::lowest();
        float lo_x = std::numeric_limits<float>::max();
        float hi_x = std::numeric_limits<float>::lowest();
        for (const cv::Point3f& p : body_pts) {
            const float u = p.y * cy + p.z * sy;
            const float v = -p.y * sy + p.z * cy;
            lo_u = std::min(lo_u, u);
            hi_u = std::max(hi_u, u);
            lo_v = std::min(lo_v, v);
            hi_v = std::max(hi_v, v);
            lo_x = std::min(lo_x, p.x);
            hi_x = std::max(hi_x, p.x);
            sum_u += u;
        }
        span = std::max(hi_u - lo_u, 1e-6f);
        planted = std::fabs(lo_u) / span;
        cen_h = static_cast<float>((sum_u / std::max<std::size_t>(body_pts.size(), 1) - lo_u) /
                                   span);
        eye_h = -1.f;
        if (has_eyes) {
            const float eu = eye[1] * cy + eye[2] * sy;
            eye_h = (eu - lo_u) / span;
        }
        const float base_lim = lo_u + 0.08f * span;
        const float span_x = std::max(hi_x - lo_x, 1e-6f);
        const float span_v = std::max(hi_v - lo_v, 1e-6f);
        constexpr int kG = 24;
        std::vector<unsigned char> cells(kG * kG, 0);
        base_occ = 0;
        for (const cv::Point3f& p : body_pts) {
            const float u = p.y * cy + p.z * sy;
            if (u > base_lim) {
                continue;
            }
            const float v = -p.y * sy + p.z * cy;
            const int c0 = std::clamp(
                static_cast<int>((p.x - lo_x) / span_x * (kG - 1)), 0, kG - 1);
            const int c1 = std::clamp(
                static_cast<int>((v - lo_v) / span_v * (kG - 1)), 0, kG - 1);
            unsigned char& cell = cells[c1 * kG + c0];
            if (!cell) {
                cell = 1;
                base_occ++;
            }
        }
    };

    float best_planted = 1.f;
    float best_span = extent[1];
    float best_theta = theta;
    if (best != nullptr) {
        best_planted = best->planted;
        best_span = best->span;
    }

    // Only hunt for a diagonal when the axis-aligned answer is unplanted and
    // there are no eyes to lock the pose (Articuno). Charmander / Haunter stay
    // on their true axis; eyed models keep the discrete pick.
    if (!has_eyes && best_planted > kUnplanted && !body_pts.empty()) {
        // PCA of body in YZ: Articuno's long axis is the body+tail diagonal.
        // Among angles that plant, prefer the one that sets up ⟂ long axis so
        // the bird flies level instead of nose-up/nose-down.
        double yy = 0.0, zz = 0.0, yz = 0.0;
        cv::Point3f mean(0.f, 0.f, 0.f);
        for (const cv::Point3f& p : body_pts) {
            mean += p;
        }
        mean *= 1.f / static_cast<float>(body_pts.size());
        for (const cv::Point3f& p : body_pts) {
            const double dy = p.y - mean.y;
            const double dz = p.z - mean.z;
            yy += dy * dy;
            zz += dz * dz;
            yz += dy * dz;
        }
        yy /= body_pts.size();
        zz /= body_pts.size();
        yz /= body_pts.size();
        const double tr = yy + zz;
        const double det = yy * zz - yz * yz;
        const double disc = std::sqrt(std::max(0.0, tr * tr * 0.25 - det));
        const double l1 = tr * 0.5 + disc;
        double ly = yz;
        double lz = l1 - yy;
        if (std::fabs(ly) + std::fabs(lz) < 1e-12) {
            ly = l1 - zz;
            lz = yz;
        }
        const double ln = std::sqrt(ly * ly + lz * lz);
        if (ln > 1e-12) {
            ly /= ln;
            lz /= ln;
        }

        float accept_planted = best_planted;
        float accept_span = best_span;
        float accept_theta = best_theta;
        float accept_level = 1.f;  // |up · long|; 0 = body horizontal
        bool found = false;
        const float window = static_cast<float>(70.0 * CV_PI / 180.0);
        const float step = static_cast<float>(2.0 * CV_PI / 180.0);
        for (float th = theta - window; th <= theta + window + 1e-6f; th += step) {
            float planted = 1.f, cen_h = 0.f, span = 1.f, eye_h = -1.f;
            int base_occ = 0;
            score_theta(th, planted, cen_h, span, base_occ, eye_h);
            if (cen_h < kCenFloor) {
                continue;
            }
            if (eye_h >= 0.f && eye_h < kEyeFloor) {
                continue;
            }
            if (planted < kTipPlanted && base_occ < kTipOcc) {
                continue;
            }
            // Must actually plant — a weak improvement on another floating angle
            // is how Charizard would get dragged off +Y.
            if (planted > kUnplanted) {
                continue;
            }
            const float cy = std::cos(th);
            const float sy = std::sin(th);
            const float level = static_cast<float>(
                std::fabs(cy * ly + sy * lz));
            // Prefer level flight; use planted as a tie-break.
            if (!found || level < accept_level - 1e-4f ||
                (std::fabs(level - accept_level) <= 1e-4f &&
                 planted < accept_planted)) {
                found = true;
                accept_planted = planted;
                accept_span = span;
                accept_theta = th;
                accept_level = level;
            }
        }
        if (found) {
            best_theta = accept_theta;
            best_span = accept_span;
            best_planted = accept_planted;
        }
    }

    // Still unplanted after the search: authoring convention is +Y only when
    // there are no eyes to defend a non-Y discrete pick (Mewtwo's +Z).
    if (best_planted > kUnplanted && !has_eyes) {
        return make_orient(1.f, 0.f, extent[1]);
    }

    return make_orient(std::cos(best_theta), std::sin(best_theta), best_span);
}

float pokedexTargetInches(float height_m, float y, float scale_multiplier) {
    const float h = std::max(height_m, 1e-6f);
    const float exp = std::max(y, 1e-6f);
    const float ref = std::max(kScaleRefHeightM, 1e-6f);
    // gain chosen so reference species (Pikachu) lands at kScaleRefInches when
    // scale_multiplier == 1.
    const float gain = kScaleRefInches / std::log1p(std::pow(ref, exp));
    float target = gain * std::log1p(std::pow(h, exp));
    target *= std::max(scale_multiplier, 1e-6f);
    return std::clamp(target, kMinHeightInches, kMaxHeightInches);
}

ModelPlacement placeMeshOnCard(
    const ObjMesh& mesh,
    float card_w,
    float card_h,
    const ScaleParams& scale_params) {
    ModelPlacement placement;

    const MeshOrientation up = detectOrientation(mesh);

    // Build a proper rotation (determinant +1, so the model is never mirrored)
    // that sends the mesh's up vector to card -Z, which is "out of the card" --
    // the same convention the pyramid overlay uses.
    //
    // Card X defaults to mesh X (the bilateral symmetry axis). Snakes like Ekans
    // are authored nearly planar in YZ, so keeping X as width shows only a
    // needle-thin edge; when X is much smaller than the other horizontal axis,
    // that longer axis becomes card X instead.
    cv::Vec3f up_vec = up.up;
    const float up_len = std::sqrt(up_vec.dot(up_vec));
    if (up_len > 1e-6f) {
        up_vec *= 1.f / up_len;
    } else {
        up_vec = cv::Vec3f(0.f, 1.f, 0.f);
    }

    const float extent[3] = {
        mesh.body_max_bounds.x - mesh.body_min_bounds.x,
        mesh.body_max_bounds.y - mesh.body_min_bounds.y,
        mesh.body_max_bounds.z - mesh.body_min_bounds.z};
    // Prefer mesh X as card width (bilateral symmetry). Swap only when X is a
    // thin edge -- Ekans (YZ ribbon) and flat props like Dhelmise (wheel faces
    // ±X). Threshold is 0.25·height so Dhelmise (~0.22·h) swaps but Ponyta
    // (~0.32·h, long nose-to-tail) keeps X and faces the camera. Raichu's long
    // tail makes Z large without X being thin, so it also keeps X.
    int width_axis = 0;
    const int other_horizontal = up.axis == 1 ? 2 : 1;
    if (extent[0] < up.height * 0.25f &&
        extent[0] < extent[other_horizontal] * 0.8f) {
        width_axis = other_horizontal;
    }

    cv::Vec3f row0(0.f, 0.f, 0.f);
    row0[width_axis] = 1.f;
    // Keep width in the plane perpendicular to up (continuous pitch tilts YZ).
    row0 = row0 - up_vec * row0.dot(up_vec);
    float row0_len = std::sqrt(row0.dot(row0));
    if (row0_len < 1e-4f) {
        row0 = cv::Vec3f(0.f, up_vec[2], -up_vec[1]);
        row0_len = std::sqrt(row0.dot(row0));
    }
    row0 *= 1.f / std::max(row0_len, 1e-6f);
    const cv::Vec3f row2 = -up_vec;
    cv::Vec3f row1 = row2.cross(row0);
    const float row1_len = std::sqrt(row1.dot(row1));
    if (row1_len > 1e-6f) {
        row1 *= 1.f / row1_len;
    }
    // Re-orthogonalise in case the projection nudged things.
    row0 = row1.cross(row2);
    placement.rotation = (cv::Mat_<float>(3, 3) <<
        row0[0], row0[1], row0[2],
        row1[0], row1[1], row1[2],
        row2[0], row2[1], row2[2]);

    const cv::Point3f& lo = mesh.body_min_bounds;
    const cv::Point3f& hi = mesh.body_max_bounds;

    // Pivot: AABB centre projected onto the base plane (min along up).
    cv::Point3f pivot(0.5f * (lo.x + hi.x), 0.5f * (lo.y + hi.y), 0.5f * (lo.z + hi.z));
    float base_up = std::numeric_limits<float>::max();
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
            const float u = p.x * up_vec[0] + p.y * up_vec[1] + p.z * up_vec[2];
            base_up = std::min(base_up, u);
        }
    }
    if (base_up > 1e20f) {
        base_up = pivot.x * up_vec[0] + pivot.y * up_vec[1] + pivot.z * up_vec[2];
    }
    const float pivot_up = pivot.x * up_vec[0] + pivot.y * up_vec[1] + pivot.z * up_vec[2];
    pivot.x -= up_vec[0] * (pivot_up - base_up);
    pivot.y -= up_vec[1] * (pivot_up - base_up);
    pivot.z -= up_vec[2] * (pivot_up - base_up);

    // Size from Pokédex height (log-compressed), not authored mesh proportions.
    // Mesh authorship only supplies the body AABB height we normalize against.
    placement.height_m = std::max(scale_params.height_m, 1e-6f);
    placement.target_inches =
        pokedexTargetInches(placement.height_m, scale_params.y, scale_params.scale_multiplier);
    const float mesh_h = std::max(up.height, 1e-6f);
    const float scale = placement.target_inches / mesh_h;
    placement.vertices = transformMeshVertices(mesh, pivot, scale, placement.rotation);
    placement.height_inches = up.height * scale;

    // The pivot rests the *body* box on the card, but the body box is only the
    // materials that survived effect classification, and a hunched or reclining
    // pose can put a limb below the box base. Card +Z points into the card, so
    // anything with z > 0 has sunk through the table: lift the whole model until
    // its lowest drawn vertex just touches the surface.
    float deepest = 0.f;
    for (std::size_t i = 0; i < mesh.triangles.size(); ++i) {
        for (int corner = 0; corner < 3; ++corner) {
            const int index = mesh.triangles[i].vertices[corner];
            if (index >= 0 && index < static_cast<int>(placement.vertices.size())) {
                deepest = std::max(deepest, placement.vertices[index].z);
            }
        }
    }
    if (deepest > 1e-4f) {
        for (cv::Point3f& v : placement.vertices) {
            v.z -= deepest;
        }
    }

    // Stand it in the middle of the card face.
    for (cv::Point3f& v : placement.vertices) {
        v.x += 0.5f * card_w;
        v.y += 0.5f * card_h;
    }
    return placement;
}
