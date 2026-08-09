#pragma once

#include "obj_loader.hpp"

#include <opencv2/core.hpp>

#include <string>
#include <vector>

struct ModelEntry {
    std::string name;  // display name, e.g. "Charizard"
    std::string path;  // full path to the model file
    int dex = -1;      // national dex when known
};

// Recursively scan roots for loadable models. Collada "*_ColladaMax.DAE" is
// preferred over the "*_OpenCollada.DAE" duplicates; plain .obj is also picked up.
// Used after unpacking a downloaded archive into a scratch directory.
std::vector<ModelEntry> discoverModels(const std::vector<std::string>& roots);

// Fill entry.dex from a #NNNN folder in the path, then from the height table by
// name. No-op when dex is already set.
void resolveModelDex(ModelEntry& entry);

// Which way is up in a mesh's own coordinates. Usually an axis (±Y / ±Z), but
// some authored poses sit on a diagonal (Articuno's ~45° lean) — then `up` is a
// continuous unit vector in the YZ plane and axis/sign mirror the nearest axis.
struct MeshOrientation {
    cv::Vec3f up{0.f, 1.f, 0.f};  // unit vector in mesh space
    int axis = 1;                 // nearest axis: 1 = Y, 2 = Z
    float sign = 1.f;             // nearest sign along that axis
    float height = 1.f;           // extent along `up`, in native units
};

// Infers the up axis from the fact that these models are authored standing on a
// ground plane: the base of the up axis sits at (or very near) zero, whereas the
// other axes straddle the origin.
MeshOrientation detectOrientation(const ObjMesh& mesh);

struct ModelPlacement {
    std::vector<cv::Point3f> vertices;  // card-space vertex positions
    cv::Mat rotation;                   // object -> card rotation (3x3, CV_32F)
    float height_inches = 0.f;          // final height standing on the card
    float target_inches = 0.f;          // Pokédex log-scale target before mesh fit
    float height_m = 0.f;               // Pokédex height used (metres)
};

// Global multiplier on the log-scale target height (CLI --scale).
constexpr float kDefaultScaleMultiplier = 1.0f;

// Exponent in target = gain * log1p(height_m^y). Lower → flatter sizes.
constexpr float kDefaultScaleY = 0.85f;

// gain is derived so Pikachu (0.4 m) maps to ~1.3 in at --scale 1 and default y.
constexpr float kScaleRefHeightM = 0.4f;
constexpr float kScaleRefInches = 1.3f;

constexpr float kMinHeightInches = 0.55f;
constexpr float kMaxHeightInches = 6.0f;

// Used when dex/height is unknown — mid-size Pokémon, not authored mesh size.
constexpr float kDefaultHeightM = 1.0f;

// Kept for diagnose / older rip docs; placement no longer scales by unit family.
constexpr float kXyUnitsPerMetre = 2.4f;
constexpr float kModernRipUnitsPerMetre = 95.f;

// Deprecated alias so existing call sites compiling against the old name still
// see a sensible default for --scale.
constexpr float kDefaultInchesPerUnit = kDefaultScaleMultiplier;

struct ScaleParams {
    float scale_multiplier = kDefaultScaleMultiplier;  // CLI --scale
    float y = kDefaultScaleY;                           // CLI --scale-y
    float height_m = kDefaultHeightM;                   // Pokédex metres
};

// Pokédex height (m) → on-card inches via gain * log1p(height_m^y), clamped.
float pokedexTargetInches(float height_m, float y, float scale_multiplier);

float nativeUnitsPerMetre(const ObjMesh& mesh);

// Stand the mesh upright on the card sized to the Pokédex log-scale target.
ModelPlacement placeMeshOnCard(
    const ObjMesh& mesh,
    float card_w,
    float card_h,
    const ScaleParams& scale = ScaleParams{});
