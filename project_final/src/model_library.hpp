#pragma once

#include "obj_loader.hpp"

#include <opencv2/core.hpp>

#include <string>
#include <vector>

struct ModelEntry {
    std::string name;  // display name, e.g. "Charizard"
    std::string path;  // full path to the model file
};

// Recursively scan roots for loadable models. Collada "*_ColladaMax.DAE" is
// preferred over the "*_OpenCollada.DAE" duplicates; plain .obj is also picked up.
std::vector<ModelEntry> discoverModels(const std::vector<std::string>& roots);

// Which way is up in a mesh's own coordinates. The rips are inconsistent: the
// "Pokemon XY" exports come out +Y up, while the newer per-Pokemon exports are
// -Y up (Snorlax) or -Z up (Haunter, Onix).
struct MeshOrientation {
    int axis = 1;        // 1 = Y, 2 = Z
    float sign = 1.f;    // +1 or -1 along that axis
    float height = 1.f;  // extent along the up axis, in native units
};

// Infers the up axis from the fact that these models are authored standing on a
// ground plane: the base of the up axis sits at (or very near) zero, whereas the
// other axes straddle the origin.
MeshOrientation detectOrientation(const ObjMesh& mesh);

struct ModelPlacement {
    std::vector<cv::Point3f> vertices;  // card-space vertex positions
    cv::Mat rotation;                   // object -> card rotation (3x3, CV_32F)
    float height_inches = 0.f;          // final height standing on the card
};

// The XY Collada rips declare <unit name="inch"/>, and the meshes are already
// sized the way they appear in battle — Caterpie ~0.76, Charizard ~4.4, not
// Pokédex metres. Default 1.0 maps one mesh unit onto one inch on the card.
constexpr float kDefaultInchesPerUnit = 1.0f;

// Newer per-Pokemon rips are authored ~40× larger (the inch↔metre mix-up).
// Convert them into XY inches before applying --scale.
constexpr float kXyUnitsPerMetre = 2.4f;
constexpr float kModernRipUnitsPerMetre = 95.f;

float nativeUnitsPerMetre(const ObjMesh& mesh);

// Stand the mesh upright on the card using its authored battle proportions.
ModelPlacement placeMeshOnCard(
    const ObjMesh& mesh,
    float card_w,
    float card_h,
    float inches_per_unit = kDefaultInchesPerUnit);
