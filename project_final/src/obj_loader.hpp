#pragma once

#include <opencv2/core.hpp>

#include <map>
#include <string>
#include <vector>

struct ObjMaterial {
    std::string name;
    cv::Vec3f diffuse{1.f, 1.f, 1.f};
    // glTF baseColorFactor / Assimp opacity. Untextured jelly shells (Solosis)
    // ship RGB green with a < 1 and alphaMode BLEND — without this they render
    // as solid Kd.
    float opacity = 1.f;
    std::string diffuse_texture;
    std::string normal_texture;
    std::string specular_texture;
    std::string color2_texture;
    // Pokemon XY "*Mask.png". Only used as a cutout when it carries a real alpha
    // channel (Charizard's flame); the RGB variants are toon shading masks.
    std::string alpha_mask_texture;
    // Iris decals are coplanar duplicates of the eye quad; they are dropped when
    // the eye material already uses a pre-composited "*_Merged" texture (XY).
    bool skip_draw = false;
    // Unlit particle sheet (Charizard's flame, Weezing's gas) rather than albedo:
    // a colourless diffuse map with no normal-map sibling.
    bool is_effect = false;
    // poke-3D eye sheet: discard atlas-field texels but draw in the opaque pass.
    bool alpha_clip = false;
    // poke-3D eye stack: 0 = none, 1 = sclera (back), 2 = iris, 3 = eyelid (front).
    int eye_layer = 0;
};

struct ObjTriangle {
    cv::Vec3i vertices;
    cv::Vec3i texcoords;
    cv::Vec3i normals;
};

struct ObjMesh {
    std::vector<cv::Point3f> vertices;
    std::vector<cv::Point3f> normals;
    std::vector<cv::Point2f> texcoords;
    // Per-vertex RGB (1:1 with vertices). poke-3D models like Baltoy ship
    // colour as COLOR_0 instead of a diffuse texture; default is white.
    // The renderer only multiplies these in when the material has no diffuse map
    // — some textured GLBs carry a constant ~0.5 COLOR_0 that would darken albedo.
    std::vector<cv::Vec3f> colors;
    std::vector<ObjTriangle> triangles;
    std::vector<std::string> triangle_materials;
    std::map<std::string, ObjMaterial> materials;
    std::string base_dir;
    // File this mesh came from; identifies which rip's unit system it uses.
    std::string source_path;

    cv::Point3f min_bounds{};
    cv::Point3f max_bounds{};
    // Bounds of the Pokemon itself, ignoring effect meshes. Weezing's gas volume
    // is twice the size of its body and would otherwise dominate both the
    // up-axis detection and the on-card scale.
    cv::Point3f body_min_bounds{};
    cv::Point3f body_max_bounds{};
};

// Turns a texture reference from a model file into a real path, or "" if it is not
// on disk. Handles the quirks across rips: leading slashes ("/Textures/x.png"),
// backslashes, wrong case (Mewtwo's DAE asks for "iris1.png", the file is
// "Iris1.png") and textures sitting in an images/ or Textures/ subfolder.
std::string resolveAssetPath(const std::string& base_dir, const std::string& relative);

// True for poke-3D assets. They differ from the XY rips in enough places
// (effect volumes belong to the silhouette, materials carry glTF alpha) that
// callers downstream of the loader still need to tell them apart.
bool isGlbPath(const std::string& path);

bool loadObjMesh(const std::string& obj_path, ObjMesh& mesh);
// Load FBX/DAE/OBJ/etc via Assimp into the same ObjMesh used by GLRenderer.
bool loadAssimpMesh(const std::string& path, ObjMesh& mesh);
std::vector<cv::Point3f> transformMeshVertices(
    const ObjMesh& mesh,
    const cv::Point3f& center,
    float scale,
    const cv::Mat& rotation = cv::Mat::eye(3, 3, CV_32F));
std::vector<std::pair<int, int>> buildUniqueEdges(const ObjMesh& mesh);
