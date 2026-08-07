#include "obj_loader.hpp"

#include <assimp/Importer.hpp>
#include <assimp/postprocess.h>
#include <assimp/scene.h>

#include <opencv2/imgcodecs.hpp>

#include <algorithm>
#include <cctype>
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
    return cv::countNonZero(channels[0] != channels[1]) == 0 &&
           cv::countNonZero(channels[1] != channels[2]) == 0;
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
        sub.name = aimesh->mName.C_Str();
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

    // Materials
    for (unsigned int mi = 0; mi < scene->mNumMaterials; ++mi) {
        const aiMaterial* aimat = scene->mMaterials[mi];
        const std::string name = "mat_" + std::to_string(mi);
        ObjMaterial mat;
        mat.name = name;

        aiColor3D diffuse(0.8f, 0.8f, 0.8f);
        if (aimat->Get(AI_MATKEY_COLOR_DIFFUSE, diffuse) == AI_SUCCESS) {
            mat.diffuse = cv::Vec3f(diffuse.r, diffuse.g, diffuse.b);
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
            mat.diffuse_texture = tex_path;
        }
        resolvePokemonTextureSet(mat, mesh.base_dir);
        mesh.materials[name] = mat;
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

    for (const SubMesh& sub : sub_meshes) {
        if (sub.skip) {
            std::cerr << "Skipping alternate part '" << sub.name << "' in " << path << "\n";
            continue;
        }
        append_mesh(sub.mesh, sub.transform);
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
