#include "obj_loader.hpp"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <limits>
#include <sstream>
#include <unordered_set>
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

}  // namespace

bool loadObjMesh(const std::string& obj_path, ObjMesh& mesh) {
    mesh = ObjMesh{};
    mesh.base_dir = parentDirectory(obj_path);
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
    mesh.materials[current_material] = ObjMaterial{current_material, {1.f, 1.f, 1.f}, ""};

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
                mesh.materials[current_material] = ObjMaterial{current_material, {1.f, 1.f, 1.f}, ""};
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
                    mesh.materials[active_material] = ObjMaterial{active_material, {1.f, 1.f, 1.f}, ""};
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
