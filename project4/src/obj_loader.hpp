#pragma once

#include <opencv2/core.hpp>

#include <map>
#include <string>
#include <vector>

struct ObjMaterial {
    std::string name;
    cv::Vec3f diffuse{1.f, 1.f, 1.f};
    std::string diffuse_texture;
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
    std::vector<ObjTriangle> triangles;
    std::vector<std::string> triangle_materials;
    std::map<std::string, ObjMaterial> materials;
    std::string base_dir;

    cv::Point3f min_bounds{};
    cv::Point3f max_bounds{};
};

bool loadObjMesh(const std::string& obj_path, ObjMesh& mesh);
std::vector<cv::Point3f> transformMeshVertices(
    const ObjMesh& mesh,
    const cv::Point3f& center,
    float scale,
    const cv::Mat& rotation = cv::Mat::eye(3, 3, CV_32F));
std::vector<std::pair<int, int>> buildUniqueEdges(const ObjMesh& mesh);
