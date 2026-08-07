#pragma once

#include "obj_loader.hpp"

#include <opencv2/core.hpp>

#include <string>
#include <vector>

// What actually made it onto the GPU for one material. Used by tools/validate_models
// to check every asset imports cleanly instead of eyeballing renders.
struct TextureSlotReport {
    std::string slot;    // "diffuse", "normal", "specular", "color2", "mask"
    std::string path;    // resolved relative path, empty when the material has none
    // "ok" bound to the shader, "missing" the file would not decode,
    // "ignored" decoded but deliberately unused (an RGB "*Mask" is not a cutout).
    std::string status;
};

struct MaterialReport {
    std::string material;
    int triangles = 0;
    bool drawn = false;      // false when the material is deliberately skipped
    bool additive = false;
    bool has_alpha = false;
    std::vector<TextureSlotReport> textures;
};

class GLRenderer {
public:
    GLRenderer() = default;
    ~GLRenderer();

    bool init(int width, int height);
    // normal_rotation: same 3x3 used to place the mesh into card space (object → card).
    bool uploadMesh(
        const ObjMesh& mesh,
        const std::vector<cv::Point3f>& board_vertices,
        const cv::Mat& normal_rotation = cv::Mat::eye(3, 3, CV_32F));
    bool renderOverlay(
        const cv::Mat& camera_mat,
        const cv::Mat& rvec,
        const cv::Mat& tvec,
        cv::Mat& frame);
    void cleanup();

    bool isReady() const { return ready_; }

    std::vector<MaterialReport> materialReports() const;
    // Diagnostics: draw only this material. Empty string restores normal drawing.
    void setMaterialFilter(const std::string& material_name) { material_filter_ = material_name; }

private:
    void cleanupMeshOnly();

    bool ready_ = false;
    int width_ = 0;
    int height_ = 0;
    void* window_ = nullptr;
    unsigned int fbo_ = 0;
    unsigned int color_tex_ = 0;
    unsigned int depth_rbo_ = 0;
    unsigned int shader_program_ = 0;
    void* gpu_mesh_ = nullptr;
    std::string material_filter_;
};
