#pragma once

#include "obj_loader.hpp"

#include <opencv2/core.hpp>

#include <string>

class GLRenderer {
public:
    GLRenderer() = default;
    ~GLRenderer();

    bool init(int width, int height);
    bool uploadMesh(const ObjMesh& mesh, const std::vector<cv::Point3f>& board_vertices);
    bool renderOverlay(
        const cv::Mat& camera_mat,
        const cv::Mat& rvec,
        const cv::Mat& tvec,
        cv::Mat& frame);
    void cleanup();

    bool isReady() const { return ready_; }

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
};
