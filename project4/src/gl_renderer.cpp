#include "gl_renderer.hpp"

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <opencv2/calib3d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <cmath>
#include <iostream>
#include <map>
#include <memory>
#include <vector>

namespace {

struct GpuVertex {
    float position[3];
    float normal[3];
    float texcoord[2];
};

struct GpuDrawRange {
    unsigned int material_index = 0;
    int first_index = 0;
    int index_count = 0;
};

struct GpuMesh {
    unsigned int vao = 0;
    unsigned int vbo = 0;
    unsigned int ebo = 0;
    std::vector<GpuDrawRange> draw_ranges;
    std::vector<unsigned int> textures;
};

const char* kVertexShader = R"(
#version 330 core
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aNormal;
layout(location = 2) in vec2 aTexCoord;

uniform mat4 uMVP;
uniform mat3 uNormalMatrix;

out vec3 vNormal;
out vec2 vTexCoord;

void main() {
    gl_Position = uMVP * vec4(aPos, 1.0);
    vNormal = normalize(uNormalMatrix * aNormal);
    vTexCoord = aTexCoord;
}
)";

const char* kFragmentShader = R"(
#version 330 core
in vec3 vNormal;
in vec2 vTexCoord;

uniform sampler2D uTexture;
uniform vec3 uDiffuse;
uniform bool uUseTexture;

out vec4 FragColor;

void main() {
    vec3 light_dir = normalize(vec3(0.2, 0.4, 1.0));
    float diffuse = max(dot(normalize(vNormal), light_dir), 0.15);
    vec3 base = uUseTexture ? texture(uTexture, vTexCoord).rgb : uDiffuse;
    vec3 color = base * diffuse;
    FragColor = vec4(color, 1.0);
}
)";

unsigned int compileShader(unsigned int type, const char* source) {
    const unsigned int shader = glCreateShader(type);
    glShaderSource(shader, 1, &source, nullptr);
    glCompileShader(shader);

    int success = 0;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if (!success) {
        char log[1024];
        glGetShaderInfoLog(shader, sizeof(log), nullptr, log);
        std::cerr << "Shader compile error: " << log << std::endl;
        return 0;
    }
    return shader;
}

unsigned int createProgram() {
    const unsigned int vertex_shader = compileShader(GL_VERTEX_SHADER, kVertexShader);
    const unsigned int fragment_shader = compileShader(GL_FRAGMENT_SHADER, kFragmentShader);
    if (!vertex_shader || !fragment_shader) {
        return 0;
    }

    const unsigned int program = glCreateProgram();
    glAttachShader(program, vertex_shader);
    glAttachShader(program, fragment_shader);
    glLinkProgram(program);

    glDeleteShader(vertex_shader);
    glDeleteShader(fragment_shader);

    int success = 0;
    glGetProgramiv(program, GL_LINK_STATUS, &success);
    if (!success) {
        char log[1024];
        glGetProgramInfoLog(program, sizeof(log), nullptr, log);
        std::cerr << "Program link error: " << log << std::endl;
        return 0;
    }
    return program;
}

cv::Mat makeViewMatrix(const cv::Mat& rvec, const cv::Mat& tvec) {
    cv::Mat R;
    cv::Rodrigues(rvec, R);

    cv::Mat view = cv::Mat::eye(4, 4, CV_64F);
    for (int row = 0; row < 3; ++row) {
        for (int col = 0; col < 3; ++col) {
            view.at<double>(row, col) = R.at<double>(row, col);
        }
        view.at<double>(row, 3) = tvec.at<double>(row);
    }

    const cv::Mat cv_to_gl =
        (cv::Mat_<double>(4, 4) << 1, 0, 0, 0, 0, -1, 0, 0, 0, 0, -1, 0, 0, 0, 0, 1);
    return cv_to_gl * view * cv_to_gl;
}

cv::Mat makeProjectionMatrix(const cv::Mat& camera_mat, int width, int height, double near_plane, double far_plane) {
    const double fx = camera_mat.at<double>(0, 0);
    const double fy = camera_mat.at<double>(1, 1);
    const double cx = camera_mat.at<double>(0, 2);
    const double cy = camera_mat.at<double>(1, 2);

    cv::Mat projection = cv::Mat::zeros(4, 4, CV_64F);
    projection.at<double>(0, 0) = 2.0 * fx / width;
    projection.at<double>(1, 1) = 2.0 * fy / height;
    projection.at<double>(0, 2) = 1.0 - 2.0 * cx / width;
    projection.at<double>(1, 2) = 2.0 * cy / height - 1.0;
    projection.at<double>(2, 2) = -(far_plane + near_plane) / (far_plane - near_plane);
    projection.at<double>(2, 3) = -2.0 * far_plane * near_plane / (far_plane - near_plane);
    projection.at<double>(3, 2) = -1.0;
    return projection;
}

cv::Mat toGlMat4(const cv::Mat& mat) {
    cv::Mat gl_mat = mat.t();
    if (gl_mat.type() != CV_32F) {
        cv::Mat converted;
        gl_mat.convertTo(converted, CV_32F);
        return converted;
    }
    return gl_mat;
}

unsigned int loadTexture(const std::string& path) {
    cv::Mat image = cv::imread(path, cv::IMREAD_UNCHANGED);
    if (image.empty()) {
        return 0;
    }

    cv::Mat rgba;
    if (image.channels() == 4) {
        cv::cvtColor(image, rgba, cv::COLOR_BGRA2RGBA);
    } else if (image.channels() == 3) {
        cv::cvtColor(image, rgba, cv::COLOR_BGR2RGBA);
    } else {
        return 0;
    }

    unsigned int texture = 0;
    glGenTextures(1, &texture);
    glBindTexture(GL_TEXTURE_2D, texture);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexImage2D(
        GL_TEXTURE_2D,
        0,
        GL_RGBA,
        rgba.cols,
        rgba.rows,
        0,
        GL_RGBA,
        GL_UNSIGNED_BYTE,
        rgba.data);
    glBindTexture(GL_TEXTURE_2D, 0);
    return texture;
}

}  // namespace

GLRenderer::~GLRenderer() {
    cleanup();
}

bool GLRenderer::init(int width, int height) {
    if (ready_) {
        return true;
    }

    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW\n";
        return false;
    }

    glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    GLFWwindow* window = glfwCreateWindow(width, height, "project4_gl", nullptr, nullptr);
    if (!window) {
        std::cerr << "Failed to create GLFW window\n";
        glfwTerminate();
        return false;
    }

    window_ = window;
    width_ = width;
    height_ = height;

    glfwMakeContextCurrent(window);
    glewExperimental = GL_TRUE;
    if (glewInit() != GLEW_OK) {
        std::cerr << "Failed to initialize GLEW\n";
        cleanup();
        return false;
    }

    shader_program_ = createProgram();
    if (!shader_program_) {
        cleanup();
        return false;
    }

    glGenFramebuffers(1, &fbo_);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo_);

    glGenTextures(1, &color_tex_);
    glBindTexture(GL_TEXTURE_2D, color_tex_);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, color_tex_, 0);

    glGenRenderbuffers(1, &depth_rbo_);
    glBindRenderbuffer(GL_RENDERBUFFER, depth_rbo_);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, width, height);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, depth_rbo_);

    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
        std::cerr << "Framebuffer is not complete\n";
        cleanup();
        return false;
    }

    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    ready_ = true;
    return true;
}

bool GLRenderer::uploadMesh(const ObjMesh& mesh, const std::vector<cv::Point3f>& board_vertices) {
    if (!ready_) {
        return false;
    }

    cleanupMeshOnly();

    auto* gpu_mesh = new GpuMesh();
    std::vector<GpuVertex> vertices;
    std::vector<unsigned int> indices;

    std::map<std::string, unsigned int> material_to_index;
    std::vector<std::string> material_order;
    for (const auto& entry : mesh.materials) {
        material_to_index[entry.first] = static_cast<unsigned int>(material_order.size());
        material_order.push_back(entry.first);
        const std::string texture_path = mesh.base_dir + "/" + entry.second.diffuse_texture;
        gpu_mesh->textures.push_back(
            entry.second.diffuse_texture.empty() ? 0 : loadTexture(texture_path));
    }

    std::map<std::string, GpuDrawRange> draw_ranges;

    for (std::size_t tri_idx = 0; tri_idx < mesh.triangles.size(); ++tri_idx) {
        const ObjTriangle& triangle = mesh.triangles[tri_idx];
        const std::string& material_name = mesh.triangle_materials[tri_idx];

        if (!draw_ranges.count(material_name)) {
            GpuDrawRange range;
            range.material_index = material_to_index[material_name];
            range.first_index = static_cast<int>(indices.size());
            range.index_count = 0;
            draw_ranges[material_name] = range;
        }

        GpuDrawRange& range = draw_ranges[material_name];
        for (int corner = 0; corner < 3; ++corner) {
            const int vertex_index = triangle.vertices[corner];
            const cv::Point3f& position = board_vertices[vertex_index];

            cv::Point3f normal(0.f, 0.f, 1.f);
            const int normal_index = triangle.normals[corner];
            if (normal_index >= 0 && normal_index < static_cast<int>(mesh.normals.size())) {
                normal = mesh.normals[normal_index];
            }

            cv::Point2f texcoord(0.f, 0.f);
            const int texcoord_index = triangle.texcoords[corner];
            if (texcoord_index >= 0 && texcoord_index < static_cast<int>(mesh.texcoords.size())) {
                texcoord = mesh.texcoords[texcoord_index];
            }

            GpuVertex vertex{};
            vertex.position[0] = position.x;
            vertex.position[1] = position.y;
            vertex.position[2] = position.z;
            vertex.normal[0] = normal.x;
            vertex.normal[1] = normal.y;
            vertex.normal[2] = normal.z;
            vertex.texcoord[0] = texcoord.x;
            vertex.texcoord[1] = 1.0f - texcoord.y;

            vertices.push_back(vertex);
            indices.push_back(static_cast<unsigned int>(vertices.size() - 1));
            range.index_count += 1;
        }
    }

    for (const std::string& material_name : material_order) {
        if (draw_ranges.count(material_name) && draw_ranges[material_name].index_count > 0) {
            gpu_mesh->draw_ranges.push_back(draw_ranges[material_name]);
        }
    }

    if (vertices.empty() || indices.empty()) {
        delete gpu_mesh;
        return false;
    }

    glGenVertexArrays(1, &gpu_mesh->vao);
    glGenBuffers(1, &gpu_mesh->vbo);
    glGenBuffers(1, &gpu_mesh->ebo);

    glBindVertexArray(gpu_mesh->vao);

    glBindBuffer(GL_ARRAY_BUFFER, gpu_mesh->vbo);
    glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(GpuVertex), vertices.data(), GL_STATIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, gpu_mesh->ebo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(unsigned int), indices.data(), GL_STATIC_DRAW);

    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(GpuVertex), reinterpret_cast<void*>(0));
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(
        1, 3, GL_FLOAT, GL_FALSE, sizeof(GpuVertex), reinterpret_cast<void*>(offsetof(GpuVertex, normal)));
    glEnableVertexAttribArray(2);
    glVertexAttribPointer(
        2, 2, GL_FLOAT, GL_FALSE, sizeof(GpuVertex), reinterpret_cast<void*>(offsetof(GpuVertex, texcoord)));

    glBindVertexArray(0);
    gpu_mesh_ = gpu_mesh;
    return true;
}

bool GLRenderer::renderOverlay(
    const cv::Mat& camera_mat,
    const cv::Mat& rvec,
    const cv::Mat& tvec,
    cv::Mat& frame) {
    if (!ready_ || gpu_mesh_ == nullptr || frame.empty()) {
        return false;
    }

    GpuMesh* gpu_mesh = static_cast<GpuMesh*>(gpu_mesh_);

    GLFWwindow* window = static_cast<GLFWwindow*>(window_);
    glfwMakeContextCurrent(window);

    glBindFramebuffer(GL_FRAMEBUFFER, fbo_);
    glViewport(0, 0, width_, height_);
    glEnable(GL_DEPTH_TEST);
    glClearColor(0.f, 0.f, 0.f, 0.f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    const cv::Mat view = makeViewMatrix(rvec, tvec);
    const cv::Mat projection = makeProjectionMatrix(camera_mat, width_, height_, 0.1, 100.0);
    const cv::Mat model = cv::Mat::eye(4, 4, CV_64F);
    const cv::Mat mvp = projection * view * model;

    cv::Mat normal_matrix = cv::Mat::eye(3, 3, CV_32F);

    glUseProgram(shader_program_);
    glUniformMatrix4fv(glGetUniformLocation(shader_program_, "uMVP"), 1, GL_FALSE, toGlMat4(mvp).ptr<float>());
    glUniformMatrix3fv(
        glGetUniformLocation(shader_program_, "uNormalMatrix"), 1, GL_FALSE, normal_matrix.ptr<float>());

    glBindVertexArray(gpu_mesh->vao);
    for (const GpuDrawRange& range : gpu_mesh->draw_ranges) {
        unsigned int texture_id = 0;
        if (range.material_index < gpu_mesh->textures.size()) {
            texture_id = gpu_mesh->textures[range.material_index];
        }

        const bool use_texture = texture_id != 0;
        glUniform1i(glGetUniformLocation(shader_program_, "uUseTexture"), use_texture ? 1 : 0);
        glUniform3f(glGetUniformLocation(shader_program_, "uDiffuse"), 0.8f, 0.8f, 0.8f);

        if (use_texture) {
            glActiveTexture(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_2D, texture_id);
            glUniform1i(glGetUniformLocation(shader_program_, "uTexture"), 0);
        }

        glDrawElements(
            GL_TRIANGLES,
            range.index_count,
            GL_UNSIGNED_INT,
            reinterpret_cast<void*>(static_cast<std::uintptr_t>(range.first_index) * sizeof(unsigned int)));
    }
    glBindVertexArray(0);

    cv::Mat rgba(height_, width_, CV_8UC4);
    glReadPixels(0, 0, width_, height_, GL_RGBA, GL_UNSIGNED_BYTE, rgba.data);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    cv::Mat flipped;
    cv::flip(rgba, flipped, 0);

    cv::Mat overlay_bgra;
    cv::cvtColor(flipped, overlay_bgra, cv::COLOR_RGBA2BGRA);

    for (int row = 0; row < frame.rows; ++row) {
        cv::Vec4b* overlay_row = overlay_bgra.ptr<cv::Vec4b>(row);
        cv::Vec3b* frame_row = frame.ptr<cv::Vec3b>(row);
        for (int col = 0; col < frame.cols; ++col) {
            const cv::Vec4b& pixel = overlay_row[col];
            if (pixel[3] == 0) {
                continue;
            }
            const float alpha = pixel[3] / 255.f;
            for (int channel = 0; channel < 3; ++channel) {
                frame_row[col][channel] = static_cast<uchar>(
                    (1.f - alpha) * frame_row[col][channel] + alpha * pixel[channel]);
            }
        }
    }

    return true;
}

void GLRenderer::cleanupMeshOnly() {
    if (gpu_mesh_ == nullptr) {
        return;
    }

    GpuMesh* gpu_mesh = static_cast<GpuMesh*>(gpu_mesh_);

    if (gpu_mesh->vao) {
        glDeleteVertexArrays(1, &gpu_mesh->vao);
    }
    if (gpu_mesh->vbo) {
        glDeleteBuffers(1, &gpu_mesh->vbo);
    }
    if (gpu_mesh->ebo) {
        glDeleteBuffers(1, &gpu_mesh->ebo);
    }
    for (unsigned int texture : gpu_mesh->textures) {
        if (texture) {
            glDeleteTextures(1, &texture);
        }
    }

    delete gpu_mesh;
    gpu_mesh_ = nullptr;
}

void GLRenderer::cleanup() {
    if (!ready_ && window_ == nullptr) {
        return;
    }

    if (ready_) {
        cleanupMeshOnly();
    }

    if (shader_program_) {
        glDeleteProgram(shader_program_);
        shader_program_ = 0;
    }
    if (color_tex_) {
        glDeleteTextures(1, &color_tex_);
        color_tex_ = 0;
    }
    if (depth_rbo_) {
        glDeleteRenderbuffers(1, &depth_rbo_);
        depth_rbo_ = 0;
    }
    if (fbo_) {
        glDeleteFramebuffers(1, &fbo_);
        fbo_ = 0;
    }

    if (window_) {
        glfwDestroyWindow(static_cast<GLFWwindow*>(window_));
        window_ = nullptr;
    }

    glfwTerminate();
    ready_ = false;
}
