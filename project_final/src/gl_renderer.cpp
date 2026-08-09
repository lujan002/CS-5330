#include "gl_renderer.hpp"

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <opencv2/calib3d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <map>
#include <set>
#include <vector>

namespace {

struct GpuVertex {
    float position[3];
    float normal[3];
    float texcoord[2];
    float color[3];
};

struct GpuDrawRange {
    unsigned int material_index = 0;
    int first_index = 0;
    int index_count = 0;
};

struct GpuMaterialTextures {
    unsigned int diffuse = 0;
    unsigned int normal = 0;
    unsigned int specular = 0;
    unsigned int color2 = 0;
    unsigned int alpha_mask = 0;
    bool has_alpha = false;
    // Discard low-alpha texels but stay in the opaque pass (poke-3D eye sheets).
    bool alpha_clip = false;
    // poke-3D eye stack: 0 = none, 1 = sclera, 2 = iris, 3 = lid.
    int eye_layer = 0;
    // A face decal that did not need splitting into an eye stack, so it is still
    // a flat quad sitting exactly on the head (Mewtwo's "l_eye"). Coplanar with
    // the skull, it loses GL_LESS and vanishes without a bias.
    bool face_decal = false;
    // Unlit intensity map (Charizard's tail flame) blended additively.
    bool additive = false;
    // Flat eye plates shade darker than the curved head under the soft key;
    // lift ambient so the socket doesn't read as a darker polygon.
    bool flatten_light = false;
    cv::Vec3f tint{1.f, 1.f, 1.f};
};

struct GpuMesh {
    unsigned int vao = 0;
    unsigned int vbo = 0;
    unsigned int ebo = 0;
    std::vector<GpuDrawRange> draw_ranges;
    std::vector<GpuMaterialTextures> textures;
    std::vector<cv::Vec3f> diffuse_colors;
    std::vector<float> diffuse_opacity;
    std::vector<std::string> material_names;
    std::vector<MaterialReport> reports;
    // Object/mesh space → card space (same rotation used for vertex placement).
    cv::Mat object_to_card = cv::Mat::eye(3, 3, CV_32F);
};

const char* kVertexShader = R"(
#version 330 core
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aNormal;
layout(location = 2) in vec2 aTexCoord;
layout(location = 3) in vec3 aColor;

uniform mat4 uMVP;
uniform mat4 uView;

out vec3 vNormalObject;
out vec3 vViewPos;
out vec2 vTexCoord;
out vec3 vColor;

void main() {
    gl_Position = uMVP * vec4(aPos, 1.0);
    vNormalObject = aNormal;
    vViewPos = (uView * vec4(aPos, 1.0)).xyz;
    vTexCoord = aTexCoord;
    vColor = aColor;
}
)";

const char* kFragmentShader = R"(
#version 330 core
in vec3 vNormalObject;
in vec3 vViewPos;
in vec2 vTexCoord;
in vec3 vColor;

uniform sampler2D uDiffuseMap;
uniform sampler2D uNormalMap;
uniform sampler2D uSpecularMap;
uniform sampler2D uColor2Map;
uniform sampler2D uAlphaMask;

uniform vec3 uDiffuse;
uniform float uOpacity;
uniform bool uUseDiffuseMap;
uniform bool uUseNormalMap;
uniform bool uUseSpecularMap;
uniform bool uUseColor2Map;
uniform bool uUseAlphaMask;
uniform bool uAdditive;
uniform bool uFlattenLight;
uniform bool uUseTextureAlpha;
uniform vec3 uEffectTint;
uniform mat3 uNormalMatrix;

out vec4 FragColor;

void main() {
    vec4 diffuse_sample = uUseDiffuseMap ? texture(uDiffuseMap, vTexCoord)
                                         : vec4(uDiffuse, uOpacity);
    // Vertex colours tint untextured meshes (Baltoy). Several poke-3D GLBs also
    // ship a constant ~0.5 COLOR_0 on top of a real albedo (Abomasnow, Quilava);
    // multiplying that in halves the texture, so only apply COLOR_0 when there
    // is no diffuse map.
    if (!uUseDiffuseMap) {
        diffuse_sample.rgb *= vColor;
    }
    if (uUseAlphaMask) {
        diffuse_sample.a *= texture(uAlphaMask, vTexCoord).a;
    }

    // Particle sheets are unlit: brightness drives both colour and coverage,
    // ramping from the tint toward a hot white core. Output is premultiplied.
    if (uAdditive) {
        float intensity = diffuse_sample.r * diffuse_sample.a;
        if (intensity < 0.03) {
            discard;
        }
        vec3 glow = mix(uEffectTint, vec3(1.0, 0.95, 0.70), smoothstep(0.55, 1.0, intensity));
        FragColor = vec4(glow * intensity, intensity);
        return;
    }

    // poke-3D body atlases often ship film opacity in alpha (Greedent/Skwovet
    // ~150, Yanmega wings stuck at 92) without being cutouts. Blending is always
    // on, so writing that alpha makes solid fur look ghostly. Only honour the
    // texture alpha when loadTexture classified it as a real 0..255 cutout.
    if (!uUseTextureAlpha) {
        diffuse_sample.a = uOpacity;
    }

    // Iris (and other cutout maps) store transparency in alpha; RGB is black there.
    if (diffuse_sample.a < 0.08) {
        discard;
    }

    // Vertex normals are reliable after Assimp + placement. Object-space XY
    // normal maps often disagree with the transformed mesh and create blotchy
    // shading under AR camera angles, so they only add a light hint of relief.
    vec3 N = normalize(uNormalMatrix * normalize(vNormalObject));
    if (uUseNormalMap) {
        vec3 n_tex = texture(uNormalMap, vTexCoord).rgb * 2.0 - 1.0;
        vec3 N_map = normalize(uNormalMatrix * normalize(n_tex));
        N = normalize(mix(N, N_map, 0.15));
    }

    // Albedo is Color1 only. Color2 in these rips is a cool-toned toon shade
    // (Raticate's Body2 is blue-grey against an orange Body1); softly mixing it
    // muddies the colours under AR viewing angles, so it is left unbound.
    vec3 base = diffuse_sample.rgb;

    // Soft key + fill in view space. Ambient-heavy so the model stays readable
    // from above the card without camera-dependent blotches. Eye plates get an
    // even flatter light so a coplanar socket doesn't shade darker than the head.
    vec3 key_dir = normalize(vec3(0.25, 0.55, 0.80));
    vec3 fill_dir = normalize(vec3(-0.40, 0.20, 0.55));
    float key_n_dot_l = max(dot(N, key_dir), 0.0);
    float fill_n_dot_l = max(dot(N, fill_dir), 0.0);
    float light = uFlattenLight
        ? clamp(0.88 + 0.08 * key_n_dot_l + 0.04 * fill_n_dot_l, 0.0, 1.0)
        : clamp(0.62 + 0.28 * key_n_dot_l + 0.12 * fill_n_dot_l, 0.0, 1.0);

    float specular_mask = uUseSpecularMap ? texture(uSpecularMap, vTexCoord).r : 0.15;
    vec3 V = normalize(-vViewPos);
    vec3 H_key = normalize(key_dir + V);
    float spec = pow(max(dot(N, H_key), 0.0), 48.0) * specular_mask * 0.18;
    if (uFlattenLight) {
        spec *= 0.25;
    }

    float rim = pow(1.0 - max(dot(N, V), 0.0), 3.0) * 0.08;
    vec3 rim_color = vec3(0.35, 0.45, 0.55);

    vec3 lit = base * light + vec3(spec) + rim * rim_color;
    FragColor = vec4(lit, diffuse_sample.a);
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

    // OpenCV cameras look down +Z with +Y down; OpenGL looks down -Z with +Y up.
    // Only the camera-space result is flipped — object space must be left alone.
    const cv::Mat cv_to_gl =
        (cv::Mat_<double>(4, 4) << 1, 0, 0, 0, 0, -1, 0, 0, 0, 0, -1, 0, 0, 0, 0, 1);
    return cv_to_gl * view;
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

cv::Mat toGlMat3(const cv::Mat& mat) {
    cv::Mat m32;
    if (mat.type() != CV_32F) {
        mat.convertTo(m32, CV_32F);
    } else {
        m32 = mat;
    }
    // OpenGL wants column-major; upload with transpose=false after .t().
    return m32.t();
}

struct TextureInfo {
    unsigned int id = 0;
    bool has_alpha = false;
    // True when alpha varies enough to be a cutout, not a flat film opacity.
    // Beedrill's wing Mask is RGBA with alpha stuck at ~93; treating that as a
    // cutout made the wings ghostly. Charizard's flame Mask spans 0..255.
    bool alpha_varies = false;
    bool grayscale = false;
};

TextureInfo loadTexture(const std::string& path) {
    TextureInfo info;

    cv::Mat image = cv::imread(path, cv::IMREAD_UNCHANGED);
    if (image.empty()) {
        return info;
    }

    cv::Mat rgba;
    if (image.channels() == 4) {
        cv::cvtColor(image, rgba, cv::COLOR_BGRA2RGBA);
        // Cutout/transparent maps (iris, flame) have non-opaque texels. A channel
        // that is merely "not 255" is not enough: Beedrill's wing Mask is a flat
        // ~93 everywhere, which is a film opacity, not a cutout shape.
        double min_a = 0.0;
        double max_a = 0.0;
        cv::Mat alpha;
        cv::extractChannel(rgba, alpha, 3);
        cv::minMaxLoc(alpha, &min_a, &max_a);
        // True cutouts span near-0 to near-255 (iris, flame). Mid-range film
        // opacity (Greedent eyes ~150, Beedrill wings ~93) is not transparency.
        cv::Scalar mean;
        cv::Scalar stddev;
        cv::meanStdDev(alpha, mean, stddev);
        info.alpha_varies = stddev[0] > 15.0;
        info.has_alpha = info.alpha_varies && min_a <= 25.0 && max_a >= 230.0;
        // Film opacity (Skwovet/Greedent body ~150, Yanmega wings stuck at 92)
        // is not a cutout. Keep authored RGB but force the atlas opaque so
        // blending cannot ghost the card through solid fur/wings.
        if (!info.has_alpha && min_a < 250.0) {
            std::vector<cv::Mat> ch;
            cv::split(rgba, ch);
            ch[3].setTo(255);
            cv::merge(ch, rgba);
        }
    } else if (image.channels() == 3) {
        cv::cvtColor(image, rgba, cv::COLOR_BGR2RGBA);
    } else if (image.channels() == 1) {
        cv::Mat rgb;
        cv::cvtColor(image, rgb, cv::COLOR_GRAY2RGB);
        cv::cvtColor(rgb, rgba, cv::COLOR_RGB2RGBA);
    } else {
        return info;
    }

    // A perfectly colourless map is an intensity map (flame noise), not albedo.
    std::vector<cv::Mat> channels;
    cv::split(rgba, channels);
    info.grayscale = cv::countNonZero(channels[0] != channels[1]) == 0 &&
                     cv::countNonZero(channels[1] != channels[2]) == 0;

    unsigned int texture = 0;
    glGenTextures(1, &texture);
    glBindTexture(GL_TEXTURE_2D, texture);
    // poke-3D palette atlases are tiny (Breloom is 4×32) with real 0/255 cutout
    // alpha. Linear filtering bleeds the transparent padding into neighbouring
    // colours and the snout/face reads as a ghost. Nearest keeps the authored
    // blocks opaque.
    const bool palette_atlas = rgba.cols <= 64 && rgba.rows <= 64;
    const int filter = palette_atlas ? GL_NEAREST : GL_LINEAR;
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, filter);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, filter);
    // Eye/Mouth/Iris (and BodyB) UVs sit outside [0,1] on the XY atlas; clamp
    // pins them to the black border. Repeat matches the game's wrap sampling.
    // Palettes must clamp: repeating a 4px-wide strip samples the a=0 gutter.
    const int wrap = palette_atlas ? GL_CLAMP_TO_EDGE : GL_REPEAT;
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, wrap);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, wrap);
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
    info.id = texture;
    return info;
}

TextureInfo loadMaterialTexture(const std::string& base_dir, const std::string& relative) {
    const std::string path = resolveAssetPath(base_dir, relative);
    if (path.empty()) {
        return TextureInfo();
    }
    return loadTexture(path);
}

bool containsInsensitive(const std::string& haystack, const std::string& needle) {
    const auto at = std::search(
        haystack.begin(), haystack.end(), needle.begin(), needle.end(),
        [](unsigned char a, unsigned char b) {
            return std::tolower(a) == std::tolower(b);
        });
    return at != haystack.end();
}

void bindTexture2D(unsigned int unit, unsigned int texture_id, int uniform_loc, int sampler_unit) {
    glUniform1i(uniform_loc, sampler_unit);
    glActiveTexture(GL_TEXTURE0 + unit);
    glBindTexture(GL_TEXTURE_2D, texture_id != 0 ? texture_id : 0);
}

void deleteTexture(unsigned int& texture) {
    if (texture) {
        glDeleteTextures(1, &texture);
        texture = 0;
    }
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

    GLFWwindow* window = glfwCreateWindow(width, height, "ar_card_gl", nullptr, nullptr);
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

std::vector<MaterialReport> GLRenderer::materialReports() const {
    if (gpu_mesh_ == nullptr) {
        return {};
    }
    return static_cast<const GpuMesh*>(gpu_mesh_)->reports;
}

bool GLRenderer::uploadMesh(
    const ObjMesh& mesh,
    const std::vector<cv::Point3f>& board_vertices,
    const cv::Mat& normal_rotation) {
    if (!ready_) {
        return false;
    }

    cleanupMeshOnly();

    auto* gpu_mesh = new GpuMesh();
    if (normal_rotation.rows == 3 && normal_rotation.cols == 3) {
        normal_rotation.convertTo(gpu_mesh->object_to_card, CV_32F);
    }

    std::vector<GpuVertex> vertices;
    std::vector<unsigned int> indices;

    // Per-material extent in card space, needed below to tell a localised effect
    // (Charizard's tail flame) from an aura volume that swallows the whole model.
    std::map<std::string, std::pair<cv::Point3f, cv::Point3f>> material_extent;
    for (std::size_t tri_idx = 0; tri_idx < mesh.triangles.size(); ++tri_idx) {
        const std::string& material_name = mesh.triangle_materials[tri_idx];
        auto slot = material_extent.find(material_name);
        if (slot == material_extent.end()) {
            slot = material_extent
                       .emplace(material_name,
                                std::make_pair(cv::Point3f(1e30f, 1e30f, 1e30f),
                                               cv::Point3f(-1e30f, -1e30f, -1e30f)))
                       .first;
        }
        for (int corner = 0; corner < 3; ++corner) {
            const int vertex_index = mesh.triangles[tri_idx].vertices[corner];
            if (vertex_index < 0 || vertex_index >= static_cast<int>(board_vertices.size())) {
                continue;
            }
            const cv::Point3f& p = board_vertices[vertex_index];
            cv::Point3f& lo = slot->second.first;
            cv::Point3f& hi = slot->second.second;
            lo.x = std::min(lo.x, p.x); hi.x = std::max(hi.x, p.x);
            lo.y = std::min(lo.y, p.y); hi.y = std::max(hi.y, p.y);
            lo.z = std::min(lo.z, p.z); hi.z = std::max(hi.z, p.z);
        }
    }
    auto diagonal = [](const std::pair<cv::Point3f, cv::Point3f>& box) {
        const cv::Point3f d = box.second - box.first;
        return std::sqrt(d.x * d.x + d.y * d.y + d.z * d.z);
    };

    std::map<std::string, unsigned int> material_to_index;
    std::vector<std::string> material_order;
    std::set<std::string> skipped_materials;
    for (const auto& entry : mesh.materials) {
        material_to_index[entry.first] = static_cast<unsigned int>(material_order.size());
        material_order.push_back(entry.first);

        if (entry.second.skip_draw) {
            skipped_materials.insert(entry.first);
        }

        MaterialReport report;
        report.material = entry.first;
        report.drawn = !entry.second.skip_draw;
        auto note = [&](const char* slot, const std::string& path, const char* status) {
            if (!path.empty()) {
                report.textures.push_back(TextureSlotReport{slot, path, status});
            }
        };
        auto bound = [](unsigned int id) { return id != 0 ? "ok" : "missing"; };

        GpuMaterialTextures maps;
        const TextureInfo diffuse_info =
            loadMaterialTexture(mesh.base_dir, entry.second.diffuse_texture);
        maps.diffuse = diffuse_info.id;
        maps.has_alpha = diffuse_info.has_alpha;
        maps.normal = loadMaterialTexture(mesh.base_dir, entry.second.normal_texture).id;
        maps.specular = loadMaterialTexture(mesh.base_dir, entry.second.specular_texture).id;
        maps.color2 = loadMaterialTexture(mesh.base_dir, entry.second.color2_texture).id;
        note("diffuse", entry.second.diffuse_texture, bound(maps.diffuse));
        note("normal", entry.second.normal_texture, bound(maps.normal));
        note("specular", entry.second.specular_texture, bound(maps.specular));
        note("color2", entry.second.color2_texture, bound(maps.color2));

        // Only RGBA "*Mask" maps with a varying alpha channel are cutouts
        // (Charizard's flame). RGB masks (Venusaur) are toon shading, and some
        // RGBA masks (Beedrill's wings) carry a flat film opacity rather than a
        // cutout shape -- both of those are dropped.
        const TextureInfo mask_info =
            loadMaterialTexture(mesh.base_dir, entry.second.alpha_mask_texture);
        if (mask_info.has_alpha && mask_info.alpha_varies) {
            maps.alpha_mask = mask_info.id;
            maps.has_alpha = true;
            note("mask", entry.second.alpha_mask_texture, "ok");
        } else if (mask_info.id != 0) {
            unsigned int unused = mask_info.id;
            deleteTexture(unused);
            note("mask", entry.second.alpha_mask_texture, "ignored");
        } else {
            note("mask", entry.second.alpha_mask_texture, "missing");
        }

        // Classified by the loader, which needs the same answer to keep effect
        // volumes out of the bounds that drive orientation and scale.
        maps.additive = entry.second.is_effect;
        if (maps.additive) {
            maps.has_alpha = true;
            // Embedded GLB textures lose the Fire* filename; check material name.
            const std::string& label = entry.second.name;
            const bool fire =
                entry.second.diffuse_texture.find("Fire") != std::string::npos ||
                label.find("Fire") != std::string::npos ||
                label.find("fire") != std::string::npos;
            if (maps.diffuse == 0) {
                // Untextured overlay: its own baseColor is the glow colour.
                maps.tint = entry.second.diffuse;
            } else {
                maps.tint = fire ? cv::Vec3f(1.00f, 0.42f, 0.10f)
                                 : cv::Vec3f(1.00f, 1.00f, 1.00f);
            }
        }
        maps.alpha_clip = entry.second.alpha_clip;
        maps.eye_layer = entry.second.eye_layer;
        const bool eye_name = containsInsensitive(entry.second.name, "eye") ||
                              containsInsensitive(entry.second.name, "iris") ||
                              containsInsensitive(entry.second.name, "pupil");
        maps.flatten_light = maps.eye_layer != 0 || eye_name;
        maps.face_decal = eye_name && maps.eye_layer == 0 && !maps.additive;
        if (maps.eye_layer != 0) {
            maps.has_alpha = true;
        }
        // Untextured jelly shells (Solosis) use baseColor alpha < 1.
        if (entry.second.opacity < 0.999f) {
            maps.has_alpha = true;
        }
        // Alpha-tested eye sheets keep opaque-pass ordering so brow/mane drawn
        // later can occlude them — matching XY Eye1 in the opaque body pass.
        if (maps.alpha_clip) {
            maps.has_alpha = false;
        }
        report.additive = maps.additive;
        report.has_alpha = maps.has_alpha || maps.alpha_clip || maps.eye_layer != 0;
        gpu_mesh->reports.push_back(report);
        gpu_mesh->material_names.push_back(entry.first);
        gpu_mesh->textures.push_back(maps);
        gpu_mesh->diffuse_colors.push_back(entry.second.diffuse);
        gpu_mesh->diffuse_opacity.push_back(entry.second.opacity);
    }

    // Effect meshes in the XY rips are sized for the moves that trigger them,
    // not for the idle pose: Weezing's "FireSten" gas volume is a sphere that
    // swallows the whole Pokemon. Anything that big is hidden there.
    //
    // poke-3D is exempt: its effect geometry is part of the silhouette rather
    // than a move volume, and Rapidash's mane and tail are a single fire sheet
    // taller than the body it is attached to.
    const bool glb_source = isGlbPath(mesh.source_path);
    float body_diagonal = 0.f;
    for (std::size_t i = 0; i < material_order.size(); ++i) {
        if (!gpu_mesh->textures[i].additive) {
            const auto extent = material_extent.find(material_order[i]);
            if (extent != material_extent.end()) {
                body_diagonal = std::max(body_diagonal, diagonal(extent->second));
            }
        }
    }
    for (std::size_t i = 0; i < material_order.size() && !glb_source; ++i) {
        if (!gpu_mesh->textures[i].additive || body_diagonal <= 0.f) {
            continue;
        }
        const auto extent = material_extent.find(material_order[i]);
        if (extent != material_extent.end() && diagonal(extent->second) > body_diagonal) {
            skipped_materials.insert(material_order[i]);
            gpu_mesh->reports[i].drawn = false;
        }
    }

    // Bucket by material before emitting: a draw range is one contiguous slice of
    // the index buffer, but a model's triangles arrive interleaved by sub-mesh
    // (Gengar's mask, body, eyes and tongue all take turns).
    std::map<std::string, std::vector<std::size_t>> triangles_by_material;
    for (std::size_t tri_idx = 0; tri_idx < mesh.triangles.size(); ++tri_idx) {
        const std::string& material_name = mesh.triangle_materials[tri_idx];

        const auto material_slot = material_to_index.find(material_name);
        if (material_slot != material_to_index.end()) {
            gpu_mesh->reports[material_slot->second].triangles += 1;
        }

        if (skipped_materials.count(material_name)) {
            continue;
        }
        triangles_by_material[material_name].push_back(tri_idx);
    }

    for (const auto& bucket : triangles_by_material) {
        const std::string& material_name = bucket.first;
        GpuDrawRange range;
        range.material_index = material_to_index[material_name];
        range.first_index = static_cast<int>(indices.size());
        range.index_count = 0;

        for (const std::size_t tri_idx : bucket.second) {
            const ObjTriangle& triangle = mesh.triangles[tri_idx];
            for (int corner = 0; corner < 3; ++corner) {
                const int vertex_index = triangle.vertices[corner];
                const cv::Point3f& position = board_vertices[vertex_index];

                // Keep normals in object/mesh space; uNormalMatrix applies object→view.
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

                cv::Vec3f color(1.f, 1.f, 1.f);
                if (vertex_index >= 0 && vertex_index < static_cast<int>(mesh.colors.size())) {
                    color = mesh.colors[vertex_index];
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
                vertex.color[0] = color[0];
                vertex.color[1] = color[1];
                vertex.color[2] = color[2];

                vertices.push_back(vertex);
                indices.push_back(static_cast<unsigned int>(vertices.size() - 1));
                range.index_count += 1;
            }
        }

        if (range.index_count > 0) {
            gpu_mesh->draw_ranges.push_back(range);
        }
    }
    // Opaque first, then alpha (sclera → iris → lid), then additive glows.
    auto sort_key = [&](const GpuDrawRange& range) {
        if (range.material_index >= gpu_mesh->textures.size()) {
            return 0;
        }
        const GpuMaterialTextures& maps = gpu_mesh->textures[range.material_index];
        if (maps.additive) {
            return 300;
        }
        if (maps.has_alpha || maps.eye_layer != 0) {
            const int sub = maps.eye_layer == 0 ? 0 : maps.eye_layer;
            return 100 + sub;
        }
        // Still opaque, but after the head it is painted on.
        if (maps.face_decal) {
            return 50;
        }
        return 0;
    };
    std::stable_sort(
        gpu_mesh->draw_ranges.begin(),
        gpu_mesh->draw_ranges.end(),
        [&](const GpuDrawRange& a, const GpuDrawRange& b) {
            return sort_key(a) < sort_key(b);
        });

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
    glEnableVertexAttribArray(3);
    glVertexAttribPointer(
        3, 3, GL_FLOAT, GL_FALSE, sizeof(GpuVertex), reinterpret_cast<void*>(offsetof(GpuVertex, color)));

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
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glClearColor(0.f, 0.f, 0.f, 0.f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    const cv::Mat view = makeViewMatrix(rvec, tvec);
    // Far plane has to clear the biggest Pokemon comfortably; 100 in cut off
    // anything larger than the card.
    const cv::Mat projection = makeProjectionMatrix(camera_mat, width_, height_, 0.1, 1000.0);
    const cv::Mat model = cv::Mat::eye(4, 4, CV_64F);
    const cv::Mat mvp = projection * view * model;

    // object → card → view  (positions are already in card space; normals still object space)
    cv::Mat view_rot_d = view(cv::Rect(0, 0, 3, 3)).clone();
    cv::Mat view_rot;
    view_rot_d.convertTo(view_rot, CV_32F);
    cv::Mat normal_matrix = view_rot * gpu_mesh->object_to_card;

    glUseProgram(shader_program_);
    glUniformMatrix4fv(glGetUniformLocation(shader_program_, "uMVP"), 1, GL_FALSE, toGlMat4(mvp).ptr<float>());
    glUniformMatrix4fv(glGetUniformLocation(shader_program_, "uView"), 1, GL_FALSE, toGlMat4(view).ptr<float>());
    glUniformMatrix3fv(
        glGetUniformLocation(shader_program_, "uNormalMatrix"),
        1,
        GL_FALSE,
        toGlMat3(normal_matrix).ptr<float>());

    glBindVertexArray(gpu_mesh->vao);
    for (const GpuDrawRange& range : gpu_mesh->draw_ranges) {
        if (!material_filter_.empty() &&
            range.material_index < gpu_mesh->material_names.size() &&
            gpu_mesh->material_names[range.material_index] != material_filter_) {
            continue;
        }

        GpuMaterialTextures maps;
        if (range.material_index < gpu_mesh->textures.size()) {
            maps = gpu_mesh->textures[range.material_index];
        }

        cv::Vec3f diffuse(0.8f, 0.8f, 0.8f);
        float opacity = 1.f;
        if (range.material_index < gpu_mesh->diffuse_colors.size()) {
            diffuse = gpu_mesh->diffuse_colors[range.material_index];
        }
        if (range.material_index < gpu_mesh->diffuse_opacity.size()) {
            opacity = gpu_mesh->diffuse_opacity[range.material_index];
        }
        glUniform3f(
            glGetUniformLocation(shader_program_, "uDiffuse"),
            diffuse[0], diffuse[1], diffuse[2]);
        glUniform1f(glGetUniformLocation(shader_program_, "uOpacity"), opacity);

        glUniform1i(glGetUniformLocation(shader_program_, "uUseDiffuseMap"), maps.diffuse != 0 ? 1 : 0);
        glUniform1i(glGetUniformLocation(shader_program_, "uUseNormalMap"), maps.normal != 0 ? 1 : 0);
        glUniform1i(glGetUniformLocation(shader_program_, "uUseSpecularMap"), maps.specular != 0 ? 1 : 0);
        glUniform1i(glGetUniformLocation(shader_program_, "uUseColor2Map"), maps.color2 != 0 ? 1 : 0);
        glUniform1i(glGetUniformLocation(shader_program_, "uUseAlphaMask"), maps.alpha_mask != 0 ? 1 : 0);
        glUniform1i(glGetUniformLocation(shader_program_, "uAdditive"), maps.additive ? 1 : 0);
        glUniform1i(glGetUniformLocation(shader_program_, "uFlattenLight"), maps.flatten_light ? 1 : 0);
        // Cutouts and untextured translucent shells (Solosis) need texture/factor
        // alpha; film-opacity atlases must not.
        const bool use_tex_alpha =
            maps.has_alpha || maps.alpha_clip || maps.eye_layer != 0 ||
            (!maps.diffuse && opacity < 0.999f);
        glUniform1i(glGetUniformLocation(shader_program_, "uUseTextureAlpha"),
                    use_tex_alpha ? 1 : 0);
        glUniform3f(glGetUniformLocation(shader_program_, "uEffectTint"),
                    maps.tint[0], maps.tint[1], maps.tint[2]);

        // Premultiplied additive glow must not occlude anything behind it.
        // poke-3D eyes: sclera/iris write colour only; lids write depth last so
        // they cover the pupil without a camera-ward mesh bias.
        if (maps.additive) {
            glDisable(GL_POLYGON_OFFSET_FILL);
            glDepthFunc(GL_LESS);
            glBlendFunc(GL_ONE, GL_ONE);
            glDepthMask(GL_FALSE);
        } else if (maps.eye_layer == 1 || maps.eye_layer == 2) {
            glDisable(GL_POLYGON_OFFSET_FILL);
            glDepthFunc(GL_LEQUAL);
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
            glDepthMask(GL_FALSE);
        } else if (maps.eye_layer == 3) {
            glDisable(GL_POLYGON_OFFSET_FILL);
            glDepthFunc(GL_LEQUAL);
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
            glDepthMask(GL_TRUE);
        } else if (maps.alpha_clip) {
            glEnable(GL_POLYGON_OFFSET_FILL);
            glPolygonOffset(2.f, 8.f);
            glDepthFunc(GL_LESS);
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
            glDepthMask(GL_TRUE);
        } else if (maps.face_decal) {
            // Bias toward the camera so the quad wins against the skull it is
            // coplanar with instead of z-fighting or disappearing entirely.
            glEnable(GL_POLYGON_OFFSET_FILL);
            glPolygonOffset(-2.f, -8.f);
            glDepthFunc(GL_LEQUAL);
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
            glDepthMask(GL_TRUE);
        } else {
            glDisable(GL_POLYGON_OFFSET_FILL);
            glDepthFunc(GL_LESS);
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
            glDepthMask(GL_TRUE);
        }

        bindTexture2D(0, maps.diffuse, glGetUniformLocation(shader_program_, "uDiffuseMap"), 0);
        bindTexture2D(1, maps.normal, glGetUniformLocation(shader_program_, "uNormalMap"), 1);
        bindTexture2D(2, maps.specular, glGetUniformLocation(shader_program_, "uSpecularMap"), 2);
        bindTexture2D(3, maps.color2, glGetUniformLocation(shader_program_, "uColor2Map"), 3);
        bindTexture2D(4, maps.alpha_mask, glGetUniformLocation(shader_program_, "uAlphaMask"), 4);

        glDrawElements(
            GL_TRIANGLES,
            range.index_count,
            GL_UNSIGNED_INT,
            reinterpret_cast<void*>(static_cast<std::uintptr_t>(range.first_index) * sizeof(unsigned int)));
    }
    // Depth writes must be back on or the next frame's depth clear is a no-op.
    glDisable(GL_POLYGON_OFFSET_FILL);
    glDepthFunc(GL_LESS);
    glDepthMask(GL_TRUE);
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
    for (GpuMaterialTextures& maps : gpu_mesh->textures) {
        deleteTexture(maps.diffuse);
        deleteTexture(maps.normal);
        deleteTexture(maps.specular);
        deleteTexture(maps.color2);
        deleteTexture(maps.alpha_mask);
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
