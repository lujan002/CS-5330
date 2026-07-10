// Luke Jansen
// 7/1/2026
// main.cpp


#include <opencv2/opencv.hpp>
#include <opencv2/calib3d.hpp>
#include <fstream>

#include "obj_loader.hpp"
#include "gl_renderer.hpp"


int main(int argc, char *argv[]) {
    cv::VideoCapture *capdev;

    // default to /dev/video0 for the capture device
    const char *device = (argc > 1) ? argv[1] : "/dev/video0";
    capdev = new cv::VideoCapture(device, cv::CAP_V4L2);

    // error handling
    if( !capdev->isOpened() ) {
        printf("Unable to open video device\n");
        return(-1);
    }
    cv::namedWindow("Video", 1);
    cv::namedWindow("FrameAxesImg", 1);

    cv::Mat frame;
    std::vector<cv::Mat> calib_frames;

    cv::Size patternsize(9, 6); // number of inner corners per row and column
    std::vector<cv::Point2f> corner_set;
    std::vector<std::vector<cv::Point2f>> corner_list;

    // 3D positions of corners in world coordinates (9x6 chessboard, top-left origin)
    const float square_size = 1.0f;
    std::vector<cv::Vec3f> point_set;
    for (int i = 0; i < patternsize.height; i++) {
        for (int j = 0; j < patternsize.width; j++) {
            point_set.push_back(cv::Vec3f(j * square_size, i * square_size, 0.0f));
        }
    }
    std::vector<std::vector<cv::Vec3f>> point_list;

    cv::Size img_size = cv::Size(640, 480);

    // initialize camera intrinsic matrix
    cv::Mat camera_mat = cv::Mat::eye(3, 3, CV_64FC1);
    camera_mat.at<double>(0, 2) = img_size.width / 2.0;
    camera_mat.at<double>(1, 2) = img_size.height / 2.0;

    bool pose_mode = false;
    bool model_mode = false;
    
    cv::Mat dist_coeffs;
    cv::Mat rvec;
    cv::Mat tvec;
    bool intrinsics_loaded = false;
    bool use_nidoking_model = false;
    
    for (;;) {
        *capdev >> frame;
        if( frame.empty()) {
            printf("frame is empty\n");
            break;
        }

        // resize frame
        cv::resize(frame, frame, img_size, 0, 0, cv::INTER_AREA);

        // Convert to greyscale
        cv::Mat grey;
        cv::cvtColor(frame, grey, cv::COLOR_BGR2GRAY);
        
        // Find rough corner locations
        bool patternfound = findChessboardCorners(grey, patternsize, corner_set, 
            cv::CALIB_CB_ADAPTIVE_THRESH + cv::CALIB_CB_NORMALIZE_IMAGE
            + cv::CALIB_CB_FAST_CHECK);

        if (patternfound) {
            // Refines the corner locations
            cornerSubPix(grey, corner_set, cv::Size(11, 11), cv::Size(-1, -1),
                cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 30, 0.1));
        }

        // draw corner circles on frame 
        drawChessboardCorners(frame, patternsize, cv::Mat(corner_set), patternfound);

        char key = cv::waitKey(10);
        if (key == 'q') {
            break;
        }
        if (key == 's') {
            // save the last corner locations (2D image points + matching 3D world points)
            if (patternfound && corner_set.size() == patternsize.width * patternsize.height) {
                corner_list.push_back(corner_set);
                point_list.push_back(point_set);
                calib_frames.push_back(frame.clone());
                printf("Saved %zu corner/point sets\n", corner_list.size());
            } else {
                printf("Chessboard not detected — not saved\n");
            }
        }
        if (key == 'c') {
            // Calibrate
            if (corner_list.size() >= 5) {
                std::vector<cv::Mat> rvecs;
                std::vector<cv::Mat> tvecs;
                std::vector<double> std_intrinsics;
                std::vector<double> std_extrinsics;
                std::vector<double> view_errors;
                int flags = cv::CALIB_FIX_ASPECT_RATIO; // | cv::CALIB_USE_INTRINSIC_GUESS;
                
                double rms = cv::calibrateCamera(point_list, corner_list, img_size, 
                    camera_mat, dist_coeffs, rvecs, tvecs, 
                    std_intrinsics, std_extrinsics, view_errors, 
                    flags, cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, DBL_EPSILON)
                );

                std::cout << "Camera matrix after calibration:\n" << camera_mat << std::endl;

                printf("Total re-projection error: %f px\n", rms);
                std::cout << "Distortion coefficients:\n" << dist_coeffs << std::endl;

                // Error should be <1-3px 

                // Optionally save camera frames and use rvec, tvec to visualize 
                // camera locations relative to target

                // write camera matrix and distortion coefficients to file
                cv::FileStorage fs("camera_intrinsics.yaml", cv::FileStorage::WRITE);
                fs << "camera_mat" << camera_mat;
                fs << "dist_coeffs" << dist_coeffs;
                fs.release();

                // Draw frame axes on the target
                float length = 1.5; // same units as tvec
                int thickness = 3;
                cv::Mat calib_image = calib_frames[0];
                rvec = rvecs[0];
                tvec = tvecs[0];
                cv::drawFrameAxes(calib_image, camera_mat, dist_coeffs, rvec, tvec, length, thickness);  
                cv::imshow("FrameAxesImg", calib_image);
            }
            else {
                printf("At least 5 corner/point sets are required for calibration\n");
            }
        }
        if (key == 'p') {
            cv::FileStorage fs("camera_intrinsics.yaml", cv::FileStorage::READ);
            if (!fs.isOpened()) {
                printf("Could not open camera_intrinsics.yaml — calibrate first with 'c'\n");
                continue;
            } else {
                fs["camera_mat"] >> camera_mat;
                fs["dist_coeffs"] >> dist_coeffs;
                fs.release();
                intrinsics_loaded = true;
                pose_mode = true;
            }

        }
        if (pose_mode && patternfound && intrinsics_loaded) {
            // Calculate the current pose of the camera relative to the target
            // read camera matrix and distortion coefficients from file
            bool useExtrinsicGuess = false;
            int flags = cv::SOLVEPNP_ITERATIVE;

            // Start video loop (We already have this)
            // For each frame, detect the chessboard corners (corner_set)
            // And use solvePnP to get the board's pose in camera frame (rvec, tvec)
            bool solved = cv::solvePnP(point_set, corner_set, 
                camera_mat, dist_coeffs, rvec, tvec, useExtrinsicGuess, flags);
            
            if (solved == true) {
                // invert rvec, tvec to get camera pose in board/target frame (target at origin)
                cv::Mat R;
                cv::Rodrigues(rvec, R);
                cv::Mat t_cam = -R.t() * tvec;  // camera position in board frame (CV_64F)
                cv::Mat R_cam = R.t();          // camera orientation in board frame

                // print rotation and translation on one line (Mat << wraps long rows)
                std::cout << "\r t_cam: " << t_cam.t() << "  R_cam: [";
                const cv::Mat R_flat = R_cam.reshape(1, 1);
                for (int i = 0; i < R_flat.cols; ++i) {
                    if (i > 0) std::cout << ", ";
                    std::cout << R_flat.at<double>(0, i);
                }
                std::cout << "]" << std::flush;
            }
            // Project outside corners (or 3D axies) on the image plane in real time
            std::vector<cv::Point2f> proj_corner_set;
            cv::projectPoints(point_set, rvec, tvec, camera_mat, dist_coeffs, proj_corner_set);

            // display projected points on frame
            for (const auto& pt : proj_corner_set) {
                cv::circle(frame, pt, 5, cv::Scalar(0, 0, 255), -1);
            }
        }
        if (key == 'm') {
            cv::FileStorage fs("camera_intrinsics.yaml", cv::FileStorage::READ);
            if (!fs.isOpened()) {
                printf("Could not open camera_intrinsics.yaml — calibrate first with 'c'\n");
                continue;
            } else {
                fs["camera_mat"] >> camera_mat;
                fs["dist_coeffs"] >> dist_coeffs;
                fs.release();
                intrinsics_loaded = true;
                model_mode = true;
            }
        }
        if (model_mode && patternfound && intrinsics_loaded && !use_nidoking_model) {
            // display simple pyramid model
            std::vector<cv::Vec3f> model_point_set = {
                cv::Vec3f(0, 0, -1),
                cv::Vec3f(2, 0, -1),
                cv::Vec3f(2, 2, -1),
                cv::Vec3f(0, 2, -1),
                cv::Vec3f(1, 1, -3)
            };
            int thickness = 3;

            // Project points to 2D image plane
            bool useExtrinsicGuess = false;
            int flags = cv::SOLVEPNP_ITERATIVE;
            bool solved = cv::solvePnP(point_set, corner_set, 
                camera_mat, dist_coeffs, rvec, tvec, useExtrinsicGuess, flags);
            
            if (solved) {
                std::vector<cv::Point2f> proj_model_corner_set;
                cv::projectPoints(model_point_set, rvec, tvec, camera_mat, dist_coeffs, proj_model_corner_set);
                // display projected points on frame
                for (size_t i = 0; i < proj_model_corner_set.size(); i++) {
                    cv::circle(frame, proj_model_corner_set[i], 5, cv::Scalar(0, 0, 255), -1);
                    // Draw lines between each point in image plane (once per edge)
                    for (size_t j = i + 1; j < proj_model_corner_set.size(); j++) {
                        cv::Scalar color(
                            (37 * static_cast<int>(i) + 17 * static_cast<int>(j)) % 256,
                            (59 * static_cast<int>(i) + 23 * static_cast<int>(j)) % 256,
                            (97 * static_cast<int>(i) + 31 * static_cast<int>(j)) % 256
                        );
                        cv::line(frame, proj_model_corner_set[i], proj_model_corner_set[j], color, thickness);
                    }
                }
            }

        }
        if (model_mode && patternfound && intrinsics_loaded && use_nidoking_model) {
            static bool mesh_ready = false;
            static bool gl_ready = false;
            static ObjMesh loaded_mesh;
            static std::vector<cv::Point3f> model_vertices;
            static std::vector<std::pair<int, int>> model_edges;
            static GLRenderer gl_renderer;
            if (!mesh_ready) {
                if (loadObjMesh("../models/nidoking/nidoking.obj", loaded_mesh)) {
                    cv::Point3f center(
                        0.5f * (loaded_mesh.min_bounds.x + loaded_mesh.max_bounds.x),
                        loaded_mesh.min_bounds.y,
                        0.5f * (loaded_mesh.min_bounds.z + loaded_mesh.max_bounds.z));
                    float scale = 2.5f / (loaded_mesh.max_bounds.y - loaded_mesh.min_bounds.y);
                    cv::Mat rot = (cv::Mat_<float>(3, 3) << -1.f, 0.f, 0.f, 0.f, 0.f, -1.f, 0.f, 1.f, 0.f);
                    model_vertices = transformMeshVertices(loaded_mesh, center, scale, rot);
                    cv::Point3f board_center(
                        0.5f * (patternsize.width - 1) * square_size,
                        -0.5f * (patternsize.height - 1) * square_size,
                        0.f);
                    for (cv::Point3f& v : model_vertices) {
                        v.x += board_center.x;
                        v.y += board_center.y;
                        v.z += board_center.z;
                    }
                    model_edges = buildUniqueEdges(loaded_mesh);
                    if (gl_renderer.init(img_size.width, img_size.height)) {
                        gl_ready = gl_renderer.uploadMesh(loaded_mesh, model_vertices);
                    }
                    mesh_ready = true;
                }
            }
            if (mesh_ready) {
                bool solved = cv::solvePnP(point_set, corner_set,
                    camera_mat, dist_coeffs, rvec, tvec, false, cv::SOLVEPNP_ITERATIVE);
                if (solved) {
                    if (gl_ready) {
                        gl_renderer.renderOverlay(camera_mat, rvec, tvec, frame);
                    } else {
                        // wireframe fallback if GL init fails
                        std::vector<cv::Point2f> proj;
                        cv::projectPoints(model_vertices, rvec, tvec, camera_mat, dist_coeffs, proj);
                        for (const auto& edge : model_edges) {
                            cv::line(frame, proj[edge.first], proj[edge.second],
                                cv::Scalar(0, 255, 255), 1);
                        }
                    }
                }
            }
        }
        cv::imshow("Video", frame);
    }
}

