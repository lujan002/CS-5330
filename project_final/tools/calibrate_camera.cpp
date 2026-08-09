// Standalone camera calibration — same flow as project4.
// Writes camera_intrinsics.yaml for ar_card (run from project_final/build/).
//
// Usage: ./calibrate_camera [/dev/videoN]
//   s — save a chessboard view (need >= 5)
//   c — run calibrateCamera and write camera_intrinsics.yaml
//   p — load yaml and show live pose / projected corners
//   q — quit

#include <opencv2/opencv.hpp>
#include <opencv2/calib3d.hpp>

#include <cstdio>
#include <iostream>
#include <vector>

int main(int argc, char* argv[]) {
    const char* device = (argc > 1) ? argv[1] : "/dev/video0";
    cv::VideoCapture cap(device, cv::CAP_V4L2);

    if (!cap.isOpened()) {
        printf("Unable to open video device %s\n", device);
        return -1;
    }

    cv::namedWindow("Video", 1);
    cv::namedWindow("FrameAxesImg", 1);

    cv::Mat frame;
    std::vector<cv::Mat> calib_frames;

    cv::Size patternsize(9, 6);  // inner corners per row and column
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

    cv::Mat dist_coeffs;
    cv::Mat rvec;
    cv::Mat tvec;
    bool pose_mode = false;
    bool intrinsics_loaded = false;

    printf("Controls: s=save view  c=calibrate  p=pose check  q=quit\n");
    printf("Need at least 5 saved views before 'c'. Output: camera_intrinsics.yaml\n");

    for (;;) {
        cap >> frame;
        if (frame.empty()) {
            printf("frame is empty\n");
            break;
        }

        cv::resize(frame, frame, img_size, 0, 0, cv::INTER_AREA);

        cv::Mat grey;
        cv::cvtColor(frame, grey, cv::COLOR_BGR2GRAY);

        bool patternfound = findChessboardCorners(
            grey, patternsize, corner_set,
            cv::CALIB_CB_ADAPTIVE_THRESH + cv::CALIB_CB_NORMALIZE_IMAGE
                + cv::CALIB_CB_FAST_CHECK);

        if (patternfound) {
            cornerSubPix(grey, corner_set, cv::Size(11, 11), cv::Size(-1, -1),
                         cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER,
                                          30, 0.1));
        }

        drawChessboardCorners(frame, patternsize, cv::Mat(corner_set), patternfound);

        char key = static_cast<char>(cv::waitKey(10));
        if (key == 'q') {
            break;
        }
        if (key == 's') {
            if (patternfound &&
                corner_set.size() ==
                    static_cast<size_t>(patternsize.width * patternsize.height)) {
                corner_list.push_back(corner_set);
                point_list.push_back(point_set);
                calib_frames.push_back(frame.clone());
                printf("Saved %zu corner/point sets\n", corner_list.size());
            } else {
                printf("Chessboard not detected — not saved\n");
            }
        }
        if (key == 'c') {
            if (corner_list.size() >= 5) {
                std::vector<cv::Mat> rvecs;
                std::vector<cv::Mat> tvecs;
                std::vector<double> std_intrinsics;
                std::vector<double> std_extrinsics;
                std::vector<double> view_errors;
                int flags = cv::CALIB_FIX_ASPECT_RATIO;

                double rms = cv::calibrateCamera(
                    point_list, corner_list, img_size, camera_mat, dist_coeffs, rvecs,
                    tvecs, std_intrinsics, std_extrinsics, view_errors, flags,
                    cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30,
                                     DBL_EPSILON));

                std::cout << "Camera matrix after calibration:\n"
                          << camera_mat << std::endl;
                printf("Total re-projection error: %f px\n", rms);
                std::cout << "Distortion coefficients:\n" << dist_coeffs << std::endl;

                cv::FileStorage fs("camera_intrinsics.yaml", cv::FileStorage::WRITE);
                fs << "camera_mat" << camera_mat;
                fs << "dist_coeffs" << dist_coeffs;
                fs.release();
                printf("Wrote camera_intrinsics.yaml\n");
                intrinsics_loaded = true;

                float length = 1.5f;
                int thickness = 3;
                cv::Mat calib_image = calib_frames[0].clone();
                rvec = rvecs[0];
                tvec = tvecs[0];
                cv::drawFrameAxes(calib_image, camera_mat, dist_coeffs, rvec, tvec,
                                  length, thickness);
                cv::imshow("FrameAxesImg", calib_image);
            } else {
                printf("At least 5 corner/point sets are required for calibration\n");
            }
        }
        if (key == 'p') {
            cv::FileStorage fs("camera_intrinsics.yaml", cv::FileStorage::READ);
            if (!fs.isOpened()) {
                printf("Could not open camera_intrinsics.yaml — calibrate first with 'c'\n");
            } else {
                fs["camera_mat"] >> camera_mat;
                fs["dist_coeffs"] >> dist_coeffs;
                fs.release();
                intrinsics_loaded = true;
                pose_mode = true;
                printf("Pose mode on — showing projected corners / camera pose\n");
            }
        }
        if (pose_mode && patternfound && intrinsics_loaded) {
            bool solved = cv::solvePnP(point_set, corner_set, camera_mat, dist_coeffs,
                                       rvec, tvec, false, cv::SOLVEPNP_ITERATIVE);

            if (solved) {
                cv::Mat R;
                cv::Rodrigues(rvec, R);
                cv::Mat t_cam = -R.t() * tvec;
                cv::Mat R_cam = R.t();

                std::cout << "\r t_cam: " << t_cam.t() << "  R_cam: [";
                const cv::Mat R_flat = R_cam.reshape(1, 1);
                for (int i = 0; i < R_flat.cols; ++i) {
                    if (i > 0) {
                        std::cout << ", ";
                    }
                    std::cout << R_flat.at<double>(0, i);
                }
                std::cout << "]" << std::flush;
            }

            std::vector<cv::Point2f> proj_corner_set;
            cv::projectPoints(point_set, rvec, tvec, camera_mat, dist_coeffs,
                              proj_corner_set);
            for (const auto& pt : proj_corner_set) {
                cv::circle(frame, pt, 5, cv::Scalar(0, 0, 255), -1);
            }
        }

        cv::imshow("Video", frame);
    }

    return 0;
}
