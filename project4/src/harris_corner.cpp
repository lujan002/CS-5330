// Luke Jansen
// 7/1/2026
// harris_corner.cpp

#include <opencv2/opencv.hpp>


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

    cv::Mat frame;
    cv::Size img_size = cv::Size(640, 480);

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
        
        // blur 
        // cv::Mat blurred;
        // cv::GaussianBlur(grey, blurred, cv::Size(5, 5), 0);
    
        cv::Mat dst;
        int blockSize = 3; // for 3x3 kernel
        int aperatureSize = 19;
        double k = 0.04;
        cv::cornerHarris(grey, dst, blockSize, aperatureSize, k);
        
        cv::Mat dilated;
        cv::dilate(dst, dilated, cv::Mat());  // default 3x3 rect kernel

        double maxVal = 0;
        cv::minMaxLoc(dilated, nullptr, &maxVal);

        for (int i = 0; i < dilated.rows; i++) {
            for (int j = 0; j < dilated.cols; j++) {
                if (dilated.at<float>(i, j) > 0.1 * maxVal) {
                    // Mark this point as a red dot
                    frame.at<cv::Vec3b>(i, j) = cv::Vec3b(0, 0, 255);
                }
            }
        }

        char key = cv::waitKey(10);
        if (key == 'q') {
            break;
        }
        
        cv::imshow("Video", frame);
    }
}