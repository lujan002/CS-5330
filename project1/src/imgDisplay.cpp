// Luke Jansen
// 5/12/2026
// imgDisplay.cpp
// Reads an image file and displays it in a window.

#include <iostream>
#include <opencv2/opencv.hpp>

int main() {
    std::string image_path = "../assets/strawberry.jpeg";
    cv::Mat image = cv::imread(image_path);
    if (image.empty()) {
        std::cerr << "Could not read the image: " << image_path << std::endl;
        return 1;
    }

    cv::imshow("Image", image);

    // if the user presses the 'q' key, the program will exit
    while (true) {
        if (cv::waitKey(1) == 'q') {
            break;
        }
    }

    return 0;
}
