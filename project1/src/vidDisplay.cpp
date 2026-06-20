// Luke Jansen
// 5/12/2026
// vidDisplay.cpp
// This file is used to display a video from a camera in a window.

#include <iostream>
#include <opencv2/opencv.hpp>
#include <filters.h>
#include <faceDetect/faceDetect.h>
#include <da2-code/DA2Network.hpp>

int main(int argc, char *argv[]) {
    cv::VideoCapture *capdev;

    // open the video device (path is explicit; CAP_V4L2 avoids GStreamer probing)
    // capdev = new cv::VideoCapture("/dev/video100", cv::CAP_V4L2);

    // V4L2 capture device: default first real camera (/dev/video0). Do not use
    // /dev/video100 here unless you mean v4l2loopback—that is a dummy sink, not your webcam.
    const char *device = (argc > 1) ? argv[1] : "/dev/video0";
    capdev = new cv::VideoCapture(device, cv::CAP_V4L2);
    
    if( !capdev->isOpened() ) {
            printf("Unable to open video device\n");
            return(-1);
    }

    // get some properties of the image
    cv::Size refS( (int) capdev->get(cv::CAP_PROP_FRAME_WIDTH ),
                   (int) capdev->get(cv::CAP_PROP_FRAME_HEIGHT));
    printf("Expected size: %d %d\n", refS.width, refS.height);

    cv::namedWindow("Video", 1); // identifies a window
    // cv::namedWindow("Depth", 2); // identifies a window
    cv::Mat frame;

    int frame_count = 0;
    std::string mode = "normal";

    // Setup Depth Anything Network
    const float reduction = 0.5;
    DA2Network da_net( "../src/da2-code/model_fp16.onnx" );

    float scale_factor = 256.0 / (refS.height*reduction);
    printf("Using scale factor %.2f\n", scale_factor);

    for(;;) {
            *capdev >> frame; // get a new frame from the camera, treat as a stream
            if( frame.empty() ) {
              printf("frame is empty\n");
              break;
            }
            // resize the frame to 640x480
            cv::resize(frame, frame, cv::Size(640, 480), 0, 0, cv::INTER_AREA);

            if (mode == "greyscale") {
                cv::Mat grey;
                greyscale_filter(frame, grey);
                frame = grey;
            }
            else if (mode == "alt_greyscale") {
                cv::Mat alt_gray;
                alt_greyscale_filter(frame, alt_gray);
                frame = alt_gray;
            }
            else if (mode == "sepia") {
                cv::Mat sepia;
                sepia_filter(frame, sepia);
                frame = sepia;
            }
            else if (mode == "blur") {
                cv::Mat blur;
                blur5x5_2(frame, blur);
                frame = blur;
            }
            else if (mode == "sobelX") {
                cv::Mat sobelX;
                sobelX3x3(frame, sobelX);
                convertToUnsigned(sobelX, sobelX);
                frame = sobelX;
            }
            else if (mode == "sobelY") {
                cv::Mat sobelY;
                sobelY3x3(frame, sobelY);
                convertToUnsigned(sobelY, sobelY);
                frame = sobelY;
            }
            else if (mode == "sobelMagnitude") {
                cv::Mat sobelX;
                cv::Mat sobelY;
                cv::Mat mag;
                sobelX3x3(frame, sobelX);
                sobelY3x3(frame, sobelY);
                magnitude(sobelX, sobelY, mag);
                frame = mag;
            }
            else if (mode == "blurQuantize") {
                cv::Mat blur;
                blur5x5_2(frame, blur);                
                cv::Mat quantized;
                blurQuantize(blur, quantized, 10);
                frame = quantized;
            }
            else if (mode == "faceDetect") {
                cv::Mat grey;
                greyscale_filter(frame, grey);
                std::vector<cv::Rect> faces;
                cv::Rect last(0, 0, 0, 0);
                detectFaces(grey, faces);
                // minWidth 20: default 50 can hide smaller/distant faces at 640x480
                drawBoxes(frame, faces, 20);
                // add a little smoothing by averaging the last two detections
                if( faces.size() > 0 ) {
                    last.x = (faces[0].x + last.x)/2;
                    last.y = (faces[0].y + last.y)/2;
                    last.width = (faces[0].width + last.width)/2;
                    last.height = (faces[0].height + last.height)/2;
                }
            }
            
            else if (mode == "depth") {
                cv::Mat depth;
                cv::Mat depth_vis;
                // for speed purposes, reduce the size of the input frame by half
                cv::resize( frame, frame, cv::Size(), reduction, reduction );

                // set the network input
                da_net.set_input( frame, scale_factor );

                // run the network
                da_net.run_network( depth, frame.size() );

                // apply a color map to the depth output to get a good visualization
                // cv::applyColorMap(depth, depth_vis, cv::COLORMAP_INFERNO );

                depthInvert(frame, depth, depth_vis);
                frame = depth_vis;
            }   
            else if (mode == "halo") {
                // for speed purposes, reduce the size of the input frame by half
                cv::resize( frame, frame, cv::Size(), reduction, reduction );

                cv::Mat grey;
                greyscale_filter(frame, grey);
                std::vector<cv::Rect> faces;
                detectFaces(grey, faces);

                cv::Mat depth;
                // set the network input
                da_net.set_input( frame, scale_factor );
                // run the network
                da_net.run_network( depth, frame.size() );

                cv::Mat out;
                halo(frame, faces, depth, out);
                frame = out;
            }
            else if (mode == "quarter") {
                // Half-res for face detection speed. Composite: normal frame, then embossed
                // quarter disk (always on top of video), then embossed face rectangle.
                cv::resize( frame, frame, cv::Size(), reduction, reduction );

                cv::Mat grey;
                greyscale_filter(frame, grey);
                std::vector<cv::Rect> faces;
                detectFaces(grey, faces);

                cv::Mat out;
                faceEmbossQuarter(frame, faces, out);
                frame = out;
            }
            else if (mode == "blue") {
                cv::Mat out;
                blue(frame, out);
                frame = out;
            }
            else if (mode == "emboss") {
                cv::Mat out;
                cv::Mat grey;
                greyscale_filter(frame, grey);
                emboss(grey, out);
                frame = out;
            }

            cv::imshow("Video", frame); // Write frame

            // see if there is a waiting keystroke
            char key = cv::waitKey(10);
            if( key == 'q') {
                break;
            }

            // capture and save the current frame from the video stream when 's' is pressed
            if (key == 's') {
                std::string filename = "../frames/frame_" + std::to_string(frame_count) + ".jpg";
                if (cv::imwrite(filename, frame)) {
                    std::cout << "Captured and saved " << filename << std::endl;
                    frame_count++;
                } else {
                    std::cerr << "Failed to save " << filename << std::endl;
                }
            }
    
            // change to greyscale 
            if (key == 'g') {
                if (mode == "normal") {
                    mode = "greyscale";
                } else {
                    mode = "normal";
                }
            }

            // change to alternate greyscale
            if (key == 'h') {
                if (mode == "normal") {
                    mode = "alt_greyscale";
                } else {
                    mode = "normal";
                }
            }

            if (key == 'p') {
                if (mode == "normal") {
                    mode = "sepia";
                } else {
                    mode = "normal";
                }
            }

            if (key == 'b') {
                if (mode == "normal") {
                    mode = "blur";
                    std::cout << "Blur mode enabled" << std::endl;
                } else {
                    mode = "normal";
                }
            }

            if (key == 'x') {
                if (mode == "normal") {
                    mode = "sobelX";
                } else {
                    mode = "normal";
                }
            }

            if (key == 'y') {
                if (mode == "normal") {
                    mode = "sobelY";
                } else {
                    mode = "normal";
                }
            }

            if (key == 'm') {
                if (mode == "normal") {
                    mode = "sobelMagnitude";
                } else {
                    mode = "normal";
                }
            }

            if (key == 'z') {
                if (mode == "normal") {
                    mode = "blurQuantize";
                } else {
                    mode = "normal";
                }
            }

            if (key == 'f') {
                if (mode == "normal") {
                    mode = "faceDetect";
                } else {
                    mode = "normal";
                }
            }

            if (key == 'd') {
                if (mode == "normal") {
                    mode = "depth";
                } else {
                    mode = "normal";
                }
            }

            if (key == 'o') {
                if (mode == "normal") {
                    mode = "halo";
                } else {
                    mode = "normal";
                }
            }

            if (key == 'k') {
                if (mode == "normal") {
                    mode = "quarter";
                } else {
                    mode = "normal";
                }
            }

            if (key == 'u') {
                if (mode == "normal") {
                    mode = "blue";
                } else {
                    mode = "normal";
                }
            }
            if (key == 'e') {
                if (mode == "normal") {
                    mode = "emboss";
                } else {
                    mode = "normal";
                }
            }
    }

    delete capdev;
    return(0);
}
