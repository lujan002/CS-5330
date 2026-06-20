// Luke Jansen
// 5/12/2026
// filter.cpp
// Apply filters to an image.

#include <opencv2/opencv.hpp>
#include <algorithm>
#include <vector>

int greyscale_filter(const cv::Mat &src, cv::Mat &dst) {
    // Input: source image
    // Output: greyscale image

    cv::cvtColor(src, dst, cv::COLOR_BGR2GRAY);  // Convert to grayscale
    return 0;
}

int alt_greyscale_filter(const cv::Mat &src, cv::Mat &dst) {
    // Input: source image
    // Output: alternate greyscale image

    // CV_Assert(src.channels() == 3 && src.depth() == CV_8U);
    std::vector<cv::Mat> bgr; // create a vector to store the channels
    cv::split(src, bgr); // split interleaved soruce image into planar image (separate channels)
    cv::Mat v = 255 - (bgr[1] + bgr[0])/2; // invert the blue and green channel average
    cv::merge(std::vector<cv::Mat>{v, v, v}, dst); // rebuild interleaved (multi-channel) image
    return 0;
}

int sepia_filter(const cv::Mat &src, cv::Mat &dst) {
    // Input: source image
    // Output: sepia image

    std::vector<cv::Mat> bgr;
    cv::split(src, bgr);
    // using old color values saved in bgr, apply sepia matrix
    cv::Mat blue = 0.272 * bgr[0] + 0.534 * bgr[1] + 0.131 * bgr[2];
    cv::Mat green = 0.349 * bgr[1] + 0.686 * bgr[2] + 0.168 * bgr[0];
    cv::Mat red = 0.393 * bgr[1] + 0.769 * bgr[2] + 0.189 * bgr[0];

    // Ensure that no value is greater than 255
    cv::min(blue, 255, blue);
    cv::min(green, 255, green);
    cv::min(red, 255, red);
    cv::merge(std::vector<cv::Mat>{blue, green, red}, dst); // rebuild interleaved (multi-channel) image
    return 0;

    // A nice extension is to add vignetting (the image getting darker towards the edges) to this filter.
}

int blur5x5_1(const cv::Mat &src, cv::Mat &dst) {
    // Input: source image
    // Output: blurred image

    std::vector<cv::Mat> bgr;
    cv::split(src, bgr);

    // Create the 5x5 Gaussian kernel as a 2D array (int)
    const int gaussian_kernel[5][5] = {
        {1, 2, 4, 2, 1},
        {2, 4, 8, 4, 2},
        {4, 8, 16, 8, 4},
        {2, 4, 8, 4, 2},
        {1, 2, 4, 2, 1}
    };

    // Instead of cloning the whole image, create black (empty) images and just copy the first two rows and columns
    cv::Mat blue_blur = cv::Mat::zeros(bgr[0].size(), bgr[0].type());
    cv::Mat green_blur = cv::Mat::zeros(bgr[1].size(), bgr[1].type());
    cv::Mat red_blur = cv::Mat::zeros(bgr[2].size(), bgr[2].type());

    // Copy first two rows
    bgr[0].rowRange(0, 2).copyTo(blue_blur.rowRange(0, 2));
    bgr[1].rowRange(0, 2).copyTo(green_blur.rowRange(0, 2));
    bgr[2].rowRange(0, 2).copyTo(red_blur.rowRange(0, 2));
    // Copy last two rows
    bgr[0].rowRange(bgr[0].rows - 2, bgr[0].rows).copyTo(blue_blur.rowRange(bgr[0].rows - 2, bgr[0].rows));
    bgr[1].rowRange(bgr[1].rows - 2, bgr[1].rows).copyTo(green_blur.rowRange(bgr[1].rows - 2, bgr[1].rows));
    bgr[2].rowRange(bgr[2].rows - 2, bgr[2].rows).copyTo(red_blur.rowRange(bgr[2].rows - 2, bgr[2].rows));
    // Copy first two columns
    bgr[0].colRange(0, 2).copyTo(blue_blur.colRange(0, 2));
    bgr[1].colRange(0, 2).copyTo(green_blur.colRange(0, 2));
    bgr[2].colRange(0, 2).copyTo(red_blur.colRange(0, 2));
    // Copy last two columns
    bgr[0].colRange(bgr[0].cols - 2, bgr[0].cols).copyTo(blue_blur.colRange(bgr[0].cols - 2, bgr[0].cols));
    bgr[1].colRange(bgr[1].cols - 2, bgr[1].cols).copyTo(green_blur.colRange(bgr[1].cols - 2, bgr[1].cols));
    bgr[2].colRange(bgr[2].cols - 2, bgr[2].cols).copyTo(red_blur.colRange(bgr[2].cols - 2, bgr[2].cols));

    for (int i = 2; i < bgr[0].rows - 2; i++) {
        for (int j = 2; j < bgr[0].cols - 2; j++) {
            // int accumulators avoid uchar overflow before normalization
            int blue_sum = 0;
            int green_sum = 0;
            int red_sum = 0;
            for (int k = 0; k < 5; k++) {
                for (int l = 0; l < 5; l++) {
                    // apply the gaussian kernel to the 5x5 window centered at (i, j)
                    // “treat bgr[0] as a matrix of bytes and give me the byte at (row, col).”
                    blue_sum += (bgr[0].at<uchar>(i+k-2, j+l-2) * gaussian_kernel[k][l]);  
                    green_sum += (bgr[1].at<uchar>(i+k-2, j+l-2) * gaussian_kernel[k][l]);
                    red_sum += (bgr[2].at<uchar>(i+k-2, j+l-2) * gaussian_kernel[k][l]);
                } 
            }
            // normalize to ensure that the value is not greater than 255
            blue_blur.at<uchar>(i, j) = blue_sum/100;
            green_blur.at<uchar>(i, j) = green_sum/100;
            red_blur.at<uchar>(i, j) = red_sum/100;
        }
    }
    cv::merge(std::vector<cv::Mat>{blue_blur, green_blur, red_blur}, dst);
    return 0;
}

int blur5x5_2(const cv::Mat &src, cv::Mat &dst) {
    // Input: source image
    // Output: blurred image

    std::vector<cv::Mat> bgr;
    cv::split(src, bgr);

    // Create the 5x5 Gaussian kernel as a 1D array (int)
    const int gaussian_kernel[5] = {1, 2, 4, 2, 1};

    // Instead of cloning the whole image, create black (empty) images and just copy the first two rows and columns
    cv::Mat blue_blur = cv::Mat::zeros(bgr[0].size(), bgr[0].type());
    cv::Mat green_blur = cv::Mat::zeros(bgr[1].size(), bgr[1].type());
    cv::Mat red_blur = cv::Mat::zeros(bgr[2].size(), bgr[2].type());

    // Copy first two rows
    bgr[0].rowRange(0, 2).copyTo(blue_blur.rowRange(0, 2));
    bgr[1].rowRange(0, 2).copyTo(green_blur.rowRange(0, 2));
    bgr[2].rowRange(0, 2).copyTo(red_blur.rowRange(0, 2));
    // Copy last two rows
    bgr[0].rowRange(bgr[0].rows - 2, bgr[0].rows).copyTo(blue_blur.rowRange(bgr[0].rows - 2, bgr[0].rows));
    bgr[1].rowRange(bgr[1].rows - 2, bgr[1].rows).copyTo(green_blur.rowRange(bgr[1].rows - 2, bgr[1].rows));
    bgr[2].rowRange(bgr[2].rows - 2, bgr[2].rows).copyTo(red_blur.rowRange(bgr[2].rows - 2, bgr[2].rows));
    // Copy first two columns
    bgr[0].colRange(0, 2).copyTo(blue_blur.colRange(0, 2));
    bgr[1].colRange(0, 2).copyTo(green_blur.colRange(0, 2));
    bgr[2].colRange(0, 2).copyTo(red_blur.colRange(0, 2));
    // Copy last two columns
    bgr[0].colRange(bgr[0].cols - 2, bgr[0].cols).copyTo(blue_blur.colRange(bgr[0].cols - 2, bgr[0].cols));
    bgr[1].colRange(bgr[1].cols - 2, bgr[1].cols).copyTo(green_blur.colRange(bgr[1].cols - 2, bgr[1].cols));
    bgr[2].colRange(bgr[2].cols - 2, bgr[2].cols).copyTo(red_blur.colRange(bgr[2].cols - 2, bgr[2].cols));

    for (int i = 2; i < bgr[0].rows - 2; i++) {
        for (int j = 2; j < bgr[0].cols - 2; j++) {
            // int accumulators avoid uchar overflow before normalization
            int blue_sum = 0;
            int green_sum = 0;
            int red_sum = 0;
            // create pointers to the current row (of center pixel)
            const uchar *row_b_i = bgr[0].ptr<uchar>(i);
            const uchar *row_g_i = bgr[1].ptr<uchar>(i);
            const uchar *row_r_i = bgr[2].ptr<uchar>(i);
            for (int k = 0; k < 5; k++) {
                for (int l = 0; l < 5; l++) {
                    // create pointers to the current row (of window pixel)
                    const uchar *row_b = bgr[0].ptr<uchar>(i + k - 2);
                    const uchar *row_g = bgr[1].ptr<uchar>(i + k - 2);
                    const uchar *row_r = bgr[2].ptr<uchar>(i + k - 2);
                    // horizontal [1,2,4,2,1] sum on row i+k-2
                    const int ll = j + l - 2;
                    blue_sum += row_b[ll] * gaussian_kernel[l];
                    green_sum += row_g[ll] * gaussian_kernel[l];
                    red_sum += row_r[ll] * gaussian_kernel[l];
                }
                // add vertical [1,2,4,2,1] sum on JUST middle row
                const int jj = j + k - 2;
                blue_sum += row_b_i[jj] * gaussian_kernel[k];
                green_sum += row_g_i[jj] * gaussian_kernel[k];
                red_sum += row_r_i[jj] * gaussian_kernel[k];
            }
        
            // normalize to ensure that the value is not greater than 255
            blue_blur.ptr<uchar>(i)[j] = static_cast<uchar>(blue_sum / 60);
            green_blur.ptr<uchar>(i)[j] = static_cast<uchar>(green_sum / 60);
            red_blur.ptr<uchar>(i)[j] = static_cast<uchar>(red_sum / 60);
        }
    }
    cv::merge(std::vector<cv::Mat>{blue_blur, green_blur, red_blur}, dst);
    return 0;
}

// output image needs to be of type 16SC3 (signed short) [-255, 255]

int sobelX3x3(const cv::Mat &src, cv::Mat &dst ) {
    // Input: source image
    // Output: signed short gradient image

    std::vector<cv::Mat> bgr;
    cv::split(src, bgr);

    // Separate the 3x3 Sobel X kernel as two 1D arrays (int)
    const int sobel_kernel_x[3] = {-1, 0, 1};
    const int sobel_kernel_y[3] = {1, 2, 1};

    // Use signed 16 bit Mats for the gradients
    cv::Mat blue_grad_x(bgr[0].size(), CV_16S);
    cv::Mat green_grad_x(bgr[1].size(), CV_16S);
    cv::Mat red_grad_x(bgr[2].size(), CV_16S);

    for (int i = 1; i < bgr[0].rows - 1; i++) {
        for (int j = 1; j < bgr[0].cols - 1; j++) {
            // int accumulators avoid uchar overflow before normalization
            int blue_grad_x_sum = 0;
            int green_grad_x_sum = 0;
            int red_grad_x_sum = 0;
            // create pointers to the current row (of center pixel)
            const uchar *row_b_i = bgr[0].ptr<uchar>(i);
            const uchar *row_g_i = bgr[1].ptr<uchar>(i);
            const uchar *row_r_i = bgr[2].ptr<uchar>(i);
            for (int k = 0; k < 3; k++) {
                for (int l = 0; l < 3; l++) {
                    const int ii = i + k - 1;
                    // create pointers to the current row (of window pixel)
                    const uchar *row_b = bgr[0].ptr<uchar>(ii);
                    const uchar *row_g = bgr[1].ptr<uchar>(ii);
                    const uchar *row_r = bgr[2].ptr<uchar>(ii);
                    // horizontal kernel sum over row i+k-1
                    const int ll = j + l - 1;
                    blue_grad_x_sum += row_b[ll] * sobel_kernel_x[l] * sobel_kernel_y[k];
                    green_grad_x_sum += row_g[ll] * sobel_kernel_x[l] * sobel_kernel_y[k];
                    red_grad_x_sum += row_r[ll] * sobel_kernel_x[l] * sobel_kernel_y[k];
                }
            }
        
            blue_grad_x.ptr<short>(i)[j] = static_cast<short>(blue_grad_x_sum / 4);
            green_grad_x.ptr<short>(i)[j] = static_cast<short>(green_grad_x_sum / 4);
            red_grad_x.ptr<short>(i)[j] = static_cast<short>(red_grad_x_sum / 4);
        }
    }
    cv::merge(std::vector<cv::Mat>{blue_grad_x, green_grad_x, red_grad_x}, dst);
    return 0;
}

int sobelY3x3(const cv::Mat &src, cv::Mat &dst ) {
    // Input: source image
    // Output: signed short gradient image

    std::vector<cv::Mat> bgr;
    cv::split(src, bgr);

    // Separate the 3x3 Sobel Y kernel as two 1D arrays (int)
    const int sobel_kernel_y[3] = {-1, 0, 1};
    const int sobel_kernel_x[3] = {1, 2, 1};

    // Use signed 16 bit Mats for the gradients
    cv::Mat blue_grad_y(bgr[0].size(), CV_16S);
    cv::Mat green_grad_y(bgr[1].size(), CV_16S);
    cv::Mat red_grad_y(bgr[2].size(), CV_16S);

    for (int i = 1; i < bgr[0].rows - 1; i++) {
        for (int j = 1; j < bgr[0].cols - 1; j++) {
            // int accumulators avoid uchar overflow before normalization
            int blue_grad_y_sum = 0;
            int green_grad_y_sum = 0;
            int red_grad_y_sum = 0;
            // create pointers to the current row (of center pixel)
            const uchar *row_b_i = bgr[0].ptr<uchar>(i);
            const uchar *row_g_i = bgr[1].ptr<uchar>(i);
            const uchar *row_r_i = bgr[2].ptr<uchar>(i);
            for (int k = 0; k < 3; k++) {
                for (int l = 0; l < 3; l++) {
                    // create pointers to the current row (of window pixel)
                    const int ii = i + k - 1;
                    const uchar *row_b = bgr[0].ptr<uchar>(ii);
                    const uchar *row_g = bgr[1].ptr<uchar>(ii);
                    const uchar *row_r = bgr[2].ptr<uchar>(ii);
                    // horizontal kernel sum over row i+k-1
                    const int ll = j + l - 1;
                    blue_grad_y_sum += row_b[ll] * sobel_kernel_x[l] * sobel_kernel_y[k];
                    green_grad_y_sum += row_g[ll] * sobel_kernel_x[l] * sobel_kernel_y[k];
                    red_grad_y_sum += row_r[ll] * sobel_kernel_x[l] * sobel_kernel_y[k];
                }
            }
        
            blue_grad_y.ptr<short>(i)[j] = static_cast<short>(blue_grad_y_sum / 4);
            green_grad_y.ptr<short>(i)[j] = static_cast<short>(green_grad_y_sum / 4);
            red_grad_y.ptr<short>(i)[j] = static_cast<short>(red_grad_y_sum / 4);
        }
    }
    cv::merge(std::vector<cv::Mat>{blue_grad_y, green_grad_y, red_grad_y}, dst);
    return 0;
}

int gaborFilter(const cv::Mat &src, cv::Mat &dst, int theta) {
    // Input: source image
    // Output: gabor filtered image
    // Applies a Gabor filter to the image 

    cv::Size ksize = cv::Size(31, 31); // size of the gabor filter kernel (in pixels)
    double sigma = 1.0; // standard deviation of the gaussian envelope (in pixels) smaller values mean more focused, larger values mean more spread out
    // theta = 0.0; // orientation of the normal to the parallel stripes of a Gabor function
    double lambd = 5.0; // wavelength of the sinusoidal factor (in pixels) smaller values mean more frequent, larger values mean less frequent
    double gamma = 0.5; // spatial aspect ratio (ratio of width to height of the Gaussian function)
    double psi = 0.0; // phase offset (in radians)
    int ktype = CV_32F; // type of filter coefficients (CV_32F or CV_64F)


    cv::Mat kernel = cv::getGaborKernel(ksize, sigma, theta, lambd, gamma, psi, ktype);
    cv::filter2D(src, dst, CV_8UC3, kernel);

    return 0;
}

int convertToUnsigned(const cv::Mat &short_image, cv::Mat &dst) {
    // Input: signed short image
    // Output: unsigned char image

    // builds an 8-bit unsigned image from the 16-bit signed image
    // by scaling, taking the abs value, and clamping to [0, 255]
    // alpha and beta essentially determing the contrast of the resultingimage
    cv::convertScaleAbs(short_image, dst, 1, 0);
    return 0;
}

int magnitude(const cv::Mat &sx, const cv::Mat &sy, cv::Mat &dst ) {
    // Input: two signed short gradient images
    // Output: magnitude image (8-bit unsigned)

    std::vector<cv::Mat> sx_bgr;
    cv::split(sx, sx_bgr);
    std::vector<cv::Mat> sy_bgr;
    cv::split(sy, sy_bgr);

    // Use euclidean distance to calculate the magnitude
    cv::Mat magnitude_blue = cv::Mat::zeros(sx_bgr[0].size(), CV_8U);
    cv::Mat magnitude_green = cv::Mat::zeros(sx_bgr[1].size(), CV_8U);
    cv::Mat magnitude_red = cv::Mat::zeros(sx_bgr[2].size(), CV_8U);
  

    for (int i = 0; i < sx_bgr[0].rows; i++) {
        for (int j = 0; j < sx_bgr[0].cols; j++) {
            magnitude_blue.ptr<uchar>(i)[j] = static_cast<uchar>(sqrt(sx_bgr[0].ptr<short>(i)[j] * sx_bgr[0].ptr<short>(i)[j] + sy_bgr[0].ptr<short>(i)[j] * sy_bgr[0].ptr<short>(i)[j]));
            magnitude_green.ptr<uchar>(i)[j] = static_cast<uchar>(sqrt(sx_bgr[1].ptr<short>(i)[j] * sx_bgr[1].ptr<short>(i)[j] + sy_bgr[1].ptr<short>(i)[j] * sy_bgr[1].ptr<short>(i)[j]));
            magnitude_red.ptr<uchar>(i)[j] = static_cast<uchar>(sqrt(sx_bgr[2].ptr<short>(i)[j] * sx_bgr[2].ptr<short>(i)[j] + sy_bgr[2].ptr<short>(i)[j] * sy_bgr[2].ptr<short>(i)[j]));
        }
    }
    cv::merge(std::vector<cv::Mat>{magnitude_blue, magnitude_green, magnitude_red}, dst);
    return 0;
}

int blurQuantize(const cv::Mat &src, cv::Mat &dst, int levels ) {
    // Input: blurred image (uchar)
    // Output: blurred + quantized image (uchar)

    // Quantizes the image into a fixed number of color buckets

    std::vector<cv::Mat> bgr;
    cv::split(src, bgr);

    const int bucket_size = 255 / levels;

    // Quantize the image by dividing the value by the number of levels and rounding down to the nearest integer
    for (int i = 0; i < bgr[0].rows; i++) {
        for (int j = 0; j < bgr[0].cols; j++) {
            // value is snapped down to a multiple of bucket_size (uchar uses integer division)
            bgr[0].ptr<uchar>(i)[j] = static_cast<uchar>((bgr[0].ptr<uchar>(i)[j] / bucket_size) * bucket_size);
            bgr[1].ptr<uchar>(i)[j] = static_cast<uchar>((bgr[1].ptr<uchar>(i)[j] / bucket_size) * bucket_size);
            bgr[2].ptr<uchar>(i)[j] = static_cast<uchar>((bgr[2].ptr<uchar>(i)[j] / bucket_size) * bucket_size);
        }
    }
    cv::merge(std::vector<cv::Mat>{bgr[0], bgr[1], bgr[2]}, dst);
    return 0;
}

int depthInvert(const cv::Mat &color, const cv::Mat &depth, cv::Mat &dst) {
    // Input: color image, depth image (shrunk)
    // Output: partially-inverted depth image (shrunk)

    // Inverts pixels in the image further than d0 distance

    dst = color.clone();
    CV_Assert(depth.size() == color.size());
    CV_Assert(depth.type() == CV_8UC1);

    const int d0 = 128;
    for (int i = 0; i < depth.rows; i++) {
        const uchar *dptr = depth.ptr<uchar>(i);
        cv::Vec3b *cptr = dst.ptr<cv::Vec3b>(i);
        for (int j = 0; j < depth.cols; j++) {
            if (dptr[j] <= d0) {
                cptr[j] = cv::Vec3b(255 - cptr[j][0],
                                    255 - cptr[j][1],
                                    255 - cptr[j][2]);
            }
        }
    }
    return 0;
}

int halo(const cv::Mat &color, const std::vector<cv::Rect> &faces, const cv::Mat &depth, cv::Mat &dst) {
    // Input: color image, faces, depth image (shrunk)
    // Output: halo image (shrunk)

    // Creates a yellow circle (halo) behind the detected face(s) in the image

    dst = color.clone();
    CV_Assert(depth.size() == color.size());
    CV_Assert(depth.type() == CV_8UC1);

    const cv::Scalar yellow(0, 255, 255); // BGR
    const cv::Rect bounds(0, 0, color.cols, color.rows);

    for (const cv::Rect &face : faces) {
        if (face.width < 20) {
            continue;
        }

        cv::Point center(face.x + face.width / 2, face.y + face.height / 2);
        const int head_radius = std::max(face.width, face.height) / 2;
        const int radius = static_cast<int>(head_radius * 2); // x2 larger than head

        // cv2.circle(img, center, radius, color, thickness=None, lineType=None, shift=None)
        cv::circle(dst, center, radius, yellow, -1);

        // get the depth of the center of the face (plus some offset to push behind the head)
        const int offset = 100;
        int depth_value = depth.ptr<uchar>(center.y)[center.x] - offset;
        depth_value = std::max(depth_value, 0);
   

        // Paste the part of the image with depth value greater than the halo back on top 
        // so the halo sits behind the head
        
        for (int i = 0; i < dst.rows; i++) {
            for (int j = 0; j < dst.cols; j++) {
                if (depth.ptr<uchar>(i)[j] > depth_value) {
                    dst.ptr<cv::Vec3b>(i)[j] = color.ptr<cv::Vec3b>(i)[j];
                }
            }
        }

    }
    return 0;
}

int blue(const cv::Mat &src, cv::Mat &dst) {
    // Input: source image
    // Output: blue image

    // Sets the blue channel to the average of the three rgb channels. Sets green and red channels to 0.
    std::vector<cv::Mat> bgr; // create a vector to store the channels
    cv::split(src, bgr); // split interleaved soruce image into planar image (separate channels)

    cv::Mat avg = (bgr[0] + bgr[1] + bgr[2]) / 3; // take average of all three channels
    cv::Mat zero = cv::Mat::zeros(avg.size(), avg.type());
    cv::merge(std::vector<cv::Mat>{avg, zero, zero}, dst); // B=avg, G=0, R=0
    return 0;
}

int emboss(const cv::Mat &src, cv::Mat &dst) {
    // Input: grey image
    // Output: emboss image

    // Creates an emboss effect by applying a kernel to the image (https://en.wikipedia.org/wiki/Image_embossing)

    dst = src.clone();

    // Create the emboss kernel as a 2D array (int)
    const int emboss_kernel[3][3] = {
        {1, 0, 0},
        {0, 0, 0},
        {0, 0, -1},
    };

    for (int i = 1; i < src.rows - 1; i++) {
        for (int j = 1; j < src.cols - 1; j++) {
            // int accumulators avoid uchar overflow before normalization
            int sum = 0;
            for (int k = 0; k < 3; k++) {
                for (int l = 0; l < 3; l++) {
                    // apply the gaussian kernel to the 5x5 window centered at (i, j)
                    // “treat bgr[0] as a matrix of bytes and give me the byte at (row, col).”
                    sum += (src.at<uchar>(i+k-1, j+l-1) * emboss_kernel[k][l]);  
                } 
            }
            // classic emboss uses something like result = grey + gradient + 128 then clamp to [0,255]
            dst.at<uchar>(i, j) = std::clamp(sum + 128, 0, 255); // add 128 to center the image and clamp to [0,255]
        }
    }
    return 0;
}

int faceEmbossQuarter(const cv::Mat &color, const std::vector<cv::Rect> &faces, cv::Mat &dst) {
    // Input: color image, faces
    // Output: embossed quarter and face rectangle image overlayed on top of the color image
    // Layer order: (1) normal color image, (2) embossed quarter, (3) embossed face rect

    dst = color.clone();

    static cv::Mat g_quarter_src;
    static bool g_quarter_load_attempted = false;
    g_quarter_src = cv::imread("../assets/quarter.png");

    cv::Mat grey;
    greyscale_filter(color, grey);

    const int minFace = 20;
    const cv::Rect img_bounds(0, 0, color.cols, color.rows); // set image bounds

    for (const cv::Rect &face : faces) {
        if (face.width < minFace) {
            continue;
        }

        cv::Point center(face.x + face.width / 2, face.y + face.height / 2);
        center.x = std::clamp(center.x, 0, color.cols - 1);
        center.y = std::clamp(center.y, 0, color.rows - 1);

        const int head_radius = std::max(face.width, face.height) / 2;
        const int radius = static_cast<int>(head_radius * 2);
        if (radius < 3) {
            continue;
        }

        // Create the quarter disk, either from the loaded image or a default grey disk
        const int diam = 2 * radius + 1;
        cv::Mat quarter;
        if (g_quarter_src.empty()) {
            quarter = cv::Mat(diam, diam, CV_8UC3, cv::Scalar(128, 128, 128));
            cv::circle(quarter, cv::Point(radius, radius), std::max(1, radius - 2),
                       cv::Scalar(100, 100, 100), 2);
        } else {
            cv::resize(g_quarter_src, quarter, cv::Size(diam, diam), 0, 0, cv::INTER_AREA);
        }

        cv::Mat quarter_grey; // greyscale quarter disk
        cv::Mat quarter_emb; // embossed quarter disk
        greyscale_filter(quarter, quarter_grey); // convert quarter to greyscale
        emboss(quarter_grey, quarter_emb);

        cv::Mat quarter_emb_bgr;
        cv::cvtColor(quarter_emb, quarter_emb_bgr, cv::COLOR_GRAY2BGR); // convert embossed quarter disk to bgr

        // create a mask for the quarter image so that corners are not pasted onto the color image
        cv::Mat disk_mask(diam, diam, CV_8UC1, cv::Scalar(0));
        cv::circle(disk_mask, cv::Point(radius, radius), radius, cv::Scalar(255), -1);

        const cv::Rect disk_tl(center.x - radius, center.y - radius, diam, diam);
        const cv::Rect dst_rect = disk_tl & img_bounds;
        if (dst_rect.width <= 0 || dst_rect.height <= 0) {
            continue;
        }
        const int sx = dst_rect.x - disk_tl.x;
        const int sy = dst_rect.y - disk_tl.y;
        const cv::Rect src_rect(sx, sy, dst_rect.width, dst_rect.height);

        quarter_emb_bgr(src_rect).copyTo(dst(dst_rect), disk_mask(src_rect)); // paste the embossed quarter disk onto the color image
    }

    // Paste the embossed face rectangle onto the color image where the face rectangle intersects the image bounds
    for (const cv::Rect &face : faces) {
        if (face.width < minFace) {
            continue;
        }
        cv::Rect r = face & img_bounds;
        if (r.width < 3 || r.height < 3) {
            continue;
        }
        cv::Mat grey_face = grey(r).clone(); // greyface rectangle
        cv::Mat emb_face; // embossed face rectangle
        emboss(grey_face, emb_face);
        cv::Mat emb_bgr;
        cv::cvtColor(emb_face, emb_bgr, cv::COLOR_GRAY2BGR);
        emb_bgr.copyTo(dst(r)); // paste the embossed face rectangle onto the color image
    }

    return 0;
}