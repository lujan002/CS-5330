// Luke Jansen
// 5/27/2026
// dirToFeatureVec.cpp
// Given a directory of images, it computes the feature vector for each image and saves it to a file:
// 1. Loop over and load all the images in the directory
// 2. Compute the features for each image
// 3. Save the features to a feature vector file

// Arguments:
// 1. Directory of images (B)
// 2. Feature set
// 3. Feature vector file output (for each image) (F)

#include <fstream>
#include <iostream>
#include <filesystem>
#include <unordered_map>
#include <opencv2/opencv.hpp>
#include <csv_util/csv_util.h>
#include "filters.h"
#include <da2-code/DA2Network.hpp>

namespace fs = std::filesystem;

// Load ResNet CSV (column 0 = path/filename, rest = features). Rows are keyed by basename
// so ../olympus/pic.0001.jpg matches /any/prefix/pic.0001.jpg.
static bool load_resnet_csv_by_basename(const fs::path &csvPath,
                                         std::unordered_map<std::string, std::vector<float>> &out) {
    std::vector<char *> filenames;
    std::vector<std::vector<float>> data;
    if (read_image_data_csv(csvPath.string().c_str(), filenames, data, 0) != 0) {
        return false;
    }
    for (size_t i = 0; i < filenames.size(); ++i) {
        std::string key = fs::path(filenames[i]).filename().string();
        out[std::move(key)] = std::move(data[i]);
        delete[] filenames[i];
    }
    return true;
}

static fs::path resolve_resnet_csv_path(const std::string &imgDirectory) {
    fs::path parent = fs::path(imgDirectory).parent_path();
    fs::path a = parent / "ResNet18_olym.csv";
    if (fs::exists(a)) {
        return a;
    }
    fs::path b = fs::current_path() / "ResNet18_olym.csv";
    if (fs::exists(b)) {
        return b;
    }
    return a;
}

std::vector<float> compute_feature_vector_baseline(const cv::Mat &image) {
    // Input: image (cv::Mat)
    // Output: feature vector (std::vector<float>)
    // Process:
    // 1. Read the feature set from the csv file
    // 2. Split the image into bgr channels
    // 3. Convolve the kernel by passing over rows and columns
    // 4. Normalize the feature vector
    // 5. Return the feature vector
    
    int dim = 7;
    const int baseline_kernel[dim] = {1, 1, 1, 1, 1, 1, 1};
    // OR read the feature set from the csv file???

    std::vector<float> featureVector;
    std::vector<cv::Mat> bgr;
    cv::split(image, bgr);
    // center pixel for the 7x7 baseline kernel test
    int cX = image.cols / 2;
    int cY = image.rows / 2;

    for (int k = 0; k < dim; k++) { // collumn index for the kernel
        for (int l = 0; l < dim; l++) { // row index for the kernel
            int pixel_y_index = cY + k - (dim / 2);
            int pixel_x_index = cX + l - (dim / 2);
            featureVector.push_back(static_cast<float>(bgr[0].at<uchar>(pixel_y_index, pixel_x_index)));
            featureVector.push_back(static_cast<float>(bgr[1].at<uchar>(pixel_y_index, pixel_x_index)));
            featureVector.push_back(static_cast<float>(bgr[2].at<uchar>(pixel_y_index, pixel_x_index)));
        }
    }

    return featureVector;

}

std::vector<float> compute_feature_vector_color(const cv::Mat &image, std::string hist_type, int numBuckets, std::string spacing_type, float weight = 1.0) {
    // Input: image (cv::Mat)
    // Output: feature vector (std::vector<float>) of length numBuckets * 3
    // Process:
    // 1. Split the image into bgr channels
    // 2. Compute the histogram for each channel
    // 3. Combine the histograms into one 3D histogram
    // 4. Normalize the histogram
    // 5. Flatten the 3D histogram into a 1D feature vector
    // 6. Return the feature vector

    int px;
    int py;
    int i_start;
    int j_start;
    if (hist_type == "center") {
        px = image.cols / 2;
        py = image.rows / 2;
        i_start = image.rows / 4;
        j_start = image.cols / 4;
    } else if (hist_type == "whole") {
        px = image.cols;
        py = image.rows;
        i_start = 0;
        j_start = 0;
    }
    int spacing;
    if (spacing_type == "uniform") {
        spacing = 256 / numBuckets;
    }
    int span = 1;


    // 1. Split the image into bgr channels
    std::vector<float> featureVector;
    std::vector<cv::Mat> bgr;
    cv::split(image, bgr);
    cv::Mat bucketIDs(image.size(), CV_8UC3); // holds the bucket IDs for each pixel
    int blue_bucket_id = 0;
    int green_bucket_id = 0;
    int red_bucket_id = 0;

    const int sizes[3] = {numBuckets, numBuckets, numBuckets};
    cv::Mat global_hist = cv::Mat::zeros(3, sizes, CV_32F);

    // 2. Compute the histogram for each channel
    for (int i = i_start; i < px; i += span) { // global collumn indexes
        for (int j = j_start; j < py; j += span) { // global row index
            // use span to only compute the histogram over a sparse sample of pixels
            // bucketIDs[i][j] = bgr[0][i][j] / spacing;
            int b = bgr[0].at<uchar>(i, j);
            int g = bgr[1].at<uchar>(i, j);
            int r = bgr[2].at<uchar>(i, j);

            if (spacing_type == "log") {
                blue_bucket_id = (log(b+1) / log(256)) * numBuckets;
                green_bucket_id = (log(g+1) / log(256)) * numBuckets;
                red_bucket_id = (log(r+1) / log(256)) * numBuckets;
            }
            if (spacing_type == "uniform") {
                blue_bucket_id = b / spacing;
                green_bucket_id = g / spacing;
                red_bucket_id = r / spacing;
            }
            //clamp the bucket ids to the range [0, numBuckets-1]
            blue_bucket_id = std::clamp(blue_bucket_id, 0, numBuckets-1);
            green_bucket_id = std::clamp(green_bucket_id, 0, numBuckets-1);
            red_bucket_id = std::clamp(red_bucket_id, 0, numBuckets-1);

            // 3. Combine the histograms into one 3D histogram by incrementing the global bucket id by one
            global_hist.at<float>(blue_bucket_id, green_bucket_id, red_bucket_id)++;

            // // create a 3D histogram by combining the three RGB histograms into one histogram (3D array)
            // featureVector.push_back(static_cast<float>(blue_bucket_id));
            // featureVector.push_back(static_cast<float>(green_bucket_id));
            // featureVector.push_back(static_cast<float>(red_bucket_id));

        }
    }
    // 4. Normalize the histogram by the number of pixels in the image (or subset of image in the center case)
    for (int i = 0; i < numBuckets; i++) {
        for (int j = 0; j < numBuckets; j++) {
            for (int k = 0; k < numBuckets; k++) {
                global_hist.at<float>(i, j, k) /= (static_cast<float>(px/span) * static_cast<float>(py/span));
            }
        }
    }

    // 5. Flatten the 3D histogram into a 1D feature vector
    // since the feature vector has to be 1D, we need to flatten the 3D feature vector
    // I choose to do channel-major order: first channels (b, g, r), then pixels (i, j)
    // convert to CV_32F
    for (int i = 0; i < numBuckets; i++) {
        for (int j = 0; j < numBuckets; j++) {
            for (int k = 0; k < numBuckets; k++) {
                featureVector.push_back(static_cast<float>(global_hist.at<float>(i, j, k)*weight));
            }
        }
    }
    
    return featureVector;
}

std::vector<float> compute_feature_vector_color_1D(const cv::Mat &image, std::string hist_type, int numBuckets, std::string spacing_type, float weight = 1.0) {
    // Input: image (cv::Mat)
    // Output: feature vector (std::vector<float>) of length numBuckets * 3
    // Process:
    // 1. Compute the histogram for the image
    // 2. Normalize the histogram
    // 3. Flatten the histogram into a 1D feature vector
    // 4. Return the feature vector

    int px;
    int py;
    int i_start;
    int j_start;
    if (hist_type == "center") {
        px = image.cols / 2;
        py = image.rows / 2;
        i_start = image.rows / 4;
        j_start = image.cols / 4;
    } else if (hist_type == "whole") {
        px = image.cols;
        py = image.rows;
        i_start = 0;
        j_start = 0;
    }
    int spacing = 256 / numBuckets;
    int span = 1;
    int bucketID;

    // 1. Split the image into bgr channels
    std::vector<float> featureVector;

    cv::Mat global_hist = cv::Mat::zeros(1, numBuckets, CV_32F);

    // 2. Compute the histogram for each channel
    for (int i = i_start; i < px; i += span) { // global collumn indexes
        for (int j = j_start; j < py; j += span) { // global row index
            if (spacing_type == "log") {
                bucketID = (log(image.at<uchar>(i, j)+1) / log(256)) * numBuckets;
            }
            if (spacing_type == "uniform") {
                bucketID = image.at<uchar>(i, j) / spacing;
            }
            //clamp the bucket id to the range [0, numBuckets-1]
            bucketID = std::clamp(bucketID, 0, numBuckets-1);

            global_hist.at<float>(bucketID)++;
        }
    }
    // 3. Normalize the histogram by the number of pixels in the image (or subset of image in the center case)
    for (int i = 0; i < numBuckets; i++) {
        global_hist.at<float>(i) /= (static_cast<float>(px/span) * static_cast<float>(py/span));
    }

    // 4. Flatten the 3D histogram into a 1D feature vector

    for (int i = 0; i < numBuckets; i++) {
        featureVector.push_back(static_cast<float>(global_hist.at<float>(i)*weight));
    }
    // apply a weight to the feature vector to account for the four Gabor filters
    return featureVector;
}

std::vector<float> compute_feature_vector_texture(const cv::Mat &image) {
    // Input: image (cv::Mat)
    // Output: feature vector (std::vector<float>)
    // Process:
    // 1. Run magnitude filter from project1 (filter.cpp)
    // 2. Compute the histogram of the magnitude image
    // 3. Flatten the histogram into a 1D feature vector
    // 4. Return the feature vector
    cv::Mat sobelX;
    cv::Mat sobelY;
    sobelX3x3(image, sobelX);
    sobelY3x3(image, sobelY);
    cv::Mat magnitude_image;
    magnitude(sobelX, sobelY, magnitude_image);
    // convert the magnitude image to greyscale (this is recommended for texture histograms)
    cv::cvtColor(magnitude_image, magnitude_image, cv::COLOR_BGR2GRAY);

    // treats the magnitude image as a full rgb color image for the histogram calculation
    // this results in a 3D histogram with N^3 buckets like the color histogram
    std::vector<float> featureVector = compute_feature_vector_color_1D(magnitude_image, "whole", 8, "log");
    return featureVector;
}   

std::vector<float> compute_feature_vector_depth(const cv::Mat &image) {
    // Input: image (cv::Mat)
    // Output: feature vector (std::vector<float>)
    // Process:
    // 1. Run the DA2Network on the image
    // 2. Get the output feature vector
    // 3. Return the feature vector
    // Setup Depth Anything Network

    const float reduction = 0.5;
    DA2Network da_net( "../../project1/src/da2-code/model_fp16.onnx" );
    float scale_factor = 256.0 / (image.rows*reduction);
    // set the network input
    da_net.set_input(image, scale_factor);
    // run the network
    cv::Mat depth;
    da_net.run_network(depth, image.size());

    // create a depth histogram 
    std::vector<float> featureVector1 = compute_feature_vector_color_1D(depth, "whole", 8, "uniform");
    std::vector<float> featureVector2 = compute_feature_vector_color_1D(depth, "center", 8, "uniform");

    // append the two depth histograms together into one depth histogram
    featureVector1.insert(featureVector1.end(), featureVector2.begin(), featureVector2.end());
    return featureVector1;
}

std::vector<float> compute_feature_vector_gabor(const cv::Mat &image, int theta, float weight = 1.0) {
    // Input: image (cv::Mat)
    // Output: feature vector (std::vector<float>)
    // Process:
    // 1. Run the Gabor filter on the image
    // 2. Compute the histogram of the Gabor filtered image
    // 3. Flatten the histogram into a 1D feature vector
    // 4. Return the feature vector

    cv::Mat gabor_image;
    gaborFilter(image, gabor_image, theta);
    std::vector<float> featureVector = compute_feature_vector_color_1D(gabor_image, "whole", 16, "uniform", weight);
    return featureVector;
}

int main(int argc, char *argv[]) {
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <directory> <feature set> <feature vector file output>" << std::endl;
        return 1;
    }

    std::string imgDirectory = argv[1];
    std::string featureSet = argv[2];
    std::string featureVectorFileOutput = argv[3];

    // Truncate output so each run starts fresh (no dummy CSV row — empty first line breaks read_image_data_csv).
    {
        std::ofstream truncate(featureVectorFileOutput, std::ios::binary | std::ios::trunc);
    }

    std::unordered_map<std::string, std::vector<float>> resnetByBasename;
    if (featureSet == "depth") {
        fs::path resnetCsv = resolve_resnet_csv_path(imgDirectory);
        if (!fs::exists(resnetCsv)) {
            std::cerr << "Error: ResNet18_olym.csv not found (tried "
                      << (fs::path(imgDirectory).parent_path() / "ResNet18_olym.csv").string()
                      << " and " << (fs::current_path() / "ResNet18_olym.csv").string() << ")\n";
            return 1;
        }
        if (!load_resnet_csv_by_basename(resnetCsv, resnetByBasename)) {
            std::cerr << "Error: failed to read ResNet CSV: " << resnetCsv << std::endl;
            return 1;
        }
        std::cout << "Loaded " << resnetByBasename.size() << " ResNet rows from " << resnetCsv << std::endl;
    }

    // Loop over all the images in the image directory
    try { 
        if (fs::exists(imgDirectory) && fs::is_directory(imgDirectory)) {
            for (const auto &entry : fs::directory_iterator(imgDirectory)) {
                // Skip directories and non-file entries; this does NOT tell image vs non-image.
                if (!entry.is_regular_file()) {
                    continue;
                }
                std::string image_path = entry.path().string();

                // OpenCV decodes supported image formats; empty means not a loadable image.
                cv::Mat image = cv::imread(image_path, cv::IMREAD_COLOR);
                if (image.empty()) {
                    std::cerr << "Could not read the image: " << image_path << std::endl;
                    continue;
                }
                //debugging: display the image
                // cv::imshow("Image", image);
        
                // if the user presses the 'q' key, the program will exit
                // while (true) {
                //     if (cv::waitKey(1) == 'q') {
                //         break;
                //     }
                // }

                // compute the feature vector for the image
                std::vector<float> featureVector;
                if (featureSet == "baseline") {
                    featureVector = compute_feature_vector_baseline(image);
                } else if (featureSet == "color") {
                    featureVector = compute_feature_vector_color(image, "whole", 16, "uniform");
                } else if (featureSet == "spatial_color") {
                    std::vector<float> featureVector1 = compute_feature_vector_color(image, "whole", 16, "uniform");
                    std::vector<float> featureVector2 = compute_feature_vector_color(image, "center", 16, "uniform");
                    // append the two feature vectors together into one feature vector
                    featureVector1.insert(featureVector1.end(), featureVector2.begin(), featureVector2.end());
                    featureVector = featureVector1;
                } else if (featureSet == "sobel") {
                    std::vector<float> featureVector1 = compute_feature_vector_color(image, "whole", 16, "uniform");
                    std::vector<float> featureVector2 = compute_feature_vector_texture(image);
                    // append the two feature vectors together into one feature vector
                    featureVector1.insert(featureVector1.end(), featureVector2.begin(), featureVector2.end());
                    featureVector = featureVector1;
                    // extension idea: create a 4D histogram using the greyscale texture histogram as a 4th dim 
                } else if (featureSet == "depth") {
                    std::vector<float> featureVector1 = compute_feature_vector_depth(image);
                    // also run the normal color histogram
                    // std::vector<float> featureVector2 = compute_feature_vector_color(image, "whole", 16, "uniform");
                    // featureVector1.insert(featureVector1.end(), featureVector2.begin(), featureVector2.end());
                    std::string basename = fs::path(image_path).filename().string();
                    auto it = resnetByBasename.find(basename);
                    if (it != resnetByBasename.end()) {
                        featureVector1.insert(featureVector1.end(), it->second.begin(), it->second.end());
                    } else {
                        std::cerr << "Warning: no ResNet row for basename \"" << basename
                                  << "\" (image " << image_path << ")\n";
                    }
                    featureVector = std::move(featureVector1);
                } else if (featureSet == "gabor") {
                    // segmenet the image into 4 quadrants and compute the feature vector for each quadrant
                    std::array<cv::Mat, 5> section;
                    section[0] = image(cv::Rect(0, 0, image.cols/2, image.rows/2));
                    section[1] = image(cv::Rect(image.cols/2, 0, image.cols/2, image.rows/2));
                    section[2] = image(cv::Rect(0, image.rows/2, image.cols/2, image.rows/2));
                    section[3] = image(cv::Rect(image.cols/2, image.rows/2, image.cols/2, image.rows/2));
                    section[4] = image(cv::Rect(image.cols/4, image.rows/4, image.cols/2, image.rows/2)); // middle of the image
                    std::vector<float> featureVector1;
                    for (int i = 0; i < 5; i++) {
                        // section index
                        // for each section, compute 4 angular feature vectors and append it to the feature vector
                        for (int j = 0; j < 4; j++) {
                            // angular index
                            std::vector<float> featureVectorAngle = compute_feature_vector_gabor(section[i], j*45, 1.0/section.size());
                            featureVector1.insert(featureVector1.end(), featureVectorAngle.begin(), featureVectorAngle.end());
                        }
                    }
                    // add color histogram to the feature vector
                    std::vector<float> featureVector2 = compute_feature_vector_color(image, "whole", 16, "uniform");
                    featureVector1.insert(featureVector1.end(), featureVector2.begin(), featureVector2.end());
                    featureVector = std::move(featureVector1);
                } else {
                    std::cerr << "Error: Invalid feature set: " << featureSet << std::endl;
                    return 1;
                }

                // write the features to the CSV file
                append_image_data_csv(featureVectorFileOutput.c_str(), image_path.c_str(), featureVector);
                std::cout << "Wrote feature vector for " << image_path << std::endl;
            }
        }
    } catch (const fs::filesystem_error &e) {
            std::cerr << "Error: " << e.what() << std::endl;
            return 1;
    }
        

    return 0;
}

