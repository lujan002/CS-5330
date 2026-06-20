// Luke Jansen
// 5/27/2026
// imgMatch.cpp
// Using the feature vector file, compares the target image to all other images in 
// the feature vector file using distance metric, and identifies the top N matches.

// Arguements:
// 1. Target image (T) 
// 2. Feature vector file (F)
// 3. Distance metric (D)
// 4. Desired number of matches (N)

#include <iostream>
#include <opencv2/opencv.hpp>
#include <csv_util/csv_util.h>
#include <cmath>

int main(int argc, char *argv[]) {
    if (argc != 5) {
        std::cerr << "Usage: " << argv[0] << " <target image> <feature vector file> <desired number of matches>" << std::endl;
        return 1;
    }

    std::string targetImage = argv[1];
    // std::string featureSet = argv[2];
    std::string featureVectorFile = argv[2];
    std::string distanceMetric = argv[3];
    int numMatches = atoi(argv[4]); // convert the string to an integer

    // read the feature vector file
    std::vector<char *> filenames;
    std::vector<std::vector<float>> data;
    read_image_data_csv(featureVectorFile.c_str(), filenames, data);

    // compute the feature vector for the target image 
    // (DONT NEED TO DO THIS SINCE WE ALREADY HAVE THE FEATURE VECTOR FOR THE TARGET IMAGE)
    // std::vector<float> targetFeatureVector = compute_feature_vector(targetImage, featureSet);

    // find the target image in the filenames and extract it
    int targetImageIndex = -1;
    for (int i = 0; i < filenames.size(); i++) {
        if (strcmp(filenames[i], targetImage.c_str()) == 0) {
            targetImageIndex = i;
            break;
        }
    }
    if (targetImageIndex == -1) {
        std::cerr << "Target image not found in feature vector file" << std::endl;
        return 1;
    }

    std::vector<float> distances;
    std::vector<char *> sorted_filenames;
    if (distanceMetric == "least_squares") {
        // compute the least squares distance between the target feature vector and each of the feature vectors in the feature vector file
        for (int i = 0; i < data.size(); i++) {
            // i is the index of the image
            float distance = 0;
            for (int j = 0; j < data[targetImageIndex].size(); j++) {
                // j is the index inside the feature vector
                distance += (data[targetImageIndex][j] - data[i][j]) * (data[targetImageIndex][j] - data[i][j]);
            }
            distances.push_back(distance);
            sorted_filenames.push_back(filenames[i]);
        }
    }
    if (distanceMetric == "single_hist_intersect") {
        // compute the single histogram intersection distance between the target feature vector and each of the feature vectors in the feature vector file
        for (int i = 0; i < data.size(); i++) {
            // i is the index of the image
            float intersect_score = 0;
            for (int j = 0; j < data[targetImageIndex].size(); j++) {
                // j is the index inside the feature vector (global bucket id)
                // use span to only compute the intersection over a sparse sample of pixels for the target image
                // span has already been applied to the feature vector data, so we don't need to apply it again
                intersect_score += std::min(data[targetImageIndex][j], data[i][j]);            
                }
            float distance = 1 - intersect_score;
            distances.push_back(distance);
            sorted_filenames.push_back(filenames[i]);
        }
    }
    // if (distanceMetric == "multi_hist_custom") {
    //     // compute the multi histogram custom distance between the target feature vector and each of the feature vectors in the feature vector file
    //     for (int i = 0; i < data.size(); i++) {
    //         // i is the index of the image
    //         // float intersect_score = 0;
    //         float chi_squared_distance = 0;
    //         for (int j = 0; j < data[targetImageIndex].size(); j++) {
    //             // j is the index inside the feature vector (global bucket id)
    //             // use span to only compute the intersection over a sparse sample of pixels for the target image
    //             // ONLY DIVIDE BY 2.0 FOR THIS MULTI HIST CUSTOM DISTANCE BECAUSE WE ARE COMBINING TWO HISTOGRAMS
    //             // intersect_score += std::min(data[targetImageIndex][j], data[i][j]) / 2.0;
    //             chi_squared_distance += ((data[targetImageIndex][j] - data[i][j]) * (data[targetImageIndex][j] - data[i][j])) / (data[targetImageIndex][j] + data[i][j] + 1e-10);
    //         }

    //         float distance = chi_squared_distance;
    //         distances.push_back(distance);
    //         sorted_filenames.push_back(filenames[i]);
    //     }
    // }
    if (distanceMetric == "chi_squared_distance") {
        // compute the multi histogram custom distance between the target feature vector and each of the feature vectors in the feature vector file
        for (int i = 0; i < data.size(); i++) {
            // i is the index of the image
            // float intersect_score = 0;
            float chi_squared_distance = 0;
            for (int j = 0; j < data[targetImageIndex].size(); j++) {
                // j is the index inside the feature vector (global bucket id)
                // use span to only compute the intersection over a sparse sample of pixels for the target image
                // ONLY DIVIDE BY 2.0 FOR THIS MULTI HIST CUSTOM DISTANCE BECAUSE WE ARE COMBINING TWO HISTOGRAMS
                // intersect_score += std::min(data[targetImageIndex][j], data[i][j]) / 2.0;
                chi_squared_distance += ((data[targetImageIndex][j] - data[i][j]) * (data[targetImageIndex][j] - data[i][j])) / (data[targetImageIndex][j] + data[i][j] + 1e-10);
            }

            float distance = chi_squared_distance;
            distances.push_back(distance);
            sorted_filenames.push_back(filenames[i]);
        }
    }
    // if (distanceMetric == "deep_network") {
    //     float A_mag;
    //     float cosine_distance;
    //     for (int j = 0; j < data[targetImageIndex].size(); j++) {
    //         A_mag += data[targetImageIndex][j] * data[targetImageIndex][j];
    //     }
    //     A_mag = std::sqrt(A_mag);
        
    //     // compute the multi histogram custom distance between the target feature vector and each of the feature vectors in the feature vector file
    //     for (int i = 0; i < data.size(); i++) {
    //         // i is the index of the image
    //         float B_mag = 0;
    //         float dot_product = 0;
    //         for (int j = 0; j < data[targetImageIndex].size(); j++) {
    //             // j is the index inside the feature vector (global bucket id)
    //             B_mag += data[i][j] * data[i][j];
    //             dot_product += data[targetImageIndex][j] * data[i][j];
    //         }
    //         B_mag = std::sqrt(B_mag);
    //         cosine_distance = dot_product / (A_mag * B_mag + 1e-10);
    //         float distance = 1 - cosine_distance;
    //         distances.push_back(distance);
    //         sorted_filenames.push_back(filenames[i]);
    //     }
    // }
    if (distanceMetric == "cosine_distance") {
        float A_mag;
        float cosine_distance;
        for (int j = 0; j < data[targetImageIndex].size(); j++) {
            A_mag += data[targetImageIndex][j] * data[targetImageIndex][j];
        }
        A_mag = std::sqrt(A_mag);
        
        // compute the multi histogram custom distance between the target feature vector and each of the feature vectors in the feature vector file
        for (int i = 0; i < data.size(); i++) {
            // FOR DNN FEATURES ONLY: skip the first 16 features because they are the depth histogram
            // if (i < 16) {continue;}

            // i is the index of the image
            float B_mag = 0;
            float dot_product = 0;
            for (int j = 0; j < data[targetImageIndex].size(); j++) {
                // j is the index inside the feature vector (global bucket id)
                B_mag += data[i][j] * data[i][j];
                dot_product += data[targetImageIndex][j] * data[i][j];
            }
            B_mag = std::sqrt(B_mag);
            cosine_distance = dot_product / (A_mag * B_mag + 1e-10);
            float distance = 1 - cosine_distance;
            distances.push_back(distance);
            sorted_filenames.push_back(filenames[i]);
        }
    }
    
    // remove the target image from the filenames and data
    // filenames.erase(filenames.begin() + targetImageIndex);
    // data.erase(data.begin() + targetImageIndex);
    sorted_filenames.erase(sorted_filenames.begin() + targetImageIndex);
    distances.erase(distances.begin() + targetImageIndex);

    // sort the distances and print the top N matches
    std::vector<std::pair<float, char *>> distances_and_filenames;
    for (int i = 0; i < distances.size(); i++) {
        distances_and_filenames.push_back(std::make_pair(distances[i], sorted_filenames[i]));
    }
    std::sort(distances_and_filenames.begin(), distances_and_filenames.end());

    
    for (int i = 0; i < numMatches; i++) {
        // show the top N matches
        std::cout << distances_and_filenames[i].first << " " << distances_and_filenames[i].second << std::endl;
        cv::Mat image = cv::imread(distances_and_filenames[i].second, cv::IMREAD_COLOR);
        cv::imshow("Match " + std::to_string(i+1) + " " + distances_and_filenames[i].second, image);
        cv::waitKey(50);
    }
    cv::waitKey(0);
    return 0;
}
