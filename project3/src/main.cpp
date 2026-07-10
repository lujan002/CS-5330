// Luke Jansen
// 6/17/2026
// main.cpp
// This file is used to:
// 1. threshold the image
// 2. clean up the binary image (morphological operations: shinking/dilation)
// 3. segment features from the binary image (connected components analysis)
// 4. compute feature vectors
// 5. collect training data (stores feature vectors in a CSV file)
// 6. classify new images
// 7. evaluate the performanc
// 8. display the results

#include <algorithm>
#include <limits>
#include <opencv2/opencv.hpp>
#include <set>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <numeric> // For std::accumulate
#include <cmath>   // For std::sqrt
#include <iostream>


struct RegionTrack {
    int label;
    int display_id;
    cv::Point2d centroid;
    cv::Vec3b color;
};

struct ObjectRecord {
    std::string name;
    std::vector<double> features;
};

std::vector<ObjectRecord> loadObjectDatabase(const std::string& filename) {
    std::vector<ObjectRecord> database;
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Could not open " << filename << std::endl;
        return database;
    }
    std::string line;
    while (std::getline(file, line)) {
        if (line.empty()) {
            continue;
        }
        std::stringstream ss(line);
        std::string cell;
        ObjectRecord record;
        // First column: object name
        std::getline(ss, record.name, ',');
        // Remaining columns: feature values
        while (std::getline(ss, cell, ',')) {
            record.features.push_back(std::stod(cell));
        }
        database.push_back(record);
    }
    return database;
}

bool saveObjectDatabase(const std::string& filename, const std::vector<ObjectRecord>& database) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Could not open " << filename << " to save database." << std::endl;
        return false;
    }

    for (const auto& record : database) {
        file << record.name;
        for (double feature : record.features) {
            file << "," << feature;
        }
        file << std::endl;
    }

    return true;
}

int findObjectIndexByName(const std::vector<ObjectRecord>& database, const std::string& name) {
    auto match = std::find_if(database.begin(), database.end(),
        [&name](const ObjectRecord& record) {
            return record.name == name;
        });
    if (match == database.end()) {
        return -1;
    }
    return static_cast<int>(std::distance(database.begin(), match));
}

std::vector<std::vector<int>> createEmptyConfusionMatrix(const std::vector<ObjectRecord>& database) {
    return std::vector<std::vector<int>>(database.size(), std::vector<int>(database.size(), 0));
}

std::vector<std::vector<int>> loadConfusionMatrix(const std::string& filename,
                                                  const std::vector<ObjectRecord>& database) {
    std::vector<std::vector<int>> matrix = createEmptyConfusionMatrix(database);
    std::ifstream file(filename);
    if (!file.is_open()) {
        return matrix;
    }

    std::string line;
    std::vector<std::string> column_names;
    if (std::getline(file, line)) {
        std::stringstream header(line);
        std::string cell;
        std::getline(header, cell, ','); // top-left label cell
        while (std::getline(header, cell, ',')) {
            column_names.push_back(cell);
        }
    }

    while (std::getline(file, line)) {
        if (line.empty()) {
            continue;
        }

        std::stringstream row_stream(line);
        std::string row_name;
        std::getline(row_stream, row_name, ',');
        int row_idx = findObjectIndexByName(database, row_name);
        if (row_idx < 0) {
            continue;
        }

        std::string cell;
        int csv_col = 0;
        while (std::getline(row_stream, cell, ',')) {
            if (csv_col < static_cast<int>(column_names.size())) {
                int col_idx = findObjectIndexByName(database, column_names[csv_col]);
                if (col_idx >= 0 && !cell.empty()) {
                    matrix[row_idx][col_idx] = std::stoi(cell);
                }
            }
            csv_col++;
        }
    }

    return matrix;
}

bool saveConfusionMatrix(const std::string& filename,
                         const std::vector<std::vector<int>>& matrix,
                         const std::vector<ObjectRecord>& database) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Could not open " << filename << " to save confusion matrix." << std::endl;
        return false;
    }

    file << "actual/predicted";
    for (const auto& record : database) {
        file << "," << record.name;
    }
    file << std::endl;

    for (int row = 0; row < static_cast<int>(database.size()); row++) {
        file << database[row].name;
        for (int col = 0; col < static_cast<int>(database.size()); col++) {
            int value = 0;
            if (row < static_cast<int>(matrix.size()) && col < static_cast<int>(matrix[row].size())) {
                value = matrix[row][col];
            }
            file << "," << value;
        }
        file << std::endl;
    }

    return true;
}

std::vector<double> computeFeatureStdVector(const std::vector<ObjectRecord>& database) {
    std::vector<double> std_vector;
    for (int col_idx = 0; col_idx < static_cast<int>(database[0].features.size()); col_idx++) {
        std::vector<double> col;
        for (const auto& record : database) {
            col.push_back(record.features[col_idx]);
        }

        double sum = std::accumulate(col.begin(), col.end(), 0.0);
        double mean = sum / col.size();

        double squared_diff = 0.0;
        for (double val : col){
            squared_diff += (val - mean) * (val - mean);
        }

        std_vector.push_back(std::sqrt(squared_diff / col.size()));
    }
    return std_vector;
}

std::vector<std::pair<std::string, double>> rankObjectMatches(
    const std::vector<double>& feature_vector_img,
    const std::vector<ObjectRecord>& database,
    const std::vector<double>& std_vector) {
    std::vector<std::pair<std::string, double>> name_dist_pairs;

    for (int object_idx = 0; object_idx < static_cast<int>(database.size()); object_idx++) {
        std::string name = database[object_idx].name;
        std::vector<double> feature_vector_db = database[object_idx].features;

        assert(feature_vector_db.size() == feature_vector_img.size());

        double euclidean_dist = 0.0;
        for (int feature_idx = 0; feature_idx < static_cast<int>(feature_vector_img.size()); feature_idx++){
            double std_value = std_vector[feature_idx];
            if (std_value == 0.0) {
                std_value = 1.0;
            }
            double diff = feature_vector_img[feature_idx] - feature_vector_db[feature_idx];
            euclidean_dist += diff * diff / std_value / std_value;
        }
        name_dist_pairs.push_back({name, euclidean_dist});
    }

    std::sort(name_dist_pairs.begin(), name_dist_pairs.end(), [](const auto& a, const auto& b) {
        return a.second < b.second;
    });

    return name_dist_pairs;
}


int threshold(const cv::Mat &src, cv::Mat &dst){
    // input: BGR or grayscale image, CV_8UC3 or CV_8UC1
    // output: binary image, CV_8UC1

    cv::Mat gray;
    if (src.channels() == 3) {
        cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = src;
    }

    dst = cv::Mat::zeros(gray.size(), CV_8UC1);

    // Apply Sauvola threshold to the image.
    const int w = 50;       // local window size; use an odd number
    const int n = w / 2;    // window radius
    const double k = 0.5;   // standard deviation weight
    const double R = 128.0; // expected max std dev for 8-bit grayscale

    cv::Scalar global_mean;
    cv::Scalar global_stddev;
    cv::meanStdDev(gray, global_mean, global_stddev);

    for (int i = 0; i < gray.rows; i++) {
        for (int j = 0; j < gray.cols; j++) {
            // int x0 = std::max(0, j - n);
            // int y0 = std::max(0, i - n);
            // int x1 = std::min(gray.cols, j + n + 1);
            // int y1 = std::min(gray.rows, i + n + 1);

            // // ensure the window does not extend off the image
            // cv::Rect window(x0, y0, x1 - x0, y1 - y0);

            // // calculate the mean and stddev from the window
            // cv::Scalar mean;
            // cv::Scalar stddev;
            // cv::meanStdDev(gray(window), mean, stddev);

            // // set pixel-specific threshold
            // double T = mean[0] * (1.0 + k * ((stddev[0] / R) - 1.0));
            
            // based on threshold, check if binary image should get 255 or 0 at that pixel
            dst.at<uchar>(i, j) = gray.at<uchar>(i, j) < global_mean[0] ? 255 : 0;
        }
    }
    return 0;
}

int chrominance_threshold(const cv::Mat &src, cv::Mat &dst){
    // input: BGR color image, CV_8UC3
    // output: binary image, CV_8UC1

    cv::Mat lab;
    cv::cvtColor(src, lab, cv::COLOR_BGR2Lab);

    std::vector<cv::Mat> channels;
    cv::split(lab, channels);
    // channels[0] = L brightness
    // channels[1] = a chrominance green-red
    // channels[2] = b chrominance blue-yellow

    cv::Mat a, b, chroma;
    channels[1].convertTo(a, CV_32F);
    channels[2].convertTo(b, CV_32F);

    a -= 128.0;
    b -= 128.0;

    cv::magnitude(a, b, chroma);
    cv::normalize(chroma, chroma, 0, 255, cv::NORM_MINMAX);
    
    cv::Mat gray;
    chroma.convertTo(gray, CV_8UC1);

    dst = cv::Mat::zeros(gray.size(), CV_8UC1);

    // Apply Sauvola threshold to the image.
    const int w = 50;       // local window size; use an odd number
    const int n = w / 2;    // window radius
    const double k = 0.5;   // standard deviation weight
    const double R = 128.0; // expected max std dev for 8-bit grayscale

    cv::Scalar global_mean;
    cv::Scalar global_stddev;
    cv::meanStdDev(gray, global_mean, global_stddev);

    for (int i = 0; i < gray.rows; i++) {
        for (int j = 0; j < gray.cols; j++) {
            // int x0 = std::max(0, j - n);
            // int y0 = std::max(0, i - n);
            // int x1 = std::min(gray.cols, j + n + 1);
            // int y1 = std::min(gray.rows, i + n + 1);

            // // ensure the window does not extend off the image
            // cv::Rect window(x0, y0, x1 - x0, y1 - y0);

            // // calculate the mean and stddev from the window
            // cv::Scalar mean;
            // cv::Scalar stddev;
            // cv::meanStdDev(gray(window), mean, stddev);

            // // set pixel-specific threshold
            // double T = mean[0] * (1.0 + k * ((stddev[0] / R) - 1.0));
            
            // based on threshold, check if binary image should get 255 or 0 at that pixel
            dst.at<uchar>(i, j) = gray.at<uchar>(i, j) < global_mean[0] ? 255 : 0;
        }
    }
    return 0;
 
}

int cv_threshold(const cv::Mat &src, cv::Mat &dst){
    // input: BGR, BGRA, or grayscale image, CV_8UC3, CV_8UC4, or CV_8UC1
    // output: binary image, CV_8UC1
    if (src.empty()) {
        return -1;
    }

    cv::Mat gray;
    if (src.channels() == 3) {
        cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
    } else if (src.channels() == 4) {
        cv::cvtColor(src, gray, cv::COLOR_BGRA2GRAY);
    } else {
        gray = src;
    }

    if (gray.depth() != CV_8U) {
        cv::normalize(gray, gray, 0, 255, cv::NORM_MINMAX);
        gray.convertTo(gray, CV_8U);
    }

    cv::Mat blurred;
    cv::GaussianBlur(gray, blurred, cv::Size(5, 5), 0);

    cv::Mat binary;
    double otsu_threshold = cv::threshold(
        blurred,
        binary,
        0,
        255,
        cv::THRESH_BINARY | cv::THRESH_OTSU
    );

    // Treat the image border as background so foreground objects become black (reversed colors), foreground objects become black.
    cv::Mat border_mask = cv::Mat::zeros(blurred.size(), CV_8UC1);
    int border_width = std::max(1, std::min(blurred.rows, blurred.cols) / 20);
    cv::rectangle(border_mask, cv::Rect(0, 0, blurred.cols, border_width), cv::Scalar(255), cv::FILLED);
    cv::rectangle(border_mask, cv::Rect(0, blurred.rows - border_width, blurred.cols, border_width), cv::Scalar(255), cv::FILLED);
    cv::rectangle(border_mask, cv::Rect(0, 0, border_width, blurred.rows), cv::Scalar(255), cv::FILLED);
    cv::rectangle(border_mask, cv::Rect(blurred.cols - border_width, 0, border_width, blurred.rows), cv::Scalar(255), cv::FILLED);

    double border_mean = cv::mean(blurred, border_mask)[0];
    if (border_mean > otsu_threshold) {
        dst = binary;
    } else {
        cv::bitwise_not(binary, dst);
    }

    return 0;
}

int clean_image(const cv::Mat &src, cv::Mat &dst) {
    // input: binary image, CV_8UC1
    // output: cleaned binary image, CV_8UC1

    // Uses morphological filtering
    // specifically shrinking/dilation to remove gaps/blotches
    
    cv::Mat kernel = cv::getStructuringElement(
        // 8-Connected
        cv::MORPH_RECT,
        cv::Size(3, 3)
    );

    // n dilates -> 2n erodes -> n dilates
    cv::Mat temp = src.clone();
    int num_passes = 2;
    for (int i = 0; i < num_passes; i++) {
        cv::dilate(temp, temp, kernel); 
    }
    for (int i = 0; i < num_passes; i++) {
        cv::erode(temp, temp, kernel); 
    }
    for (int i = 0; i < num_passes; i++) {
        cv::erode(temp, temp, kernel); 
    }
    for (int i = 0; i < num_passes; i++) {
        cv::dilate(temp, temp, kernel); 
    }

    dst = temp;
    // cv::morphologyEx(src, dst, cv::MORPH_OPEN, kernel);
    // cv::morphologyEx(dst, dst, cv::MORPH_CLOSE, kernel);
    
    return 0;
}

int segment_image(const cv::Mat &src, cv::Mat &dst, cv::Mat &labels, cv::Mat &stats) {
    // input: cleaned binary image, CV_8UC1
    // dst output: colored segmentation visualization, CV_8UC3
    // labels output: largest N region label image, CV_32S

    // Extracts Labels, Spatial Measurements, and Geometric Centers for every object
    cv::Mat all_labels; // maps each pixel to region ID
    // cv::Mat stats; // array that contains structural data for each label ID:
    // x/y coords of bounding box
    // horizontal/vertical size of bounding box 
    // total pixel count in bounding box
    cv::Mat centroids; // x/y coords for the label COM
    
    // segments non-zero pixels only
    int num_labels = cv::connectedComponentsWithStats(
        src,
        all_labels, // CV_32SC1
        stats,
        centroids,
        8,
        CV_32S
    );

    // remove segments that are too small 
    int N = 5; // select the largest N regions
    int min_area = 10*10; // total pixel count of label

    // Remove any regions whose label appears on the edge of the image
    std::set<int> edge_labels;

    // Top and bottom rows
    for (int col = 0; col < all_labels.cols; ++col) {
        int top_label = all_labels.at<int>(0, col);
        int bottom_label = all_labels.at<int>(all_labels.rows - 1, col);
        if (top_label != 0) edge_labels.insert(top_label);
        if (bottom_label != 0) edge_labels.insert(bottom_label);
    }
    // Left and right columns
    for (int row = 0; row < all_labels.rows; ++row) {
        int left_label = all_labels.at<int>(row, 0);
        int right_label = all_labels.at<int>(row, all_labels.cols - 1);
        if (left_label != 0) edge_labels.insert(left_label);
        if (right_label != 0) edge_labels.insert(right_label);
    }

    // Find the largest N non-background, non-edge regions by area
    // The area for each label is stored in stats.at<int>(label, cv::CC_STAT_AREA)

    std::vector<std::pair<int, int>> label_areas; // {label, area}
    for (int label = 1; label < num_labels; ++label) { // skip background (label 0)
        if (edge_labels.count(label) > 0) {
            continue;
        }

        int area = stats.at<int>(label, cv::CC_STAT_AREA);
        if (area >= min_area) {
            label_areas.push_back({label, area});
        }
        
    };
    // Sort in descending order by area ()
    std::sort(label_areas.begin(), label_areas.end(), [](const auto& a, const auto& b) {
        return a.second > b.second;
    });

    // Collect the label ids of the N largest regions ()
    std::set<int> largest_labels;
    for (int i = 0; i < std::min(N, (int)label_areas.size()); ++i) {
        largest_labels.insert(label_areas[i].first);
    }

    // This will be filled with stable display IDs after matching current regions
    // to previous-frame regions.
    labels = cv::Mat::zeros(all_labels.size(), CV_32S);

    // Match current regions to previous-frame centroids so colors stay stable.
    static std::vector<RegionTrack> previous_regions;
    static cv::RNG rng(12345);

    std::vector<RegionTrack> current_regions;
    for (const auto& label_area : label_areas) {
        int label = label_area.first;
        if (largest_labels.count(label) == 0) {
            continue;
        }

        current_regions.push_back({
            label,
            0,
            cv::Point2d(centroids.at<double>(label, 0), centroids.at<double>(label, 1)),
            cv::Vec3b(0, 0, 0)
        });
    }

    std::vector<cv::Vec3b> colors(num_labels, cv::Vec3b(0, 0, 0));
    std::vector<int> display_ids(num_labels, 0);
    std::vector<bool> previous_used(previous_regions.size(), false);
    std::set<int> used_display_ids;
    double max_match_distance = 0.15 * std::max(src.cols, src.rows);

    for (RegionTrack& region : current_regions) {
        int best_previous = -1;
        double best_distance = std::numeric_limits<double>::max();

        for (int i = 0; i < (int)previous_regions.size(); ++i) {
            if (previous_used[i]) {
                continue;
            }

            double distance = cv::norm(region.centroid - previous_regions[i].centroid);
            if (distance < best_distance) {
                best_distance = distance;
                best_previous = i;
            }
        }

        if (best_previous >= 0 && best_distance <= max_match_distance) {
            region.color = previous_regions[best_previous].color;
            region.display_id = previous_regions[best_previous].display_id;
            previous_used[best_previous] = true;
            used_display_ids.insert(region.display_id);
        }
    }

    for (RegionTrack& region : current_regions) {
        if (region.display_id == 0) {
            int display_id = 1;
            while (used_display_ids.count(display_id) > 0) {
                display_id++;
            }

            region.display_id = display_id;
            region.color = cv::Vec3b(rng.uniform(50, 255), rng.uniform(50, 255), rng.uniform(50, 255));
            used_display_ids.insert(display_id);
        }

        colors[region.label] = region.color;
        display_ids[region.label] = region.display_id;
    }

    previous_regions = current_regions;
    
    // Create color image for visualization
    cv::Mat color_map = cv::Mat::zeros(labels.size(), CV_8UC3);
    for (int i = 0; i < labels.rows; ++i) {
        for (int j = 0; j < labels.cols; ++j) {
            int label = all_labels.at<int>(i, j);
            if (label == 0 || largest_labels.count(label) == 0) {
                color_map.at<cv::Vec3b>(i, j) = cv::Vec3b(0,0,0); // BG or not top N
            } else {
                color_map.at<cv::Vec3b>(i, j) = colors[label];
                labels.at<int>(i, j) = display_ids[label];
            }
        }
    }
    dst = color_map;

    return 0;
}

int compute_features(const cv::Mat &labels, const cv::Mat &stats, cv::Mat &dst, std::vector<std::pair<int, std::vector<double>>> &feature_table) {
    // input: 
    // region label image, CV_32S
    // region stats,  
    // output: 
    // feature visualization image, CV_8UC3
    // feature data table,  

    // requireds oriented bounding box (provided by cv2.connectedComponentsWithStats?)
    // and axis of least central moment 

    // compute default moment for each region
    // M = x^p * y^q * f(x, y)
    // m00: foreground sum
    // m10: sum of x coords in region (global moment)
    // m01: sum of y coords in region (global moment)
    // m11: sum of x*y coords in region (global moment)
    // mu11: 
    // mu20: x central moment
    // mu02: y central moment
    // f(x, y): 1 if foreground/region, 0 if background/other region
    
    std::set<int> region_ids;

    // Collect all non-background region IDs in the segmentation image.
    for (int row = 0; row < labels.rows; ++row) {
        for (int col = 0; col < labels.cols; ++col) {
            int region_id = labels.at<int>(row, col);
            if (region_id != 0) {
                region_ids.insert(region_id);
            }
        }
    }
    
    for (int region_id : region_ids) {
        double m00 = 0.0;
        double m10 = 0.0;
        double m01 = 0.0;
        double m11 = 0.0;
        double m20 = 0.0;
        double m02 = 0.0;
        std::vector<cv::Point2d> pixels;
        int region_left = labels.cols;
        int region_top = labels.rows;
        for (int row = 0; row < labels.rows; ++row) {
            for (int col = 0; col < labels.cols; ++col) {
                int f = 0;
                if (labels.at<int>(row, col) == region_id) {
                    // add pixels to vector for use later in computing bounding box
                    pixels.emplace_back(col, row);
                    region_left = std::min(region_left, col);
                    region_top = std::min(region_top, row);
                    f = 1;
                }
                // int pixel_region_id = labels.at<int>(row, col);
                // int f = pixel_region_id != 0 && pixel_region_id == region_id ? 1 : 0;
                m00 += f;
                m10 += col * f;
                m01 += row * f;
                m11 += col * row * f;
                m20 += col * col * f;
                m02 += row * row * f;
            }
        }
        double mean_x = m10/m00;
        double mean_y = m01/m00;

        // central moments
        double mu11 = (m11 / m00) - (mean_x * mean_y);
        double mu02 = (m02 / m00) - (mean_y * mean_y);
        double mu20 = (m20 / m00) - (mean_x * mean_x);

        // angle of least central moment (dir of primary eigenvector)
        double alpha = 0.5 * atan2((2 * mu11), (mu20 - mu02)); // atan2 takes (num, denom) args
        
        // compute second order moment about axis of least central moment
        // double beta = alpha + math.pi/2
        // double mu22 = 

        // build eigenvectors
        std::vector<double> e1 = {cos(alpha), sin(alpha)};
        std::vector<double> e2 = {-sin(alpha), cos(alpha)};


        // To compute the smallest aligned bounding box for a region, project all pixels of the region
        // onto the primary eigenvector (orientation) and its perpendicular.
        // Then find the min/max of the projected coordinates.

        // Collect all pixel coordinates for the current region
        // std::vector<cv::Point2d> pixels;
        // for (int row = 0; row < src.rows; ++row) {
        //     for (int col = 0; col < src.cols; ++col) {
        //         if (src.at<int>(row, col) == region_id) {
        //             pixels.emplace_back(col, row);
        //         }
        //     }
        // }

        // Project each pixel onto the axes aligned with the orientation (alpha)
        // e1 = (cos(alpha), sin(alpha)), e2 = (-sin(alpha), cos(alpha))
        double cos_a = cos(alpha);
        double sin_a = sin(alpha);

        double min_proj1 = std::numeric_limits<double>::max(), max_proj1 = -std::numeric_limits<double>::max();
        double min_proj2 = std::numeric_limits<double>::max(), max_proj2 = -std::numeric_limits<double>::max();

        for (const auto& pt : pixels) {
            // Shift to centroid
            double x_shifted = pt.x - mean_x;
            double y_shifted = pt.y - mean_y;

            // Projection onto eigenvectors
            double proj1 =  x_shifted * cos_a + y_shifted * sin_a; // projection on e1
            double proj2 = -x_shifted * sin_a + y_shifted * cos_a; // projection on e2
            if (proj1 < min_proj1) min_proj1 = proj1;
            if (proj1 > max_proj1) max_proj1 = proj1;
            if (proj2 < min_proj2) min_proj2 = proj2;
            if (proj2 > max_proj2) max_proj2 = proj2;
        }

        // Now, the four corners of the bounding box (in image coordinates) are:
        // (min_proj1, min_proj2), (min_proj1, max_proj2),
        // (max_proj1, min_proj2), (max_proj1, max_proj2)
        // Convert these back to image coordinates:
        std::vector<cv::Point2d> box_corners;
        box_corners.push_back({mean_x + min_proj1 * cos_a - min_proj2 * sin_a, mean_y + min_proj1 * sin_a + min_proj2 * cos_a});
        box_corners.push_back({mean_x + max_proj1 * cos_a - min_proj2 * sin_a, mean_y + max_proj1 * sin_a + min_proj2 * cos_a});
        box_corners.push_back({mean_x + max_proj1 * cos_a - max_proj2 * sin_a, mean_y + max_proj1 * sin_a + max_proj2 * cos_a});
        box_corners.push_back({mean_x + min_proj1 * cos_a - max_proj2 * sin_a, mean_y + min_proj1 * sin_a + max_proj2 * cos_a});

        // The corners in `box_corners` are the minimal bounding box aligned with the region orientation.
    
        // add bounding box overlay (lime green) to the output image

        // Draw bounding box with lines between the four box corners
        // Lime green color (BGR): (0, 255, 128)
        cv::Scalar limeGreen(0, 255, 128);
        int thickness = 2;

        for (int i = 0; i < 4; ++i) {
            cv::Point pt1(cvRound(box_corners[i].x), cvRound(box_corners[i].y));
            cv::Point pt2(cvRound(box_corners[(i+1)%4].x), cvRound(box_corners[(i+1)%4].y));
            cv::line(dst, pt1, pt2, limeGreen, thickness);            
        }
        // draw center and eigenvectors if desired
        cv::Scalar red(0, 0, 255);
        cv::Scalar blue (255, 0, 0);
        cv::Point center(cvRound(mean_x), cvRound(mean_y));

        cv::circle(dst, center, 5, red, thickness);
        // Draw major axis (red) from center to bounding box limit in the major axis direction
        cv::Point2d major_axis_end1(
            mean_x + max_proj1 * cos_a,
            mean_y + max_proj1 * sin_a
        );
        cv::Point2d major_axis_end2(
            mean_x + min_proj1 * cos_a,
            mean_y + min_proj1 * sin_a
        );

        cv::line(dst, cv::Point(cvRound(major_axis_end1.x), cvRound(major_axis_end1.y)), 
                      cv::Point(cvRound(major_axis_end2.x), cvRound(major_axis_end2.y)), red, thickness);

        // Draw minor axis (blue) from center to bounding box limit in the minor axis direction
        cv::Point2d minor_axis_end1(
            mean_x - min_proj2 * sin_a,
            mean_y + min_proj2 * cos_a
        );
        cv::Point2d minor_axis_end2(
            mean_x - max_proj2 * sin_a,
            mean_y + max_proj2 * cos_a
        );
        cv::line(dst, cv::Point(cvRound(minor_axis_end1.x), cvRound(minor_axis_end1.y)),
                      cv::Point(cvRound(minor_axis_end2.x), cvRound(minor_axis_end2.y)), blue, thickness);


        // compute features
        // need to attatch the feature vector to the region ID somehow
        // std::vector<std::pair<int, std::vector<double>>> feature_table; // {region_id, features}
        std::vector<double> features;

        // bounding box aspect-ratio
        // For a rectangle in OpenCV, width = distance from pt0 to pt1, height = distance from pt1 to pt2
        double width = cv::norm(box_corners[1] - box_corners[0]);
        double height = cv::norm(box_corners[2] - box_corners[1]);
        // take the smaller of the two dimensions for the numerator (esentially normalization)
        double larger_dim = std::max(width, height);
        double smaller_dim = std::min(width, height);
        double aspect_ratio = smaller_dim / larger_dim;
        features.push_back(aspect_ratio);

        // percentage filled
        int area = static_cast<int>(m00); // pixel count in region
        double percent_filled = area / (width * height);
        features.push_back(percent_filled);

        // (ADD MORE FEATURES HERE)


        // add single feature vector to feature vector table
        feature_table.push_back({region_id, features});

        // print region ID near the top-left corner of the region
        cv::Point region_id_org(region_left, std::max(region_top - 5, 15));
        std::string region_id_text = std::to_string(region_id);
        int font = cv::FONT_HERSHEY_SIMPLEX;
        double font_scale = 0.5;
        int thickness_text = 1;
        cv::putText(dst, region_id_text, region_id_org, font, font_scale, cv::Scalar(0, 0, 0), thickness_text + 2);
        cv::putText(dst, region_id_text, region_id_org, font, font_scale, cv::Scalar(255, 255, 255), thickness_text);

        // print percentage filled next to bounding box lower right corner
        char text[50];
        snprintf(text, sizeof(text), "Filled: %.2f%%", percent_filled * 100.0);
        // Bottom-right corner of bounding box is box_corners[2]
        cv::Point text_org(cvRound(box_corners[2].x), cvRound(box_corners[2].y));
        cv::putText(dst, text, text_org, font, font_scale, cv::Scalar(0,255,255), thickness_text);
        

    }
    return 0;
}

int compute_features_resnet(const cv::Mat &frame, const cv::Mat &labels, const cv::Mat &stats, cv::Mat &dst, std::vector<std::pair<int, std::vector<double>>> &feature_table) {
    // input: 
    // region label image, CV_32S
    // region stats,  
    // output: 
    // feature visualization image, CV_8UC3
    // feature data table,  

    // requireds oriented bounding box (provided by cv2.connectedComponentsWithStats?)
    // and axis of least central moment 

    // compute default moment for each region
    // M = x^p * y^q * f(x, y)
    // m00: foreground sum
    // m10: sum of x coords in region (global moment)
    // m01: sum of y coords in region (global moment)
    // m11: sum of x*y coords in region (global moment)
    // mu11: 
    // mu20: x central moment
    // mu02: y central moment
    // f(x, y): 1 if foreground/region, 0 if background/other region
    
    std::set<int> region_ids;

    // Load once, not inside the region loop:
    cv::dnn::Net resnet_net = cv::dnn::readNetFromONNX("../src/resnet18-v2-7.onnx");

    // Collect all non-background region IDs in the segmentation image.
    for (int row = 0; row < labels.rows; ++row) {
        for (int col = 0; col < labels.cols; ++col) {
            int region_id = labels.at<int>(row, col);
            if (region_id != 0) {
                region_ids.insert(region_id);
            }
        }
    }
    
    for (int region_id : region_ids) {
        double m00 = 0.0;
        double m10 = 0.0;
        double m01 = 0.0;
        double m11 = 0.0;
        double m20 = 0.0;
        double m02 = 0.0;
        std::vector<cv::Point2d> pixels;
        int region_left = labels.cols;
        int region_top = labels.rows;
        for (int row = 0; row < labels.rows; ++row) {
            for (int col = 0; col < labels.cols; ++col) {
                int f = 0;
                if (labels.at<int>(row, col) == region_id) {
                    // add pixels to vector for use later in computing bounding box
                    pixels.emplace_back(col, row);
                    region_left = std::min(region_left, col);
                    region_top = std::min(region_top, row);
                    f = 1;
                }
                // int pixel_region_id = labels.at<int>(row, col);
                // int f = pixel_region_id != 0 && pixel_region_id == region_id ? 1 : 0;
                m00 += f;
                m10 += col * f;
                m01 += row * f;
                m11 += col * row * f;
                m20 += col * col * f;
                m02 += row * row * f;
            }
        }
        double mean_x = m10/m00;
        double mean_y = m01/m00;

        // central moments
        double mu11 = (m11 / m00) - (mean_x * mean_y);
        double mu02 = (m02 / m00) - (mean_y * mean_y);
        double mu20 = (m20 / m00) - (mean_x * mean_x);

        // angle of least central moment (dir of primary eigenvector)
        double alpha = 0.5 * atan2((2 * mu11), (mu20 - mu02)); // atan2 takes (num, denom) args
        
        // compute second order moment about axis of least central moment
        // double beta = alpha + math.pi/2
        // double mu22 = 

        // build eigenvectors
        std::vector<double> e1 = {cos(alpha), sin(alpha)};
        std::vector<double> e2 = {-sin(alpha), cos(alpha)};


        // To compute the smallest aligned bounding box for a region, project all pixels of the region
        // onto the primary eigenvector (orientation) and its perpendicular.
        // Then find the min/max of the projected coordinates.

        // Collect all pixel coordinates for the current region
        // std::vector<cv::Point2d> pixels;
        // for (int row = 0; row < src.rows; ++row) {
        //     for (int col = 0; col < src.cols; ++col) {
        //         if (src.at<int>(row, col) == region_id) {
        //             pixels.emplace_back(col, row);
        //         }
        //     }
        // }

        // Project each pixel onto the axes aligned with the orientation (alpha)
        // e1 = (cos(alpha), sin(alpha)), e2 = (-sin(alpha), cos(alpha))
        double cos_a = cos(alpha);
        double sin_a = sin(alpha);

        double min_proj1 = std::numeric_limits<double>::max(), max_proj1 = -std::numeric_limits<double>::max();
        double min_proj2 = std::numeric_limits<double>::max(), max_proj2 = -std::numeric_limits<double>::max();

        for (const auto& pt : pixels) {
            // Shift to centroid
            double x_shifted = pt.x - mean_x;
            double y_shifted = pt.y - mean_y;

            // Projection onto eigenvectors
            double proj1 =  x_shifted * cos_a + y_shifted * sin_a; // projection on e1
            double proj2 = -x_shifted * sin_a + y_shifted * cos_a; // projection on e2
            if (proj1 < min_proj1) min_proj1 = proj1;
            if (proj1 > max_proj1) max_proj1 = proj1;
            if (proj2 < min_proj2) min_proj2 = proj2;
            if (proj2 > max_proj2) max_proj2 = proj2;
        }

        // Now, the four corners of the bounding box (in image coordinates) are:
        // (min_proj1, min_proj2), (min_proj1, max_proj2),
        // (max_proj1, min_proj2), (max_proj1, max_proj2)
        // Convert these back to image coordinates:
        std::vector<cv::Point2d> box_corners;
        box_corners.push_back({mean_x + min_proj1 * cos_a - min_proj2 * sin_a, mean_y + min_proj1 * sin_a + min_proj2 * cos_a});
        box_corners.push_back({mean_x + max_proj1 * cos_a - min_proj2 * sin_a, mean_y + max_proj1 * sin_a + min_proj2 * cos_a});
        box_corners.push_back({mean_x + max_proj1 * cos_a - max_proj2 * sin_a, mean_y + max_proj1 * sin_a + max_proj2 * cos_a});
        box_corners.push_back({mean_x + min_proj1 * cos_a - max_proj2 * sin_a, mean_y + min_proj1 * sin_a + max_proj2 * cos_a});

        // The corners in `box_corners` are the minimal bounding box aligned with the region orientation.
    
        // add bounding box overlay (lime green) to the output image

        // Draw bounding box with lines between the four box corners
        // Lime green color (BGR): (0, 255, 128)
        cv::Scalar limeGreen(0, 255, 128);
        int thickness = 2;

        for (int i = 0; i < 4; ++i) {
            cv::Point pt1(cvRound(box_corners[i].x), cvRound(box_corners[i].y));
            cv::Point pt2(cvRound(box_corners[(i+1)%4].x), cvRound(box_corners[(i+1)%4].y));
            cv::line(dst, pt1, pt2, limeGreen, thickness);            
        }
        // draw center and eigenvectors if desired
        cv::Scalar red(0, 0, 255);
        cv::Scalar blue (255, 0, 0);
        cv::Point center(cvRound(mean_x), cvRound(mean_y));

        cv::circle(dst, center, 5, red, thickness);
        // Draw major axis (red) from center to bounding box limit in the major axis direction
        cv::Point2d major_axis_end1(
            mean_x + max_proj1 * cos_a,
            mean_y + max_proj1 * sin_a
        );
        cv::Point2d major_axis_end2(
            mean_x + min_proj1 * cos_a,
            mean_y + min_proj1 * sin_a
        );

        cv::line(dst, cv::Point(cvRound(major_axis_end1.x), cvRound(major_axis_end1.y)), 
                      cv::Point(cvRound(major_axis_end2.x), cvRound(major_axis_end2.y)), red, thickness);

        // Draw minor axis (blue) from center to bounding box limit in the minor axis direction
        cv::Point2d minor_axis_end1(
            mean_x - min_proj2 * sin_a,
            mean_y + min_proj2 * cos_a
        );
        cv::Point2d minor_axis_end2(
            mean_x - max_proj2 * sin_a,
            mean_y + max_proj2 * cos_a
        );
        cv::line(dst, cv::Point(cvRound(minor_axis_end1.x), cvRound(minor_axis_end1.y)),
                      cv::Point(cvRound(minor_axis_end2.x), cvRound(minor_axis_end2.y)), blue, thickness);



        // extract part of image inside bounding box from ORIGINAL COLOR IMAGE
        double width = cv::norm(box_corners[1] - box_corners[0]);
        double height = cv::norm(box_corners[2] - box_corners[1]);
        cv::Mat extracted_image = frame(cv::Rect(region_left, region_top, static_cast<int>(width), static_cast<int>(height)));

        // rotate extracted_image align with primary axis
        cv::Mat rotated_image;
        cv::Mat M = cv::getRotationMatrix2D(cv::Point2f(mean_x - region_left, mean_y - region_top), alpha * 180/M_PI, 1.0);
        cv::warpAffine(extracted_image, rotated_image, M, extracted_image.size());

        // reshape into 224x224 image
        cv::Mat reshaped_image;
        cv::resize(rotated_image, reshaped_image, cv::Size(224, 224));

        // compute resnet embedding
        cv::Mat embedding;
        cv::Mat blob;

        // Converts standard 2D images into 4D 'blobs' that neural networks can ingest. 
        // The function handles resizing, mean subtraction, channel scaling, and red-blue channel swapping
        cv::dnn::blobFromImage(
            reshaped_image,
            blob,
            (1.0 / 255.0) * (1.0 / 0.226),
            cv::Size(224, 224),
            cv::Scalar(124, 116, 104),
            true,
            false,
            CV_32F
        );

        resnet_net.setInput(blob);
        embedding = resnet_net.forward("resnetv22_flatten0_reshape0");

        std::vector<double> resnet_feature_vector;
        resnet_feature_vector.reserve(embedding.total());

        const float *data = embedding.ptr<float>();
        for (size_t i = 0; i < embedding.total(); i++) {
            resnet_feature_vector.push_back(static_cast<double>(data[i]));
        }

        feature_table.push_back({region_id, resnet_feature_vector}); 

        // print region ID near the top-left corner of the region
        cv::Point region_id_org(region_left, std::max(region_top - 5, 15));
        std::string region_id_text = std::to_string(region_id);
        int font = cv::FONT_HERSHEY_SIMPLEX;
        double font_scale = 0.5;
        int thickness_text = 1;
        cv::putText(dst, region_id_text, region_id_org, font, font_scale, cv::Scalar(0, 0, 0), thickness_text + 2);
        cv::putText(dst, region_id_text, region_id_org, font, font_scale, cv::Scalar(255, 255, 255), thickness_text);

    }
    return 0;
}

void showConfusionMatrixGui(const std::vector<std::vector<int>> &confusion_matrix,
                            const std::vector<ObjectRecord> &database) {
    if (confusion_matrix.empty() || database.empty()) {
        std::cout << "Confusion matrix is empty.\n";
        return;
    }

    const int row_label_width = 150;
    const int column_width = 100;
    const int row_height = 40;
    const int header_height = 60;
    const int margin = 20;

    int image_width = margin * 2 + row_label_width + column_width * static_cast<int>(database.size());
    int image_height = margin * 2 + header_height + row_height * static_cast<int>(database.size());
    cv::Mat matrix_image(image_height, image_width, CV_8UC3, cv::Scalar(255, 255, 255));

    int table_left = margin + row_label_width;
    int table_top = margin + header_height;
    int table_right = table_left + column_width * static_cast<int>(database.size());
    int table_bottom = table_top + row_height * static_cast<int>(database.size());

    cv::putText(matrix_image, "Confusion Matrix", cv::Point(margin, 30),
                cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 0, 0), 2);
    cv::putText(matrix_image, "Actual / Predicted", cv::Point(margin, table_top - 15),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);

    for (int col = 0; col < static_cast<int>(database.size()); col++) {
        int x = table_left + col * column_width;
        cv::rectangle(matrix_image, cv::Rect(x, table_top - row_height, column_width, row_height),
                      cv::Scalar(230, 230, 230), cv::FILLED);
        cv::rectangle(matrix_image, cv::Rect(x, table_top - row_height, column_width, row_height),
                      cv::Scalar(0, 0, 0), 1);
        cv::putText(matrix_image, database[col].name,
                    cv::Point(x + 8, table_top - 14), cv::FONT_HERSHEY_SIMPLEX,
                    0.5, cv::Scalar(0, 0, 0), 1);
    }

    for (int row = 0; row < static_cast<int>(database.size()); row++) {
        int y = table_top + row * row_height;
        cv::rectangle(matrix_image, cv::Rect(margin, y, row_label_width, row_height),
                      cv::Scalar(230, 230, 230), cv::FILLED);
        cv::rectangle(matrix_image, cv::Rect(margin, y, row_label_width, row_height),
                      cv::Scalar(0, 0, 0), 1);
        cv::putText(matrix_image, database[row].name, cv::Point(margin + 8, y + 25),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);

        for (int col = 0; col < static_cast<int>(database.size()); col++) {
            int x = table_left + col * column_width;
            cv::rectangle(matrix_image, cv::Rect(x, y, column_width, row_height),
                          cv::Scalar(0, 0, 0), 1);
            cv::putText(matrix_image, std::to_string(confusion_matrix[row][col]),
                        cv::Point(x + column_width / 2 - 8, y + 25),
                        cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 255), 2);
        }
    }

    cv::rectangle(matrix_image, cv::Point(table_left, table_top),
                  cv::Point(table_right, table_bottom), cv::Scalar(0, 0, 0), 1);
    cv::imshow("Confusion Matrix", matrix_image);
    cv::waitKey(0);
    cv::destroyWindow("Confusion Matrix");
}

int promptCorrectObjectIndex(const std::vector<ObjectRecord>& database) {
    std::cout << "Known objects:";
    for (const auto& record : database) {
        std::cout << " " << record.name;
    }
    std::cout << std::endl;

    while (true) {
        std::string actual_name;
        std::cout << "Enter correct object name (blank to skip): ";
        std::getline(std::cin, actual_name);
        if (actual_name.empty()) {
            return -1;
        }

        int actual_idx = findObjectIndexByName(database, actual_name);
        if (actual_idx >= 0) {
            return actual_idx;
        }

        std::cout << "Object '" << actual_name << "' is not in the database.\n";
    }
}

int evaluate(const std::vector<std::pair<int, std::vector<double>>> &feature_table) {
    std::string database_path = "../src/object_db.csv";
    std::string confusion_matrix_path = "../src/confusion_matrix.csv";
    std::vector<ObjectRecord> database = loadObjectDatabase(database_path);
    if (database.empty()) {
        std::cerr << "Database is empty. Add at least one object first.\n";
        return -1;
    }

    std::vector<std::vector<int>> confusion_matrix = loadConfusionMatrix(confusion_matrix_path, database);
    std::vector<double> std_vector = computeFeatureStdVector(database);

    if (feature_table.empty()) {
        std::cout << "No detected regions to evaluate.\n";
        showConfusionMatrixGui(confusion_matrix, database);
        return 0;
    }

    for (int i = 0; i < static_cast<int>(feature_table.size()); i++) {
        std::vector<std::pair<std::string, double>> name_dist_pairs =
            rankObjectMatches(feature_table[i].second, database, std_vector);
        if (name_dist_pairs.empty()) {
            continue;
        }

        const auto& top_match = name_dist_pairs[0];
        int predicted_idx = findObjectIndexByName(database, top_match.first);
        if (predicted_idx < 0) {
            continue;
        }

        std::cout << "Region " << feature_table[i].first << " top match: "
                  << top_match.first << " (distance: " << top_match.second << ")\n";
        std::string response;
        std::cout << "Was this match correct? (y/n, blank to skip): ";
        std::getline(std::cin, response);

        if (response.empty()) {
            continue;
        }

        int actual_idx = -1;
        if (response[0] == 'y' || response[0] == 'Y') {
            actual_idx = predicted_idx;
        } else if (response[0] == 'n' || response[0] == 'N') {
            actual_idx = promptCorrectObjectIndex(database);
        } else {
            std::cout << "Response not recognized. Skipping this detection.\n";
            continue;
        }

        if (actual_idx >= 0) {
            confusion_matrix[actual_idx][predicted_idx]++;
            saveConfusionMatrix(confusion_matrix_path, confusion_matrix, database);
        }
    }

    showConfusionMatrixGui(confusion_matrix, database);
    return 0;
}

int train(const std::vector<std::pair<int, std::vector<double>>> &feature_table) {
    // Input: 
    // feature visualization image, CV_8UC3
    // feature data table

    // Output:
    // n/a 

    // Updates database with new objects. Database includes name of object, feature vector, 

    // Compare feature vector for regions in feature image, if a match exists in the database, skip
    // If no match exists, if the user clicks inside the bounding box, prompt the user to enter the name of the objec
    // and save the object, feature vector pair to the database
    std::string database_path = "../src/object_db.csv";
    std::vector<ObjectRecord> database = loadObjectDatabase(database_path);
    if (database.empty()) {
        std::cerr << "Database is empty. Add at least one object first.\n";
        return -1;
    }

    // compute the std for all collumns in database
    std::vector<double> std_vector = computeFeatureStdVector(database);

    for (int i = 0; i < feature_table.size(); i++){
        std::vector<double> feature_vector_img = feature_table[i].second; // grab the second part of the pair

        std::vector<std::pair<std::string, double>> name_dist_pairs = rankObjectMatches(feature_vector_img, database, std_vector);

        // rank the top N matches (sort by euclidean_dist)
        int N = 5;
        // Print the top N matches for this region
        std::cout << "Top " << N << " matches for region " << feature_table[i].first << ":\n";
        for (int match_idx = 0; match_idx < std::min(N, (int)name_dist_pairs.size()); ++match_idx) {
            std::cout << "  " << name_dist_pairs[match_idx].first 
                      << " (distance: " << name_dist_pairs[match_idx].second << ")\n";
        }

        // EXTENSION: determine if top match meets threshold to be new object
        double threshold = 0.1; 
        // Find the top match
        if (!name_dist_pairs.empty()) {
            const auto& top_match = name_dist_pairs[0];  

            if (top_match.second > threshold) {
                std::cout << "Region " << feature_table[i].first << ": Unrecognized object detected (distance: " << top_match.second << ").\n";
                std::string user_input;
                do {
                    std::cout << "Please enter the name for the new object (or leave blank to skip): ";
                    std::getline(std::cin, user_input);

                    if (!user_input.empty()) {
                        // Check if user_input is already in the database
                        auto existing = std::find_if(database.begin(), database.end(),
                            [&user_input](const ObjectRecord& record) {
                                return record.name == user_input;
                            });
                        if (existing != database.end()) {
                            std::cout << "Object already exists in database. Overwriting.\n";
                            existing->features = feature_vector_img;
                            if (saveObjectDatabase(database_path, database)) {
                                std::cout << "Updated object '" << user_input << "' in database.\n";
                            }
                            break;
                        }

                        // Save the name and feature vector into new row in database
                        std::ofstream dbfile(database_path, std::ios::app);
                        if (dbfile.is_open()) {
                            database.push_back({user_input, feature_vector_img});
                            dbfile << user_input;
                            for (double feature : feature_vector_img) {
                                dbfile << "," << feature;
                            }
                            dbfile << std::endl;
                            dbfile.close();
                            std::cout << "Added object '" << user_input << "' to database.\n";
                        } else {
                            std::cerr << "Could not open object_db.csv to add new object.\n";
                        }
                        break; // move to the next object
                    } else {
                        // If input is blank, wait and ask again (do not move to next object until user enters input or explicitly skips)
                        std::cout << "No name entered. Press Enter to skip or enter a name to add." << std::endl;
                        // If user presses Enter again (keeps empty), break to skip
                        if (user_input.empty()) {
                            break;
                        }
                    }
                } while (true);
            }
            else {
                std::cout << "Recognized object: " << top_match.first << " (distance: " << top_match.second << ")\n";
            }
        }
    }  
    return 0;
}

int train_resnet(const std::vector<std::pair<int, std::vector<double>>> &feature_table) {
    // Input: 
    // feature visualization image, CV_8UC3
    // feature data table

    // Output:
    // n/a 

    // Updates database with new objects. Database includes name of object, feature vector, 

    // Compare feature vector for regions in feature image, if a match exists in the database, skip
    // If no match exists, if the user clicks inside the bounding box, prompt the user to enter the name of the objec
    // and save the object, feature vector pair to the database
    std::string database_path = "../src/resnet_object_db.csv";
    std::vector<ObjectRecord> database = loadObjectDatabase(database_path);
    if (database.empty()) {
        std::cerr << "Database is empty. Add at least one object first.\n";
        return -1;
    }

    // compute the std for all collumns in database
    std::vector<double> std_vector = computeFeatureStdVector(database);

    for (int i = 0; i < feature_table.size(); i++){
        std::vector<double> feature_vector_img = feature_table[i].second; // grab the second part of the pair

        std::vector<std::pair<std::string, double>> name_dist_pairs = rankObjectMatches(feature_vector_img, database, std_vector);

        // rank the top N matches (sort by euclidean_dist)
        int N = 5;
        // Print the top N matches for this region
        std::cout << "Top " << N << " matches for region " << feature_table[i].first << ":\n";
        for (int match_idx = 0; match_idx < std::min(N, (int)name_dist_pairs.size()); ++match_idx) {
            std::cout << "  " << name_dist_pairs[match_idx].first 
                      << " (distance: " << name_dist_pairs[match_idx].second << ")\n";
        }

        // EXTENSION: determine if top match meets threshold to be new object
        double threshold = 1000000; 
        // Find the top match
        if (!name_dist_pairs.empty()) {
            const auto& top_match = name_dist_pairs[0];  

            if (top_match.second > threshold) {
                std::cout << "Region " << feature_table[i].first << ": Unrecognized object detected (distance: " << top_match.second << ").\n";
                std::string user_input;
                do {
                    std::cout << "Please enter the name for the new object (or leave blank to skip): ";
                    std::getline(std::cin, user_input);

                    if (!user_input.empty()) {
                        // Check if user_input is already in the database
                        auto existing = std::find_if(database.begin(), database.end(),
                            [&user_input](const ObjectRecord& record) {
                                return record.name == user_input;
                            });
                        if (existing != database.end()) {
                            std::cout << "Object already exists in database. Overwriting.\n";
                            existing->features = feature_vector_img;
                            if (saveObjectDatabase(database_path, database)) {
                                std::cout << "Updated object '" << user_input << "' in database.\n";
                            }
                            break;
                        }

                        // Save the name and feature vector into new row in database
                        std::ofstream dbfile(database_path, std::ios::app);
                        if (dbfile.is_open()) {
                            database.push_back({user_input, feature_vector_img});
                            dbfile << user_input;
                            for (double feature : feature_vector_img) {
                                dbfile << "," << feature;
                            }
                            dbfile << std::endl;
                            dbfile.close();
                            std::cout << "Added object '" << user_input << "' to database.\n";
                        } else {
                            std::cerr << "Could not open object_db.csv to add new object.\n";
                        }
                        break; // move to the next object
                    } else {
                        // If input is blank, wait and ask again (do not move to next object until user enters input or explicitly skips)
                        std::cout << "No name entered. Press Enter to skip or enter a name to add." << std::endl;
                        // If user presses Enter again (keeps empty), break to skip
                        if (user_input.empty()) {
                            break;
                        }
                    }
                } while (true);
            }
            else {
                std::cout << "Recognized object: " << top_match.first << " (distance: " << top_match.second << ")\n";
            }
        }
    }  
    return 0;
}
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
    cv::namedWindow("Blurred", 1);
    cv::namedWindow("Binary", 1);
    cv::namedWindow("Cleaned", 1);
    cv::namedWindow("Segmented", 1);
    cv::namedWindow("Features", 1);

    cv::Mat frame;

    for (;;) {
        *capdev >> frame;
        if( frame.empty()) {
            printf("frame is empty\n");
        }

        cv::resize(frame, frame, cv::Size(640, 480), 0, 0, cv::INTER_AREA);

        // Preprocess video before thresholding
        // blur filter
        cv::Mat blurred;
        cv::GaussianBlur(frame, blurred, cv::Size(5, 5), 0);

        // Thresholding algorithm (turns image into binary image)
        cv::Mat binary;
        threshold(blurred, binary);

        //--- Cleanup the binary image
        cv::Mat cleaned;
        clean_image(binary, cleaned);

        //--- Segment the image
        cv::Mat segmented;
        cv::Mat labels;
        cv::Mat stats;
        segment_image(cleaned, segmented, labels, stats);

        cv::Mat feature_image;
        std::vector<std::pair<int, std::vector<double>>> feature_table;
        feature_image = segmented.clone(); // Make a copy to draw on
        compute_features(labels, stats, feature_image, feature_table);

        //--- display bounding boxes on top of colored region map
        
        //--- compute features for each major region
        // start with % filled and bounding box aspect ratio
        // Any features you use should be translation, scale, and rotation invariant such as moments around the central axis of rotation
        // display at least one feature on the real-time video output
        cv::imshow("Blurred", blurred);
        cv::imshow("Binary", binary);
        cv::imshow("Cleaned", cleaned);
        cv::imshow("Segmented", segmented);
        cv::imshow("Features", feature_image);

        char key = cv::waitKey(10);
        if (key == 'q') {
            break;
        }
        if (key == 'n') {
            train(feature_table);
            // won't get here until train finishes running and all unknown objects are named
        }
        if (key == 'e') {
            evaluate(feature_table);
        }
        if (key == 'r') {
            std::vector<std::pair<int, std::vector<double>>> resnet_feature_table;
            compute_features_resnet(frame, labels, stats, feature_image, resnet_feature_table);
            train_resnet(resnet_feature_table);
        }
    }
    delete capdev;
    return (0);
}