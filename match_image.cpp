/*
  Hyuk Jin Chung
  2/5/26

  Compares the given image to the images in the database based on the feature vectors in csv form
  Identifies the closest N matches in order of similarity (based on the criteria)
*/

#include <cstdio>
#include <cstring>
#include <cstdlib>
#include "opencv2/opencv.hpp"
#include "features.hpp"
#include "csv_util.h"

// available distance metric types
enum MetricType
{
    SSD,
    INTERSECTION,
    MULTI_INTERSECTION,
    SOBEL_INTERSECTION,
    COSINE
};

// Calculates the cosine distance between the 2 feature vectors
// Returns a float: 1 - cos(theta) = 1 - (v1 dot v2)/(|v1||v2|)
//                            ^ theta is the angle between the 2 vectors
float cosine(std::vector<float> &featVec, std::vector<float> &data)
{
    if (featVec.size() != data.size())
    {
        printf("Error: Vector size mismatch! Image: %lu vs Database: %lu\n", featVec.size(), data.size());
        exit(-1);
    }

    float dot = 0.0f;         // dot product
    float featVec_mag = 0.0f; // magnitude of featVec
    float data_mag = 0.0f;    // magnitude of data

    for (int i = 0; i < featVec.size(); i++)
    {
        dot += featVec[i] * data[i];
        featVec_mag += featVec[i] * featVec[i];
        data_mag += data[i] * data[i];
    }

    featVec_mag = std::sqrt(featVec_mag);
    data_mag = std::sqrt(data_mag);

    return 1.0f - (dot / (featVec_mag * data_mag));
}

// Calculates the histogram intersection sum betweeen the 2 vectors (normalized histogram)
// Returns a float: sum of min(a[i], b[i])
float intersection(std::vector<float> &featVec, std::vector<float> &data)
{
    if (featVec.size() != data.size())
    {
        printf("Error: Vector size mismatch! Image: %lu vs Database: %lu\n", featVec.size(), data.size());
        exit(-1);
    }

    float sum = 0;

    for (int i = 0; i < featVec.size(); i++)
    {
        sum += std::min(featVec[i], data[i]);
    }

    return sum;
}

// Calculates the histogram intersection distance betweeen the 2 vectors (normalized histogram)
// Divides the sum by 2 to account for 2 histograms (whole and sobel magnitude histogram images)
// Returns a float: 1 - sum of min(a[i], b[i]) / 2
float sobel_intersection(std::vector<float> &featVec, std::vector<float> &data)
{
    float sum = intersection(featVec, data);
    return 1.0f - (sum / 2.0f);
}

// Calculates the histogram intersection distance betweeen the 2 vectors (normalized histogram)
// Divides the sum by 4 to account for 4 histograms (whole, top half, bottom half, center histogram images)
// Returns a float: 1 - sum of min(a[i], b[i]) / 4
float multihist_intersection(std::vector<float> &featVec, std::vector<float> &data)
{
    float sum = intersection(featVec, data);
    return 1.0f - (sum / 4.0f);
}

// Calculates the histogram intersection distance betweeen the 2 vectors (normalized histogram)
// Returns a float: 1 - sum of min(a[i], b[i])
float hist_intersection(std::vector<float> &featVec, std::vector<float> &data)
{
    float sum = intersection(featVec, data);
    return 1.0f - sum;
}

// Calculates the sum squared distance betweeen the 2 vectors
// Returns a float: sum of (a[i] - b[i])^2
float ssd(std::vector<float> &featVec, std::vector<float> &data)
{
    if (featVec.size() != data.size())
    {
        printf("Error: Vector size mismatch! Image: %lu vs Database: %lu\n", featVec.size(), data.size());
        exit(-1);
    }

    float dist = 0;
    float diff;

    for (int i = 0; i < featVec.size(); i++)
    {
        diff = featVec[i] - data[i];
        dist += diff * diff;
    }

    return dist;
}

// Applies the chosen distance metric to calculate the distance between 2 feature vectors
// Returns the distance as a float
float apply_metric(MetricType metric, std::vector<float> &featVec, std::vector<float> &data)
{
    float dist;

    switch (metric)
    {
    case SSD:
        dist = ssd(featVec, data);
        break;
    case INTERSECTION:
        dist = hist_intersection(featVec, data);
        break;
    case MULTI_INTERSECTION:
        dist = multihist_intersection(featVec, data);
        break;
    case SOBEL_INTERSECTION:
        dist = sobel_intersection(featVec, data);
        break;
    }

    return dist;
}

/*
    Extracts the feature vectors from the csv database and compares every entry to the given feature vector
    Distance metric is chosen based on metric integer
    Prints out N closest matches and opens those images

    Args:
        - csv: csv database filename
        - featVec: feature vector to be filled
        - img_filepath: image file path
        - metric: int value corresponding to a distance metric
        - N: number of closest matches to be returned
        - ascending: whether to sort the results in ascending or descending order (best or worst match)
*/
void print_closest_match(char *csv, std::vector<float> &featVec, char *img_filepath,
                         MetricType metric, int N, bool ascending = true)
{
    std::vector<char *> filenames;
    std::vector<std::vector<float>> data;
    float distance;
    std::vector<std::pair<float, char *>> results;
    char filepath[256];
    char *last_slash;
    int dir_len;
    char dir[256];
    cv::Mat temp;
    bool skip_first = false;

    if (read_image_data_csv(csv, filenames, data) != 0)
    {
        printf("Invalid image filepath\n");
        exit(-1);
    }

    for (int i = 0; i < filenames.size(); i++)
    {
        distance = apply_metric(metric, featVec, data[i]);
        results.push_back({distance, filenames[i]});
    }
    // sort the results
    if (ascending)
        std::sort(results.begin(), results.end());
    else
        std::sort(results.begin(), results.end(), std::greater<>());

    // grab directory name from the img_filepath
    last_slash = strrchr(img_filepath, '/'); // find the index of the last slash
    dir_len = last_slash - img_filepath + 1; // find the length of directory name
    strncpy(dir, img_filepath, dir_len);     // copy everything up to the last slash
    dir[dir_len] = '\0';                     // null terminate
    int move_window = 0;

    cv::imshow(img_filepath, cv::imread(img_filepath));
    cv::moveWindow(img_filepath, 0, 0);

    for (int i = 0; i < N + 1; i++)
    {
        // if first match was not skipped, only print N matches
        if (!skip_first && i == N)
            continue;

        // reconstruct the filepath for each image for viewing
        strcpy(filepath, dir);
        strcat(filepath, results[i].second);

        // skip first match if the image is identical to the given image
        if (strstr(img_filepath, results[i].second) != NULL)
        {
            skip_first = true;
            continue;
        }

        // .second is the filename, .first is the distance
        temp = cv::imread(filepath);
        cv::imshow(filepath, temp);
        // move the image windows to stagger them for easier viewing
        move_window += temp.cols / 2;
        cv::moveWindow(filepath, move_window, 0);
        printf("Image: %s (Dist: %.4f)\n", results[i].second, results[i].first);
    }

    // wait for any key press and close all windows
    printf("Press any key to close all windows\n");
    cv::waitKey(0);
    cv::destroyAllWindows();
}

/*
    Based on user defined comparison method, extracts the feature vector from the image
    and returns an integer value corresponding to a distance metric

    Args:
        - feature_mode: user defined comparison method as a string
        - csv: csv filename to be assigned based on the feature_mode
        - src: cv::Mat image used for feature extraction
        - featVec: feature vector to be filled
*/
MetricType set_feature_mode(char *feature_mode, char *csv, cv::Mat &src, std::vector<float> &featVec)
{
    MetricType dist_metric;

    // find the csv filename based on the requested comparison method
    // and extract the feature vector from the image
    if (strcmp(feature_mode, "baseline") == 0)
    {
        strcpy(csv, "features_baseline.csv");
        extract_baseline_features(src, featVec);
        dist_metric = SSD;
    }
    else if (strcmp(feature_mode, "hist") == 0)
    {
        strcpy(csv, "features_histogram.csv");
        extract_histogram_features(src, featVec);
        dist_metric = INTERSECTION;
    }
    else if (strcmp(feature_mode, "hist2") == 0)
    {
        strcpy(csv, "features_histogram_rgb.csv");
        extract_histogram_rgb_features(src, featVec);
        dist_metric = INTERSECTION;
    }
    else if (strcmp(feature_mode, "multihist") == 0)
    {
        strcpy(csv, "features_multihistogram.csv");
        extract_multihist_features(src, featVec);
        dist_metric = MULTI_INTERSECTION;
    }
    else if (strcmp(feature_mode, "sobel") == 0)
    {
        strcpy(csv, "features_sobel_magnitude.csv");
        extract_sobel_features(src, featVec);
        dist_metric = SOBEL_INTERSECTION;
    }
    else
    {
        printf("Invalid comparison method\n");
        printf("Please use one of: baseline, hist, hist2, multihist, sobel, texture, dnn\n");
        exit(-1);
    }
    return dist_metric;
}

/*
    Compares a given image to all images in the database based on a chosen metric
    Prints out N closest matches found in the database

    Argv:
        - img_filepath: filepath of image to be compared with
        - metric: metric used to compare images (baseline, histogram, multi-histogram, etc.)
        - N: number of closest matches to be printed
*/
int main(int argc, char *argv[])
{
    std::vector<float> featVec; // flattened feature vector
    char img_filepath[256];
    char feature_mode[256];
    char csv[256];
    int N;
    cv::Mat src;
    bool ascending = true;

    // check for sufficient arguments
    if (argc < 4)
    {
        printf("usage: %s <image filepath>, <comparison method>, <number of matches>\n", argv[0]);
        exit(-1);
    }

    // get the arguments
    strcpy(img_filepath, argv[1]);
    strcpy(feature_mode, argv[2]);
    N = atoi(argv[3]);
    if (argc == 5 && strcmp("bot", argv[4]) == 0)
        ascending = false;

    // read the image
    src = cv::imread(img_filepath);
    if (src.empty())
    {
        printf("Invalid image filepath\n");
        exit(-1);
    }

    // extracts the feature vector from the image and returns an integer value corresponding to the distance metric to be used
    MetricType metric = set_feature_mode(feature_mode, csv, src, featVec);
    // compares the image to every image in the database and prints N closest matches
    print_closest_match(csv, featVec, img_filepath, metric, N, ascending);

    return (0);
}