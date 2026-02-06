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

// Calculates the histogram intersection distance betweeen the 2 vectors (normalized histogram)
// Returns a float: 1 - sum of min(a[i], b[i])
float hist_intersection(std::vector<float> &featVec, std::vector<float> &data)
{
    float sum = 0;

    for (int i = 0; i < featVec.size(); i++)
    {
        sum += std::min(featVec[i], data[i]);
    }

    return 1.0f - sum;
}

// Calculates the sum squared distance betweeen the 2 vectors
// Returns a float: sum of (a[i] - b[i])^2
float ssd(std::vector<float> &featVec, std::vector<float> &data)
{
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
float apply_metric(int metric, std::vector<float> &featVec, std::vector<float> &data)
{
    float dist;

    switch (metric)
    {
    case 1:
        dist = ssd(featVec, data);
        break;
    case 2:
        dist = hist_intersection(featVec, data);
        break;
    default:
        dist = ssd(featVec, data);
    }

    return dist;
}

/*
    Extracts the feature vectors from the csv database and compares every entry to the given feature vector
    Distance metric is chosen based on metric integer
    Prints out N closest matches

    Args:
        - csv: csv database filename
        - featVec: feature vector to be filled
        - metric: int value corresponding to a distance metric
        - N: number of closest matches to be returned
*/
void print_closest_match(char *csv, std::vector<float> &featVec, int metric, int N)
{
    std::vector<char *> filenames;
    std::vector<std::vector<float>> data;
    float distance;
    std::vector<std::pair<float, char *>> results;

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
    // sort the results by ascending distance
    std::sort(results.begin(), results.end());

    for (int i = 1; i < N + 1; i++)
    {
        // .second is the filename, .first is the distance
        printf("Image: %s (Dist: %.4f)\n", results[i].second, results[i].first);
    }
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
int set_feature_mode(char *feature_mode, char *csv, cv::Mat &src, std::vector<float> &featVec)
{
    int dist_metric;

    // find the csv filename based on the requested comparison method
    // and extract the feature vector from the image
    if (strcmp(feature_mode, "baseline") == 0)
    {
        strcpy(csv, "features_baseline.csv");
        extract_baseline_features(src, featVec);
        dist_metric = 1;
    }
    else if (strcmp(feature_mode, "hist") == 0)
    {
        strcpy(csv, "features_histogram.csv");
        extract_histogram_features(src, featVec);
        dist_metric = 2;
    }
    else if (strcmp(feature_mode, "hist2") == 0)
    {
        strcpy(csv, "features_histogram_rgb.csv");
        extract_histogram_rgb_features(src, featVec);
        dist_metric = 2;
    }
    else
    {
        printf("Invalid comparison method\nPlease use one of: baseline, hist, hist2, multihist, texture, dnn\n");
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

    // read the image
    src = cv::imread(img_filepath);
    if (src.empty())
    {
        printf("Invalid image filepath\n");
        exit(-1);
    }

    // extracts the feature vector from the image and returns an integer value corresponding to the distance metric to be used
    int metric = set_feature_mode(feature_mode, csv, src, featVec);
    // compares the image to every image in the database and prints N closest matches
    print_closest_match(csv, featVec, metric, N);

    return (0);
}