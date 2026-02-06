/*
    Hyuk Jin Chung
    2/5/2026
    Function signatures for feature extraction functions
*/

#ifndef FEATURES_H
#define FEATURES_H

// Using the 7x7 square in the middle of the image, builds a feature vector of RGB colors (7x7 image x 3 channels)
// Args: src     - cv::Mat image
//       featVec - feature vector to be filled
void extract_baseline_features(cv::Mat &src, std::vector<float> &featVec);

void extract_histogram_features(cv::Mat &src, std::vector<float> &featVec);

#endif