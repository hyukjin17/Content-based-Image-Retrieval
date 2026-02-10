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

// Creates a 2D normalized rg chromaticity histogram from the src image (with 16 bins per color channel)
// Builds a feature vector from the histogram (16x16 values)
// Args: src     - cv::Mat image
//       featVec - feature vector to be filled
void extract_histogram_features(cv::Mat &src, std::vector<float> &featVec);

// Creates a 3D normalized RGB histogram from the src image (with 8 bins per color channel)
// Builds a feature vector from the histogram (8x8x8 values)
// Args: src     - cv::Mat image
//       featVec - feature vector to be filled
void extract_histogram_rgb_features(cv::Mat &src, std::vector<float> &featVec);

// Creates a 2D normalized hs chromaticity histogram from the src image (with 16 bins per color channel)
// Adds another histogram of just the center piece of the image
// Builds a feature vector from the histograms (16x16 x 2 histograms)
// Args: src     - cv::Mat image
//       featVec - feature vector to be filled
void extract_histogram_hsv_features(cv::Mat &src, std::vector<float> &featVec);

// Creates normalized histograms (RGB whole image and top and bottom rectangle histograms)
// from the src image (with 8 bins per color channel)
// Builds a feature vector from the histogram (8x8x8 x 2 histograms)
// Args: src     - cv::Mat image
//       featVec - feature vector to be filled
void extract_multihist_features(cv::Mat &src, std::vector<float> &featVec);

// Creates a 3D normalized RGB histogram from the src image (with 8 bins per color channel)
// Adds another 3D normalized RGB histogram for the top and bottom halves of the image
// Builds a feature vector from the histograms (8x8x8 x 3 histograms)
// Args: src     - cv::Mat image
//       featVec - feature vector to be filled
void extract_multihist_features(cv::Mat &src, std::vector<float> &featVec);

// Creates a 3D normalized RGB histogram from the src image (with 8 bins per color channel)
// Adds another 3D normalized RGB histogram for the sobel magnitude image
// Builds a feature vector from the histograms (8x8x8 x 2 histograms)
// Args: src     - cv::Mat image
//       featVec - feature vector to be filled
void extract_sobel_features(cv::Mat &src, std::vector<float> &featVec);

// Creates a 2D normalized hs chromaticity histogram from the src image (with 16 bins per color channel)
// Adds another histogram of just the center piece of the image or the face (if the image contains a face)
// Builds a feature vector from the histograms ((16x16 + 1 flag to indicate face presence) x 2 histograms)
// Args: src     - cv::Mat image
//       featVec - feature vector to be filled
void extract_face_features(cv::Mat &src, std::vector<float> &featVec);

// Append the DNN embeddings to the existing feature vector by matching the filenames
// Finds the feature vector with the same filename as the current image and appends its DNN embeddings to the vector
// Args: featVec   - feature vector to be filled
//       filename  - file name of the current image
//       filenames - vector of filenames of DNN embeddings
//       data      - vector of DNN embeddings for each image in the DB
void append_dnn_vector(std::vector<float> &featVec, char *filename,
                      std::vector<char *> &filenames, std::vector<std::vector<float>> &data);

#endif