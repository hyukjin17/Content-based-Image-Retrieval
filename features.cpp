/*
  Hyuk Jin Chung
  2/5/26

  Creates various feature vectors for image processing
*/

#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <fstream>
#include "opencv2/opencv.hpp"
#include "csv_util.h"

int extract_features(char *image_filename, cv::Mat &src, char *csv, int reset_file) {
    // find the center of image
    int cx = src.cols / 2;
    int cy = src.rows / 2;
    // define a 7x7 square in the center as the feature
    cv::Rect centerSquare(cx - 3, cy - 3, 7, 7);
    cv::Mat feature = src(centerSquare);
    std::vector<float> featVec;

    // convert into a feature vector
    for (int i = 0; i < feature.rows; i++) {
        cv::Vec3b *ptr = feature.ptr<cv::Vec3b>(i);
        for (int j = 0; j < feature.cols; j++) {
            for (int k = 0; k < 3; k++) {
                featVec.push_back((float) ptr[j][k]);
            }
        }
    }

    append_image_data_csv(csv, image_filename, featVec, reset_file);

    return (0);
}