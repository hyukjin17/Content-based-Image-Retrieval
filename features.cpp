/*
  Hyuk Jin Chung
  2/5/26

  Creates various feature vectors for image processing and saves the feature vectors into a csv file using csv_util.cpp
*/

#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <fstream>
#include "opencv2/opencv.hpp"
#include "csv_util.h"

// Using the 7x7 square in the middle of the image, builds a feature vector of RGB colors (7x7 image x 3 channels)
// Args: src     - cv::Mat image
//       featVec - feature vector to be filled
void extract_baseline_features(cv::Mat &src, std::vector<float> &featVec)
{
    // find the center of image
    int cx = src.cols / 2;
    int cy = src.rows / 2;
    // define a 7x7 square in the center as the feature
    cv::Rect centerSquare(cx - 3, cy - 3, 7, 7);
    cv::Mat feature = src(centerSquare);

    // convert the square image into a feature vector
    for (int i = 0; i < feature.rows; i++)
    {
        cv::Vec3b *ptr = feature.ptr<cv::Vec3b>(i);
        for (int j = 0; j < feature.cols; j++)
        {
            for (int k = 0; k < 3; k++)
            {
                // flattens the 2D image and 3 color channels into a single vector containing floats
                featVec.push_back((float)ptr[j][k]);
            }
        }
    }
}

// 
void extract_histogram_features(cv::Mat &src, std::vector<float> &featVec)
{    

}