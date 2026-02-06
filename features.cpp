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

// Creates a 2D normalized rg chromaticity histogram from the src image (with 16 bins per color channel)
// Builds a feature vector from the histogram (16x16 values)
// Args: src     - cv::Mat image
//       featVec - feature vector to be filled
void extract_histogram_features(cv::Mat &src, std::vector<float> &featVec)
{
    const int histsize = 16;

    cv::Mat hist = cv::Mat::zeros(cv::Size(histsize, histsize), CV_32FC1);

    // loop over all pixels
    for (int i = 0; i < src.rows; i++)
    {
        cv::Vec3b *ptr = src.ptr<cv::Vec3b>(i);
        for (int j = 0; j < src.cols; j++)
        {
            // get RGB values
            float B = ptr[j][0];
            float G = ptr[j][1];
            float R = ptr[j][2];

            // compute the rg chromaticity
            float divisor = R + G + B;
            divisor = divisor > 0.0 ? divisor : 1.0; // check for divide-by-zero error
            float r = R / divisor;
            float g = G / divisor;

            // compute the index for r and g
            // r and g are in [0, 1]
            int rindex = (int)(r * (histsize - 1) + 0.5);
            int gindex = (int)(g * (histsize - 1) + 0.5);

            // increment the histogram
            hist.at<float>(rindex, gindex)++;
        }
    }

    // normalize the histogram by the number of pixels
    hist /= (float)(src.rows * src.cols); // divides all elements of a cv::Mat by the number of pixels

    // convert the histogram image into a feature vector
    for (int i = 0; i < histsize; i++)
    {
        float *ptr = hist.ptr<float>(i);
        for (int j = 0; j < histsize; j++)
        {
            // flattens the 2D histogram image into a single vector containing floats
            featVec.push_back(ptr[j]);
        }
    }
}

// Creates a 3D normalized RGB histogram from the src image (with 8 bins per color channel)
// Builds a feature vector from the histogram (8x8x8 values)
// Args: src     - cv::Mat image
//       featVec - feature vector to be filled
void extract_histogram_rgb_features(cv::Mat &src, std::vector<float> &featVec)
{
    const int histsize = 8;
    int size[] = {histsize, histsize, histsize};

    cv::Mat hist = cv::Mat::zeros(3, size, CV_32FC1);

    // loop over all pixels
    for (int i = 0; i < src.rows; i++)
    {
        cv::Vec3b *ptr = src.ptr<cv::Vec3b>(i);
        for (int j = 0; j < src.cols; j++)
        {
            // get RGB values
            uchar B = ptr[j][0];
            uchar G = ptr[j][1];
            uchar R = ptr[j][2];

            // quantize RGB values into histogram bins
            int divisor = 256 / histsize;
            // compute the index for r, g, and b
            int rindex = R / divisor;
            int gindex = G / divisor;
            int bindex = B / divisor;

            // increment the histogram
            hist.at<float>(rindex, gindex, bindex)++;
        }
    }

    // normalize the histogram by the number of pixels
    hist /= (float)(src.rows * src.cols); // divides all elements of a cv::Mat by the number of pixels

    // convert the histogram image into a feature vector
    for (int i = 0; i < histsize; i++)
    {
        for (int j = 0; j < histsize; j++)
        {
            for (int k = 0; k < histsize; k++)
            {
                // flattens the 3D histogram image into a single vector containing floats
                featVec.push_back(hist.at<float>(i, j, k));
            }
        }
    }
}