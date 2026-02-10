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
#include "faceDetect.h"

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
            int rindex = std::min((int)std::floor(r * histsize), histsize - 1);
            int gindex = std::min((int)std::floor(g * histsize), histsize - 1);

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

// Helper method for the extract_histogram_hsv_features function
// Creates a 2D normalized hs chromaticity histogram from the src image (with 16 bins per color channel)
// Builds a feature vector from the histogram (16x16 hs values + black bin + gray bin)
// Args: src     - cv::Mat hsv image
//       featVec - feature vector to be filled
void extract_hsv_features(cv::Mat &src, std::vector<float> &featVec)
{
    const int histsize = 16;

    float hist[histsize][histsize] = {0};
    float black_bin = 0;
    float gray_bin = 0;

    // loop over all pixels
    for (int i = 0; i < src.rows; i++)
    {
        cv::Vec3b *ptr = src.ptr<cv::Vec3b>(i);
        for (int j = 0; j < src.cols; j++)
        {
            // get HSV values
            float H = ptr[j][0];
            float S = ptr[j][1];
            float V = ptr[j][2];

            // normalize s and v values between 0 and 1
            float s_norm = S / 255.0f;
            float v_norm = V / 255.0f;

            // take out dark and pale pixels into separate bins to avoid confusing the histogram
            // instead of trying to assign hue value to a dark pixel, take it out and count them separately
            if (v_norm < 0.2f)
            {
                black_bin += 1.0f; // count dark pixels
            }
            else if (s_norm < 0.2f)
            {
                gray_bin += 1.0f; // count gray/white pixels
            }
            else
            {
                // actual valid color
                // compute the index for h and s
                int hindex = (int)(H / (180.0f / histsize));
                int sindex = (int)(S / (256.0f / histsize));
                // clamp values to (histsize - 1)
                if (hindex >= histsize)
                    hindex = histsize - 1;
                if (sindex >= histsize)
                    sindex = histsize - 1;
                // add the normalized saturation value to the histogram
                hist[hindex][sindex] += s_norm;
            }
        }
    }

    // calculate the total sum of all bins to be used for histogram normalization
    float total_weight = black_bin + gray_bin;
    for (int i = 0; i < histsize; i++)
    {
        for (int j = 0; j < histsize; j++)
        {
            total_weight += hist[i][j];
        }
    }

    // convert the histogram into a feature vector  of size 258 (16x16 histogram + black bin + gray bin)
    for (int i = 0; i < histsize; i++)
    {
        for (int j = 0; j < histsize; j++)
        {
            // flattens the 2D histogram into a single vector containing floats
            // normalize the values by the total
            featVec.push_back(hist[i][j] / total_weight);
        }
    }
    // append the black and gray bins to the end
    featVec.push_back(black_bin / total_weight);
    featVec.push_back(gray_bin / total_weight);
}

// Creates a 2D normalized hs chromaticity histogram from the src image (with 16 bins per color channel)
// Adds another histogram of just the center piece of the image
// Builds a feature vector from the histograms ((16x16+2) x 2 histograms)
// Args: src     - cv::Mat image
//       featVec - feature vector to be filled
void extract_histogram_hsv_features(cv::Mat &src, std::vector<float> &featVec)
{
    cv::Mat hsvImage;
    cv::cvtColor(src, hsvImage, cv::COLOR_BGR2HSV); // creates new HSV image
    extract_hsv_features(hsvImage, featVec);

    // // top half
    // cv::Rect topRect(0, 0, hsvImage.cols, hsvImage.rows / 2);
    // cv::Mat top = hsvImage(topRect);
    // extract_hsv_features(top, featVec);

    // // bottom half
    // cv::Rect botRect(0, hsvImage.rows / 2, hsvImage.cols, hsvImage.rows / 2);
    // cv::Mat bot = hsvImage(botRect);
    // extract_hsv_features(bot, featVec);

    // find the center of image
    int cx = hsvImage.cols / 2;
    int cy = hsvImage.rows / 2;
    // define a rectangle in the center as the feature (1/2 of image sidelength)
    cv::Rect centerRect(cx - hsvImage.cols / 4, cy - hsvImage.rows / 4, hsvImage.cols / 2, hsvImage.rows / 2);
    cv::Mat center = hsvImage(centerRect);
    extract_hsv_features(center, featVec);
}

// Creates a 2D normalized hs chromaticity histogram from the src image (with 16 bins per color channel)
// Adds another histogram of just the center piece of the image or the face (if the image contains a face)
// Builds a feature vector from the histograms ((16x16 + 1 flag to indicate face presence) x 2 histograms)
// Args: src     - cv::Mat image
//       featVec - feature vector to be filled
void extract_face_features(cv::Mat &src, std::vector<float> &featVec)
{
    cv::Mat hsvImage;
    cv::cvtColor(src, hsvImage, cv::COLOR_BGR2HSV); // creates new HSV image
    extract_hsv_features(hsvImage, featVec);

    cv::Mat gray;                // grayscale frame used for face detection
    std::vector<cv::Rect> faces; // used for face detection (vector of detected faces to be filled)
    cv::Rect face;
    // convert the image to grayscale
    cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY, 0);

    detectFaces(gray, faces); // find all faces in the image

    if (faces.size() > 0)
    {
        // only takes the first face found in the image
        face = faces[0];
        // makes sure the rectangle is strictly within image bounds
        face = face & cv::Rect(0, 0, hsvImage.cols, hsvImage.rows);
        cv::Mat face_img = hsvImage(face);
        extract_hsv_features(face_img, featVec);

        // set a flag in the feature vector to tag that the image contains a face
        featVec.push_back(1.0f);
    }
    else
    {
        // find the center of image
        int cx = hsvImage.cols / 2;
        int cy = hsvImage.rows / 2;
        // define a rectangle in the center as the feature (1/2 of image sidelength)
        cv::Rect centerRect(cx - hsvImage.cols / 4, cy - hsvImage.rows / 4, hsvImage.cols / 2, hsvImage.rows / 2);
        cv::Mat center = hsvImage(centerRect);
        extract_hsv_features(center, featVec);

        // set a flag in the feature vector to tag that the image does not contain a face
        featVec.push_back(0.0f);
    }
}

// Creates a 3D normalized RGB histogram from the src image (with 8 bins per color channel)
// Adds 3 more 3D normalized RGB histograms for the top and bottom halves and the center of the image
// Builds a feature vector from the histograms (8x8x8 x 4 histograms)
// Args: src     - cv::Mat image
//       featVec - feature vector to be filled
void extract_multihist_features(cv::Mat &src, std::vector<float> &featVec)
{
    // histogram of entire image
    extract_histogram_rgb_features(src, featVec);

    // top half
    cv::Rect topRect(0, 0, src.cols, src.rows / 2);
    cv::Mat top = src(topRect);
    extract_histogram_rgb_features(top, featVec);

    // bottom half
    cv::Rect botRect(0, src.rows / 2, src.cols, src.rows / 2);
    cv::Mat bot = src(botRect);
    extract_histogram_rgb_features(bot, featVec);

    // find the center of image
    int cx = src.cols / 2;
    int cy = src.rows / 2;
    // define a rectangle in the center as the feature (1/2 of image sidelength)
    cv::Rect centerRect(cx - src.cols / 4, cy - src.rows / 4, src.cols / 2, src.rows / 2);
    cv::Mat center = src(centerRect);
    extract_histogram_rgb_features(center, featVec);
}

// 3x3 Sobel X filter as separable 1x3 filters (detects vertical edges)
// Args: color src image     Return: 16-bit signed short dst image
int sobelX3x3(cv::Mat &src, cv::Mat &dst)
{
    static cv::Mat temp;
    // makes an intermediate temp matrix
    temp = cv::Mat::zeros(src.size(), CV_16SC3);

    // makes a dst matrix for the final image
    dst = cv::Mat::zeros(src.size(), CV_16SC3);

    // first pass (horizontal filter)
    for (int i = 0; i < dst.rows; i++)
    {
        cv::Vec3b *srcPtr = src.ptr<cv::Vec3b>(i);   // gets the row pointer from the src image
        cv::Vec3s *tempPtr = temp.ptr<cv::Vec3s>(i); // gets the row pointer from the temp image
        for (int j = 1; j < dst.cols - 1; j++)
        {
            // loop over RGB color channels
            for (int k = 0; k < 3; k++)
            {
                // sum of the horizontally neighboring pixel values from the original src image
                // multiply the left pixel by -1 and the right pixel by 1 (middle pixel is already 0)
                // filterX = {-1, 0, 1}
                tempPtr[j][k] = -1 * srcPtr[j - 1][k] + srcPtr[j + 1][k]; // update the temp image
            }
        }
    }

    // vertical part of Sobel X filter
    int filterY[3] = {1, 2, 1};
    // second pass (vertical filter) using the generated temp image
    for (int i = 1; i < dst.rows - 1; i++)
    {
        // gets the 3 row pointers from the temp image
        cv::Vec3s *p1 = temp.ptr<cv::Vec3s>(i - 1);
        cv::Vec3s *p2 = temp.ptr<cv::Vec3s>(i);
        cv::Vec3s *p3 = temp.ptr<cv::Vec3s>(i + 1);

        cv::Vec3s *dstPtr = dst.ptr<cv::Vec3s>(i);
        for (int j = 0; j < dst.cols; j++)
        {
            // loop over RGB color channels
            for (int k = 0; k < 3; k++)
            {
                // sum of the vertically neighboring pixel values from each of the 3 rows in the temp image (multiplied by filterY)
                // divide by the sum of values in the blur vector to normalize and multiply by 2 to make the edges brighter
                dstPtr[j][k] = (p1[j][k] * filterY[0] + p2[j][k] * filterY[1] + p3[j][k] * filterY[2]) / 2;
            }
        }
    }

    return (0);
}

// 3x3 Sobel Y filter as separable 1x3 filters (detects horizontal edges)
// Args: color src image     Return: 16-bit signed short dst image
int sobelY3x3(cv::Mat &src, cv::Mat &dst)
{
    static cv::Mat temp;
    // makes an intermediate temp matrix
    temp = cv::Mat::zeros(src.size(), CV_16SC3);

    // makes a dst matrix for the final image
    dst = cv::Mat::zeros(src.size(), CV_16SC3);

    // horizontal part of Sobel Y filter
    int filterX[3] = {1, 2, 1};

    // first pass (horizontal filter)
    for (int i = 0; i < dst.rows; i++)
    {
        cv::Vec3b *srcPtr = src.ptr<cv::Vec3b>(i);   // gets the row pointer from the src image
        cv::Vec3s *tempPtr = temp.ptr<cv::Vec3s>(i); // gets the row pointer from the temp image
        for (int j = 1; j < dst.cols - 1; j++)
        {
            // loop over RGB color channels
            for (int k = 0; k < 3; k++)
            {
                // sum of the horizontally neighboring pixel values from the original src image (multiplied by filterX)
                // divide by the sum of values in the blur vector to normalize
                tempPtr[j][k] = (srcPtr[j - 1][k] * filterX[0] + srcPtr[j][k] * filterX[1] + srcPtr[j + 1][k] * filterX[2]) / 4;
            }
        }
    }

    // vertical part of Sobel Y filter
    int filterY[3] = {1, 0, -1};
    // second pass (vertical filter) using the generated temp image
    for (int i = 1; i < dst.rows - 1; i++)
    {
        // gets the 3 row pointers from the temp image
        cv::Vec3s *p1 = temp.ptr<cv::Vec3s>(i - 1);
        cv::Vec3s *p2 = temp.ptr<cv::Vec3s>(i);
        cv::Vec3s *p3 = temp.ptr<cv::Vec3s>(i + 1);

        cv::Vec3s *dstPtr = dst.ptr<cv::Vec3s>(i);
        for (int j = 0; j < dst.cols; j++)
        {
            // loop over RGB color channels
            for (int k = 0; k < 3; k++)
            {
                // sum of the vertically neighboring pixel values from each of the 3 rows in the temp image (multiplied by filterY)
                // multiply by 2 to make the edges brighter
                dstPtr[j][k] = (p1[j][k] * filterY[0] + p2[j][k] * filterY[1] + p3[j][k] * filterY[2]) * 2;
            }
        }
    }

    return (0);
}

// Generates a gradient magnitude image from the X and Y Sobel images
// Args: 16-bit signed short Sobel X and Sobel Y images     Return: 8-bit uchar dst image
int magnitude(cv::Mat &sx, cv::Mat &sy, cv::Mat &dst)
{
    dst.create(sx.size(), CV_8UC3);
    for (int i = 0; i < dst.rows; i++)
    {
        cv::Vec3s *sxPtr = sx.ptr<cv::Vec3s>(i); // row pointer for sx image
        cv::Vec3s *syPtr = sy.ptr<cv::Vec3s>(i); // row pointer for sy image
        cv::Vec3b *ptr = dst.ptr<cv::Vec3b>(i);  // row pointer for dst image
        for (int j = 0; j < dst.cols; j++)
        {
            for (int k = 0; k < 3; k++)
            {
                int val = std::sqrt(sxPtr[j][k] * sxPtr[j][k] + syPtr[j][k] * syPtr[j][k]);
                if (val > 255)
                    val = 255; // clamp the values to 255
                ptr[j][k] = (uchar)val;
            }
        }
    }

    return (0);
}

// Creates a 3D normalized RGB histogram from the src image (with 8 bins per color channel)
// Adds another 3D normalized RGB histogram for the sobel magnitude image
// Builds a feature vector from the histograms (8x8x8 x 2 histograms)
// Args: src     - cv::Mat image
//       featVec - feature vector to be filled
void extract_sobel_features(cv::Mat &src, std::vector<float> &featVec)
{
    // histogram of entire image
    extract_histogram_rgb_features(src, featVec);

    // compute sobel magnitude using sobel X and Y
    cv::Mat sX, sY, mag;
    sobelX3x3(src, sX);
    sobelY3x3(src, sY);
    magnitude(sX, sY, mag);
    extract_histogram_rgb_features(mag, featVec);
}

// Append the DNN embeddings to the existing feature vector by matching the filenames
// Finds the feature vector with the same filename as the current image and appends its DNN embeddings to the vector
// Args: featVec   - feature vector to be filled
//       filename  - file name of the current image
//       filenames - vector of filenames of DNN embeddings
//       data      - vector of DNN embeddings for each image in the DB
void append_dnn_vector(std::vector<float> &featVec, char *filename,
                      std::vector<char *> &filenames, std::vector<std::vector<float>> &data)
{
    for (int i = 0; i < filenames.size(); i++)
    {
        if (strcmp(filename, filenames[i]) == 0)
        {
            featVec.insert(featVec.end(), data[i].begin(), data[i].end());
        }
    }
}