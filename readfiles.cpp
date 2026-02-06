/*
  Hyuk Jin Chung
  2/5/26

  Modified code to read the images and build a feature vector for each image


  Bruce A. Maxwell
  S21

  Sample code to identify image fils in a directory
*/
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <dirent.h>
#include "opencv2/opencv.hpp"
#include "features.hpp"
#include "csv_util.h"

/*
  Extracts features based on the chosen feature extraction method and saves the feature vector to the appropriate csv
  Feature extraction methods: baseline, hist, hist2, multihist, texture, dnn, all

  Args:
    - src: cv::Mat image used for feature extraction
    - img_filename: image filename
    - featVec: feature vector to be filled
    - feat_extraction: feature extraction method set by the user
    - reset_file: if true, open the file in 'write' mode and clear the existing contents
                  else, open the file in 'append' mode
*/
void extract_feature_to_csv(cv::Mat &src, char *img_filename,
                            std::vector<float> &featVec, char *feature_mode, int &reset_file)
{
  char baseline[] = "features_baseline.csv";
  char hist[] = "features_histogram.csv";
  char hist2[] = "features_histogram_rgb.csv";

  bool do_baseline = (strcmp(feature_mode, "baseline") == 0 || strcmp(feature_mode, "all") == 0);
  bool do_hist = (strcmp(feature_mode, "hist") == 0 || strcmp(feature_mode, "all") == 0);
  bool do_hist_rgb = (strcmp(feature_mode, "hist2") == 0 || strcmp(feature_mode, "all") == 0);
  bool do_nothing = true;

  if (do_baseline)
  {
    // extract the baseline features (7x7 square) into a csv file
    extract_baseline_features(src, featVec);
    // appends the image filename and feature vector into the csv
    append_image_data_csv(baseline, img_filename, featVec, reset_file);
    featVec.clear(); // clear the feature vector before reusing it
    do_nothing = false;
  }
  if (do_hist)
  {
    // extract the rg chromaticity histogram data into a csv file
    extract_histogram_features(src, featVec);
    append_image_data_csv(hist, img_filename, featVec, reset_file);
    featVec.clear(); // clear the feature vector before reusing it
    do_nothing = false;
  }
  if (do_hist_rgb)
  {
    // extract the rgb histogram data into a csv file
    extract_histogram_rgb_features(src, featVec);
    append_image_data_csv(hist2, img_filename, featVec, reset_file);
    featVec.clear(); // clear the feature vector before reusing it
    do_nothing = false;
  }
  if (do_nothing) // if nothing happened
  {
    printf("Invalid feature extraction method\nPlease use one of: baseline, hist, hist2, multihist, texture, dnn, all\n");
    exit(-1);
  }
}

/*
  Given a directory on the command line, scans through the directory for image files.

  Prints out the full path name for each file.  This can be used as an argument to fopen or to cv::imread.
 */
int main(int argc, char *argv[])
{
  char dirname[256];
  char feat_extraction[256];
  char buffer[256];
  char img_filename[256];
  DIR *dirp;
  struct dirent *dp;
  cv::Mat src;
  std::vector<float> featVec; // flattened feature vector

  // check for sufficient arguments
  if (argc < 3)
  {
    printf("usage: %s <directory path>, <feature extraction method>\n", argv[0]);
    exit(-1);
  }

  // get the directory path
  strcpy(dirname, argv[1]);
  // get the feature extraction method
  strcpy(feat_extraction, argv[2]);
  printf("Processing directory %s\n", dirname);

  // open the directory
  dirp = opendir(dirname);
  if (dirp == NULL)
  {
    printf("Cannot open directory %s\n", dirname);
    exit(-1);
  }

  int reset_file = 1; // resets the files initially to clear them before writing to them

  // loop over all the files in the image file listing
  while ((dp = readdir(dirp)) != NULL)
  {
    strcpy(img_filename, dp->d_name);

    // check if the file is an image
    if (strstr(img_filename, ".jpg") || strstr(img_filename, ".png") || strstr(img_filename, ".ppm") || strstr(img_filename, ".tif"))
    {
      printf("Processing image file: %s\n", img_filename);

      // build the overall filename
      strcpy(buffer, dirname);
      strcat(buffer, "/");
      strcat(buffer, img_filename);

      // read the image
      src = cv::imread(buffer);
      if (src.empty())
        continue;

      // extracts the features and appends them to the csv
      extract_feature_to_csv(src, img_filename, featVec, feat_extraction, reset_file);

      reset_file = 0; // append to the file after writing the first line
    }
  }

  printf("Terminating\n");

  return (0);
}
