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
  Given a directory on the command line, scans through the directory for image files.

  Prints out the full path name for each file.  This can be used as an argument to fopen or to cv::imread.
 */
int main(int argc, char *argv[])
{
  char dirname[256];
  char buffer[256];
  DIR *dirp;
  struct dirent *dp;
  cv::Mat src;
  std::vector<float> featVec; // flattened feature vector
  char baseline[] = "features_baseline.csv";
  char hist[] = "features_histogram.csv";

  // check for sufficient arguments
  if (argc < 2)
  {
    printf("usage: %s <directory path>\n", argv[0]);
    exit(-1);
  }

  // get the directory path
  strcpy(dirname, argv[1]);
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

    // check if the file is an image
    if (strstr(dp->d_name, ".jpg") || strstr(dp->d_name, ".png") || strstr(dp->d_name, ".ppm") || strstr(dp->d_name, ".tif"))
    {
      printf("Processing image file: %s\n", dp->d_name);

      // build the overall filename
      strcpy(buffer, dirname);
      strcat(buffer, "/");
      strcat(buffer, dp->d_name);

      // read the image
      src = cv::imread(buffer);
      if (src.empty())
        continue;

      // extract the baseline features (7x7 square) into a csv file
      extract_baseline_features(src, featVec);
      // appends the image filename and feature vector into the csv
      append_image_data_csv(baseline, dp->d_name, featVec, reset_file);
      featVec.clear(); // clear the vector before reusing it

      // extract the histogram data into a csv file
      extract_histogram_features(src, featVec);
      append_image_data_csv(hist, dp->d_name, featVec, reset_file);
      reset_file = 0; // append to the file after writing the first line
    }
  }

  printf("Terminating\n");

  return (0);
}
