# Content-Based Image Retrieval (CBIR) System

**Author:** Hyuk Jin Chung  
**Date:** February 2026  
**Language:** C++ (OpenCV)

## Overview

This project is a robust **Content-Based Image Retrieval (CBIR)** system implemented in C++. It allows users to query a database of images and retrieve the most visually similar results based on various feature extraction techniques.

The system supports a wide range of matching strategies, from classical computer vision methods (color histograms, texture analysis) to modern deep learning approaches (ResNet18 embeddings) and object-specific matching (Face Detection).

## Features

The application supports the following feature extraction and matching modes:

| Mode | Description | Distance Metric |
| :--- | :--- | :--- |
| **`baseline`** | Matches images using a 7x7 pixel square from the image center. | Sum-of-Squared Difference (SSD) |
| **`hist`** | 2D **RG Chromaticity** Histogram (16x16 bins). Ignores intensity/lightness. | Histogram Intersection |
| **`hist2`** | 3D **RGB** Histogram (8x8x8 bins). Captures full color distribution. | Histogram Intersection |
| **`multihist`** | **Spatial Grid** of RGB Histograms (Top, Bottom, Center). Captures spatial layout. | Weighted Intersection |
| **`hsv`** | 2D **HS Chromaticity** Histogram (16x16 bins). Includes specialized bins for Black/White pixels to handle achromatics. | Histogram Intersection |
| **`sobel`** | **Texture Matching** using Gradient Magnitude (Sobel X/Y). Matches edge density. | Histogram Intersection |
| **`face`** | **Face Detection** (Haar Cascade). Extracts HSV features *only* from the detected face. Falls back to center crop if no face is found. | Custom "Face" Metric (Intersection + Penalty) |
| **`dnn`** | **Deep Learning** Embeddings (ResNet18). Uses a pre-computed 512-dimensional vector. | Cosine Distance |
| **`dnn_hsv`** | **Multi-Modal** Matching. Combines ResNet18 embeddings (semantics) with HSV Histograms (color). | Weighted Sum (Cosine + Intersection) |

## Dependencies

To build and run this project, you need:

* **C++ Compiler**
* **CMake**
* **OpenCV**

## Project Structure

```text
.
├── main.cpp                # Entry point and argument parsing
├── features.cpp / .hpp     # Feature extraction logic (Histograms, Sobel, DNN helpers)
├── faceDetect.cpp / .h     # Wrapper for OpenCV Haar Cascade Face Detection
├── csv_util.cpp / .h       # Utilities for reading/writing feature vectors to CSV
├── CMakeLists.txt          # Build configuration
├── haarcascade_frontalface_alt2.xml  # Required for 'face' mode
└── ResNet18_olym.csv       # Pre-computed Deep Learning embeddings (Required for 'dnn' modes)
```

## Build Instructions

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/hyukjin17/Content-based-Image-Retrieval.git
    cd Content-based-Image_Retrieval
    ```

2.  **Create a build directory:** (use release config for faster processing)
    ```bash
    cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
    ```

3.  **Run CMake and Compile:**
    ```bash
    cmake --build build --config Release
    ```

## Usage

1.  **Run the Application (read all image files and create feature vectors in csv):**
    Ensure the image directory is in the same directory as the executable.
    Also ensure the `haarcascade_frontalface_alt2.xml` file is in the same directory as the executable, or that the path in the code points to it correctly.
    ```bash
    ./build/read <directory>
    ```
    Replace **directory** with the name of image file directory

2.  **Compare chosen image to images in the database:**
    ```bash
    ./build/cbir <directory_path> <feature_method> <num_matches> [bot]
    ```
    - <directory_path>: Path to the image to be used for matching (e.g., olympus/pic.0001.jpg).
    - <feature_method>: One of the modes listed in the Features section (e.g., hsv, dnn_hsv, multihist).
    - <num_matches>: Integer. The number of top matches to display (excluding the query image itself).
    - [bot] (Optional): If provided, sorts results in descending order (worst matches first). Useful for debugging.

### Examples

1.  Find top 3 matches using HSV Color Histograms:
    ```bash
    ./build/cbir olympus/pic.0001.jpg hsv 3
    ```
2.  Find top 5 matches using Face Detection:
    ```bash
    ./build/cbir olympus/pic.0001.jpg face 5


## Methodology Details

### Face Detection Strategy

The face mode uses haarcascade_frontalface_alt2.xml.

1.  It attempts to detect a face in the query image
2.  **If a face is found**: It extracts a feature vector from only the face region
3.  **If no face is found**: It falls back to extracting features from the center 50% of the image
4.  **Matching**: Uses a flag to penalize matches between a "Face" image and a "Non-Face" image.

### Deep Learning (DNN)

The dnn mode relies on ResNet18_olym.csv. This file must contain pre-computed 512-dimensional feature vectors for every image in your database. The C++ program reads these vectors to perform high-speed Cosine Distance matching.

### HSV Color Space

The hsv and face modes utilize a custom 16x16 Hue-Saturation histogram.

- **Black/White Handling**: Low saturation and low value pixels are moved to specific "Black" and "Gray" bins to prevent them from polluting the color data.
- **Saturation Weighting**: Pixels are weighted by their saturation, so vibrant colors contribute more to the histogram than washed-out colors.

### Troubleshooting

- **"Unable to load face cascade file"**: Ensure haarcascade_frontalface_alt2.xml is in the directory where you are running the executable (or update FACE_CASCADE_FILE in faceDetect.h).

- **"Invalid image filepath"**: Ensure the path provided in the command line exists and contains images.

- **Segmentation Fault**: Often caused by a mismatch between the CSV database and the current image directory. Delete the .csv files generated by the program to force a rebuild of the feature database.