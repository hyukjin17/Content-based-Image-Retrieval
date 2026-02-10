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

## Prerequisites

To build and run this project, you need:

* **C++ Compiler** (GCC/Clang/MSVC)
* **CMake** (Version 3.10 or higher)
* **OpenCV** (Version 4.x)
    * Must include `opencv_objdetect` for face detection.
    * Must include `opencv_highgui` and `opencv_imgproc`.

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
    git clone [https://github.com/hyukjin17/Content-based-Image-Retrieval.git](https://github.com/hyukjin17/Content-based-Image-Retrieval.git)
    cd Content-based-Image_Retrieval
    ```

2.  **Create a build directory:** (use release config for faster processing)
    ```bash
    cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
    ```

3.  **Run CMake and Compile:**
    ```bash
    cmake --build build --config Release
    make
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