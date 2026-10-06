OpenCV Panorama Pipeline
========================

A reusable computer-vision pipeline for reconstructing 2D panoramas from overlapping images using Python and OpenCV.

The project implements the main stages of panorama reconstruction explicitly instead of relying on OpenCV's high-level Stitcher API. This makes each stage inspectable, reusable, and easier to modify.

Pipeline
--------

Camera Calibration
        |
        v
Image Undistortion
        |
        v
SIFT Feature Extraction
        |
        v
Image Pair Selection
        |
        v
Feature Matching
        |
        v
RANSAC Homography Estimation
        |
        v
Perspective Warping
        |
        v
Feather Blending
        |
        v
Final Panorama

Features
--------

- Camera calibration using a planar checkerboard
- Manual Zhang-style calibration implementation for comparison
- OpenCV calibration for the production pipeline
- Lens-distortion correction
- SIFT keypoint and descriptor extraction
- Sequential image-pair generation
- KNN descriptor matching
- Lowe ratio filtering
- RANSAC outlier rejection
- Pairwise homography estimation
- Global homography composition
- Perspective warping with cv2.warpPerspective
- Automatic panorama-canvas calculation
- Feather blending in overlapping regions
- Automatic panorama cropping
- Debug images and intermediate outputs for every major stage
- Reusable pipeline modules shared by multiple examples

## Repository Structure

```text
opencv-panorama-pipeline/
├── pipeline/
│   ├── __init__.py
│   ├── config.py
│   ├── calibration.py
│   ├── undistortion.py
│   ├── feature_extraction.py
│   ├── image_matching.py
│   ├── feature_matching.py
│   ├── homography.py
│   ├── warping.py
│   ├── stitching.py
│   ├── calibration_data/
│   └── calibration_results/
│
├── example1_cerro/
│   ├── main.py
│   ├── media/
│   └── output/
│
├── example2_drawing/
│   ├── main.py
│   ├── media/
│   └── output/
│
├── requirements.txt
├── .gitignore
├── README.md
└── LICENSE.txt
```
Requirements
------------

- Python 3
- NumPy
- OpenCV

Install dependencies with:

    pip install -r requirements.txt

A minimal requirements.txt can contain:

    numpy
    opencv-python

How the Pipeline Works
----------------------

1. Camera Calibration

The pipeline uses a calibrated camera model to estimate the camera intrinsic matrix and radial lens distortion.

The intrinsic matrix has the form:

    K = [ fx   0   cx ]
        [  0  fy   cy ]
        [  0   0    1 ]

The calibration implementation includes both a manual Zhang-style planar calibration implementation and OpenCV's optimized camera-calibration routine.

The OpenCV solution is stored as the calibration used by the rest of the pipeline:

    pipeline/calibration_results/camera_calibration.npz

2. Image Undistortion

Each input image is corrected using the saved camera calibration before feature extraction. This reduces the effect of lens distortion and improves geometric consistency.

3. SIFT Feature Extraction

SIFT is used to detect distinctive keypoints and compute local descriptors for every image. Each descriptor contains 128 values describing the local appearance around a keypoint.

4. Image Pair Selection

The current implementation assumes panorama images were captured sequentially.

For:

    1 -> 2 -> 3 -> 4 -> 5

the pipeline evaluates:

    1 <-> 2
    2 <-> 3
    3 <-> 4
    4 <-> 5

5. Feature Matching

Descriptors from adjacent images are matched using K-nearest-neighbor matching. Lowe's ratio test rejects ambiguous correspondences before geometric estimation.

6. Homography Estimation with RANSAC

A projective transformation relates corresponding points between two images:

    p2 ~ H p1

RANSAC is used to estimate the homography while rejecting incorrect geometric matches.

7. Perspective Warping

Pairwise transformations are composed toward a central reference image. Each image is projected into a shared panorama coordinate system using:

    cv2.warpPerspective()

8. Stitching

Warped images are combined using feather blending. Pixels in overlapping regions are weighted gradually instead of being overwritten directly, reducing hard seams.

Running an Example
------------------

Move into an example directory:

    cd example1_cerro

Run:

    python3 main.py

By default, the program uses the saved camera calibration.

Equivalent command:

    python3 main.py --calibration saved

To run calibration again first:

    python3 main.py --calibration run

Input Images
------------

Place source images inside the example's media directory:

    example1_cerro/media/

or:

    example2_drawing/media/

For best results, images should:

- come from the same camera
- contain significant overlap with adjacent images
- be captured in sequence
- use the same camera mode and resolution
- minimize strong camera translation when possible
- avoid large moving objects
- maintain reasonably consistent exposure

Output
------

Running main.py creates the output directory automatically.

Typical output structure:

    output/
    |-- undistorted/
    |-- features/
    |-- pairs/
    |-- matches/
    |-- homographies/
    |-- warped/
    |-- warped_masks/
    |-- debug/
    |-- panorama_geometry.npz
    |-- warping_summary.json
    `-- panorama.png

The final panorama is saved as:

    output/panorama.png

Included Examples
-----------------

Example 1 - Outdoor Panorama

The first example reconstructs a wide outdoor panorama from multiple overlapping photographs of Cerro de la Silla and the surrounding scene.

Run it with:

    cd example1_cerro
    python3 main.py

Example 2 - Whiteboard Drawing Reconstruction

The second example reconstructs a complete whiteboard drawing from multiple photographs showing different overlapping sections of the drawing.

Run it with:

    cd example2_drawing
    python3 main.py

This example demonstrates that the same pipeline can also work well for approximately planar content such as drawings, posters, documents, murals, and diagrams.

Committed Example Outputs
-------------------------

The repository intentionally includes the complete output directories for both examples.

This makes it possible to inspect every stage without first running the code, including:

- undistorted images
- SIFT feature data
- image-pair metadata
- feature matches
- homographies
- RANSAC debug visualizations
- warped images
- warped masks
- stitching debug outputs
- final panoramas

Using the Pipeline for Your Own Dataset
---------------------------------------

Create a new directory beside pipeline:

    my_panorama/
    |-- main.py
    `-- media/

Copy main.py from either included example:

    cp example1_cerro/main.py my_panorama/main.py

Place sequential overlapping photographs inside:

    my_panorama/media/

Then run:

    cd my_panorama
    python3 main.py

The pipeline creates:

    my_panorama/output/

and saves the final panorama to:

    my_panorama/output/panorama.png

Camera Calibration for a Different Camera
------------------------------------------

The saved calibration corresponds to the camera configuration used for the included examples.

If you use a different camera, lens, camera mode, or significantly different image configuration, recalibration is recommended.

Place checkerboard calibration images inside:

    pipeline/calibration_data/

Configure the checkerboard geometry in:

    pipeline/config.py

Then run:

    python3 main.py --calibration run

The new result is stored in:

    pipeline/calibration_results/camera_calibration.npz

Important Assumptions
---------------------

The current implementation is designed primarily for ordered image sequences.

It assumes:

1. Adjacent images overlap.
2. Images are provided in their intended panorama order.
3. A homography is a reasonable approximation for the scene relationship.
4. The same camera calibration applies to all source images.

Strong parallax from nearby objects may produce ghosting or local misalignment because a single homography cannot fully model large depth changes.

Potential Improvements
----------------------

Possible future extensions include:

- automatic unordered image-overlap discovery
- FLANN-based descriptor matching
- cylindrical or spherical projection
- exposure compensation
- seam optimization
- multi-band blending
- bundle adjustment
- global homography refinement
- automatic reference-frame selection
- parallel feature extraction
- automatic video-frame extraction

## License

This project is licensed under the [MIT License](LICENSE).

You are free to use, modify, and distribute this software under the terms of the MIT License.
