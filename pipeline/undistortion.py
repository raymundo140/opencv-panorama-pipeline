import argparse
from pathlib import Path

import cv2
import numpy as np

from calibration import load_calibration


# ============================================================
# IMAGE SEARCH
# ============================================================

def find_images(input_dir):
    """
    Finds supported images inside a directory.
    """

    valid_extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".webp",
    }

    images = []

    for path in Path(input_dir).iterdir():

        if (
            path.is_file()
            and path.suffix.lower() in valid_extensions
        ):
            images.append(path)

    return sorted(images)


# ============================================================
# CAMERA MATRIX SCALING
# ============================================================

def scale_camera_matrix(
    camera_matrix,
    calibration_size,
    image_size,
):
    """
    Scales the intrinsic matrix K if the input image
    resolution is different from the resolution used
    during calibration.

    This is valid when the image was resized without
    changing its aspect ratio or field of view.

    Example:

        calibration: 1600 x 800
        image:        800 x 400

    Then all pixel-based intrinsic parameters are
    scaled by 0.5.
    """

    calibration_width, calibration_height = (
        calibration_size
    )

    image_width, image_height = (
        image_size
    )

    # No scaling required.
    if (
        calibration_width == image_width
        and calibration_height == image_height
    ):
        return camera_matrix.copy()

    calibration_aspect = (
        calibration_width
        / calibration_height
    )

    image_aspect = (
        image_width
        / image_height
    )

    # --------------------------------------------------------
    # Make sure aspect ratio is basically the same.
    # --------------------------------------------------------

    aspect_difference = abs(
        calibration_aspect
        - image_aspect
    )

    if aspect_difference > 0.01:

        raise RuntimeError(
            "\nInput image aspect ratio does not match "
            "the calibration images.\n"
            f"Calibration resolution: "
            f"{calibration_width} x "
            f"{calibration_height}\n"
            f"Input resolution: "
            f"{image_width} x "
            f"{image_height}\n\n"
            "Do not blindly rescale K when the camera "
            "image has been cropped or captured using "
            "a different aspect ratio."
        )

    scale_x = (
        image_width
        / calibration_width
    )

    scale_y = (
        image_height
        / calibration_height
    )

    scaled_K = (
        camera_matrix.copy()
        .astype(np.float64)
    )

    # Focal lengths
    scaled_K[0, 0] *= scale_x
    scaled_K[1, 1] *= scale_y

    # Principal point
    scaled_K[0, 2] *= scale_x
    scaled_K[1, 2] *= scale_y

    # Skew, if present.
    scaled_K[0, 1] *= scale_x

    return scaled_K


# ============================================================
# UNDISTORT ONE IMAGE
# ============================================================

def undistort_image(
    image,
    camera_matrix,
    dist_coeffs,
    calibration_size,
):
    """
    Removes lens distortion from one image.

    The output keeps the same resolution as the input.
    """

    if image is None:

        raise ValueError(
            "Input image is None."
        )

    height, width = image.shape[:2]

    image_size = (
        width,
        height,
    )

    # --------------------------------------------------------
    # Adjust K if image resolution differs from calibration.
    # --------------------------------------------------------

    K = scale_camera_matrix(
        camera_matrix,
        calibration_size,
        image_size,
    )

    # --------------------------------------------------------
    # Remove lens distortion.
    #
    # We deliberately use K again as the output camera matrix.
    # This keeps the image geometry simple for the following
    # feature extraction and homography stages.
    # --------------------------------------------------------

    undistorted = cv2.undistort(
        image,
        K,
        dist_coeffs,
        None,
        K,
    )

    return undistorted


# ============================================================
# CREATE DEBUG COMPARISON
# ============================================================

def create_comparison(
    original,
    undistorted,
):
    """
    Creates:

        ORIGINAL | UNDISTORTED

    for visual verification.
    """

    if (
        original.shape
        != undistorted.shape
    ):

        raise ValueError(
            "Images must have the same size "
            "for comparison."
        )

    comparison = np.hstack(
        (
            original,
            undistorted,
        )
    )

    return comparison


# ============================================================
# PROCESS A DIRECTORY
# ============================================================

def undistort_directory(
    input_dir,
    output_dir,
    save_comparisons=True,
):
    """
    Undistorts every supported image in a directory.

    Returns a list containing the paths of the
    generated undistorted images.
    """

    input_dir = Path(
        input_dir
    )

    output_dir = Path(
        output_dir
    )

    if not input_dir.exists():

        raise RuntimeError(
            f"\nInput directory does not exist:\n"
            f"{input_dir}\n"
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    debug_dir = (
        output_dir.parent
        / "debug"
        / "undistortion"
    )

    if save_comparisons:

        debug_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    # --------------------------------------------------------
    # Load the OFFICIAL OpenCV calibration.
    # --------------------------------------------------------

    calibration = (
        load_calibration()
    )

    camera_matrix = (
        calibration[
            "camera_matrix"
        ]
    )

    dist_coeffs = (
        calibration[
            "dist_coeffs"
        ]
    )

    calibration_size = (
        calibration[
            "image_size"
        ]
    )

    print()
    print("========================================")
    print(" IMAGE UNDISTORTION")
    print("========================================")
    print()

    print(
        "Calibration resolution:"
    )

    print(
        f"{calibration_size[0]} x "
        f"{calibration_size[1]}"
    )

    print()

    print(
        "Camera matrix K:"
    )

    print(
        camera_matrix
    )

    print()

    print(
        "Distortion coefficients:"
    )

    print(
        dist_coeffs
    )

    print()

    # --------------------------------------------------------
    # Find images
    # --------------------------------------------------------

    image_paths = (
        find_images(
            input_dir
        )
    )

    if not image_paths:

        raise RuntimeError(
            "\nNo images found in:\n"
            f"{input_dir}\n"
        )

    print(
        f"Images found: "
        f"{len(image_paths)}"
    )

    print()

    output_paths = []

    # --------------------------------------------------------
    # Process all images
    # --------------------------------------------------------

    for index, image_path in enumerate(
        image_paths,
        start=1,
    ):

        print(
            f"[{index}/{len(image_paths)}] "
            f"{image_path.name}"
        )

        image = cv2.imread(
            str(image_path)
        )

        if image is None:

            print(
                "  ERROR: image could not be read"
            )

            continue

        height, width = (
            image.shape[:2]
        )

        print(
            f"  Input resolution: "
            f"{width} x {height}"
        )

        # ----------------------------------------------------
        # Undistort
        # ----------------------------------------------------

        undistorted = (
            undistort_image(
                image,
                camera_matrix,
                dist_coeffs,
                calibration_size,
            )
        )

        # ----------------------------------------------------
        # Save undistorted image
        # ----------------------------------------------------

        output_path = (
            output_dir
            / image_path.name
        )

        success = cv2.imwrite(
            str(output_path),
            undistorted,
        )

        if not success:

            print(
                "  ERROR: could not save output"
            )

            continue

        output_paths.append(
            output_path
        )

        print(
            f"  Saved: "
            f"{output_path}"
        )

        # ----------------------------------------------------
        # Save visual comparison
        # ----------------------------------------------------

        if save_comparisons:

            comparison = (
                create_comparison(
                    image,
                    undistorted,
                )
            )

            comparison_name = (
                f"{image_path.stem}"
                f"_comparison.jpg"
            )

            comparison_path = (
                debug_dir
                / comparison_name
            )

            cv2.imwrite(
                str(comparison_path),
                comparison,
            )

    print()
    print("========================================")
    print(" UNDISTORTION COMPLETE")
    print("========================================")
    print()

    print(
        f"Images processed: "
        f"{len(output_paths)}/"
        f"{len(image_paths)}"
    )

    print()

    print(
        "Undistorted images:"
    )

    print(
        output_dir
    )

    if save_comparisons:

        print()

        print(
            "Comparison images:"
        )

        print(
            debug_dir
        )

    print()

    return output_paths


# ============================================================
# COMMAND LINE
# ============================================================

def parse_arguments():

    parser = argparse.ArgumentParser(
        description=(
            "Undistort images using the "
            "saved camera calibration."
        )
    )

    parser.add_argument(
        "--input",
        required=True,
        help=(
            "Directory containing "
            "the original images."
        ),
    )

    parser.add_argument(
        "--output",
        required=True,
        help=(
            "Directory where undistorted "
            "images will be saved."
        ),
    )

    parser.add_argument(
        "--no-comparison",
        action="store_true",
        help=(
            "Do not save side-by-side "
            "debug comparison images."
        ),
    )

    return parser.parse_args()


# ============================================================
# RUN DIRECTLY
# ============================================================

if __name__ == "__main__":

    args = parse_arguments()

    undistort_directory(
        input_dir=args.input,
        output_dir=args.output,
        save_comparisons=(
            not args.no_comparison
        ),
    )