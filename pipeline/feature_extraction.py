import argparse
import re
from pathlib import Path

import cv2
import numpy as np


# ============================================================
# IMAGE SEARCH
# ============================================================

def natural_sort_key(path):
    """
    Natural sorting:

        image1
        image2
        image10

    instead of:

        image1
        image10
        image2
    """

    return [
        int(text) if text.isdigit()
        else text.lower()
        for text in re.split(
            r"(\d+)",
            path.name,
        )
    ]


def find_images(input_dir):
    """
    Finds supported image files.
    """

    valid_extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".webp",
    }

    input_dir = Path(input_dir)

    images = []

    for path in input_dir.iterdir():

        if (
            path.is_file()
            and path.suffix.lower()
            in valid_extensions
        ):
            images.append(path)

    return sorted(
        images,
        key=natural_sort_key,
    )


# ============================================================
# MASK INVALID BLACK UNDISTORTION BORDERS
# ============================================================

def create_valid_mask(
    image,
    erosion_size=7,
):
    """
    Creates a mask so SIFT ignores the artificial
    black borders produced by undistortion.

    White:
        SIFT may search here.

    Black:
        SIFT ignores this region.
    """

    if len(image.shape) == 3:

        valid = (
            np.max(
                image,
                axis=2,
            )
            > 2
        )

    else:

        valid = (
            image > 2
        )

    mask = (
        valid.astype(np.uint8)
        * 255
    )

    # --------------------------------------------------------
    # Shrink the valid area slightly.
    #
    # This prevents SIFT from detecting features exactly
    # on the black/artificial border.
    # --------------------------------------------------------

    if erosion_size > 0:

        kernel = np.ones(
            (
                erosion_size,
                erosion_size,
            ),
            dtype=np.uint8,
        )

        mask = cv2.erode(
            mask,
            kernel,
            iterations=1,
        )

    return mask


# ============================================================
# SIFT
# ============================================================

def create_sift_detector(
    max_features=5000,
):
    """
    Creates the OpenCV SIFT detector.

    SIFT produces:

        keypoints
        descriptors

    Each SIFT descriptor has 128 values.
    """

    if not hasattr(
        cv2,
        "SIFT_create",
    ):

        raise RuntimeError(
            "This OpenCV installation "
            "does not support SIFT."
        )

    sift = cv2.SIFT_create(
        nfeatures=max_features,
    )

    return sift


# ============================================================
# EXTRACT FEATURES FROM ONE IMAGE
# ============================================================

def extract_features(
    image,
    sift=None,
    use_mask=True,
):
    """
    Extracts SIFT keypoints and descriptors
    from one image.

    Returns:

        keypoints
        descriptors
        mask
    """

    if image is None:

        raise ValueError(
            "Input image is None."
        )

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY,
    )

    if sift is None:

        sift = create_sift_detector()

    # --------------------------------------------------------
    # Ignore black undistortion borders
    # --------------------------------------------------------

    if use_mask:

        mask = create_valid_mask(
            image
        )

    else:

        mask = None

    # --------------------------------------------------------
    # SIFT does two things:
    #
    # 1. Detect important points
    # 2. Calculate a descriptor for each point
    # --------------------------------------------------------

    keypoints, descriptors = (
        sift.detectAndCompute(
            gray,
            mask,
        )
    )

    # In unusual cases SIFT may find nothing.
    if descriptors is None:

        descriptors = np.empty(
            (0, 128),
            dtype=np.float32,
        )

    return (
        keypoints,
        descriptors,
        mask,
    )


# ============================================================
# CONVERT KEYPOINTS TO NUMPY
# ============================================================

def keypoints_to_array(
    keypoints,
):
    """
    Converts OpenCV KeyPoint objects into a numeric
    array that can be stored inside an NPZ file.

    Columns:

        0 -> x
        1 -> y
        2 -> size
        3 -> angle
        4 -> response
        5 -> octave
        6 -> class_id
    """

    if not keypoints:

        return np.empty(
            (0, 7),
            dtype=np.float32,
        )

    data = []

    for kp in keypoints:

        data.append(
            [
                kp.pt[0],
                kp.pt[1],
                kp.size,
                kp.angle,
                kp.response,
                kp.octave,
                kp.class_id,
            ]
        )

    return np.asarray(
        data,
        dtype=np.float32,
    )


# ============================================================
# SAVE FEATURE DATA
# ============================================================

def save_features(
    output_path,
    image_path,
    keypoints,
    descriptors,
):
    """
    Saves SIFT results to an NPZ file.

    This allows later pipeline stages to use
    the features without running SIFT again.
    """

    keypoint_array = (
        keypoints_to_array(
            keypoints
        )
    )

    np.savez_compressed(
        output_path,

        image_name=np.array(
            image_path.name
        ),

        image_path=np.array(
            str(image_path)
        ),

        keypoints=keypoint_array,

        descriptors=descriptors,
    )


# ============================================================
# LOAD FEATURE DATA
# ============================================================

def load_features(
    feature_file,
):
    """
    Loads a feature NPZ file.

    Later feature-matching and homography stages
    can use this function.
    """

    data = np.load(
        feature_file
    )

    return {
        "image_name": str(
            data["image_name"]
        ),

        "image_path": str(
            data["image_path"]
        ),

        "keypoints": (
            data["keypoints"]
        ),

        "descriptors": (
            data["descriptors"]
        ),
    }


# ============================================================
# DEBUG VISUALIZATION
# ============================================================

def create_feature_visualization(
    image,
    keypoints,
    max_drawn=800,
):
    """
    Creates an image showing SIFT keypoints.

    Only the strongest keypoints are drawn so
    the debug image does not become unreadable.
    """

    if not keypoints:

        return image.copy()

    # --------------------------------------------------------
    # Sort by response.
    #
    # Stronger SIFT features have a larger response.
    # --------------------------------------------------------

    strongest = sorted(
        keypoints,
        key=lambda kp: kp.response,
        reverse=True,
    )

    strongest = strongest[
        :max_drawn
    ]

    visualization = (
        cv2.drawKeypoints(
            image,
            strongest,
            None,
            flags=(
                cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
            ),
        )
    )

    return visualization


# ============================================================
# PROCESS DIRECTORY
# ============================================================

def extract_features_directory(
    input_dir,
    output_dir,
    max_features=5000,
):
    """
    Runs SIFT on every image in a directory.

    Saves:

        .npz feature data
        debug keypoint images

    Returns information about all processed images.
    """

    input_dir = Path(
        input_dir
    )

    output_dir = Path(
        output_dir
    )

    if not input_dir.exists():

        raise RuntimeError(
            "\nInput directory does not exist:\n"
            f"{input_dir}\n"
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    debug_dir = (
        output_dir.parent
        / "debug"
        / "features"
    )

    debug_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    image_paths = find_images(
        input_dir
    )

    if not image_paths:

        raise RuntimeError(
            "\nNo images found in:\n"
            f"{input_dir}\n"
        )

    print()
    print("========================================")
    print(" SIFT FEATURE EXTRACTION")
    print("========================================")
    print()

    print(
        f"Images found: "
        f"{len(image_paths)}"
    )

    print(
        f"Maximum features per image: "
        f"{max_features}"
    )

    print()

    sift = create_sift_detector(
        max_features=max_features
    )

    results = []

    total_keypoints = 0

    # ========================================================
    # PROCESS EACH IMAGE
    # ========================================================

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
                "  ERROR: could not read image"
            )

            continue

        height, width = (
            image.shape[:2]
        )

        print(
            f"  Resolution: "
            f"{width} x {height}"
        )

        # ----------------------------------------------------
        # SIFT
        # ----------------------------------------------------

        (
            keypoints,
            descriptors,
            mask,
        ) = extract_features(
            image,
            sift=sift,
            use_mask=True,
        )

        number_keypoints = len(
            keypoints
        )

        total_keypoints += (
            number_keypoints
        )

        print(
            f"  SIFT keypoints: "
            f"{number_keypoints}"
        )

        print(
            f"  Descriptor shape: "
            f"{descriptors.shape}"
        )

        # ----------------------------------------------------
        # Save feature data
        # ----------------------------------------------------

        feature_path = (
            output_dir
            / (
                image_path.stem
                + "_features.npz"
            )
        )

        save_features(
            feature_path,
            image_path,
            keypoints,
            descriptors,
        )

        print(
            f"  Features saved: "
            f"{feature_path}"
        )

        # ----------------------------------------------------
        # Debug visualization
        # ----------------------------------------------------

        visualization = (
            create_feature_visualization(
                image,
                keypoints,
            )
        )

        debug_path = (
            debug_dir
            / (
                image_path.stem
                + "_sift.jpg"
            )
        )

        cv2.imwrite(
            str(debug_path),
            visualization,
        )

        # ----------------------------------------------------
        # Save mask for debugging
        # ----------------------------------------------------

        mask_path = (
            debug_dir
            / (
                image_path.stem
                + "_mask.jpg"
            )
        )

        cv2.imwrite(
            str(mask_path),
            mask,
        )

        results.append(
            {
                "image_path": (
                    image_path
                ),

                "feature_path": (
                    feature_path
                ),

                "keypoints": (
                    keypoints
                ),

                "descriptors": (
                    descriptors
                ),

                "number_keypoints": (
                    number_keypoints
                ),
            }
        )

        print()

    # ========================================================
    # SUMMARY
    # ========================================================

    print("========================================")
    print(" FEATURE EXTRACTION COMPLETE")
    print("========================================")
    print()

    print(
        f"Images processed: "
        f"{len(results)}/"
        f"{len(image_paths)}"
    )

    print(
        f"Total SIFT keypoints: "
        f"{total_keypoints}"
    )

    if results:

        average = (
            total_keypoints
            / len(results)
        )

        print(
            f"Average keypoints per image: "
            f"{average:.1f}"
        )

    print()

    print(
        "Feature files:"
    )

    print(
        output_dir
    )

    print()

    print(
        "Debug visualizations:"
    )

    print(
        debug_dir
    )

    print()

    return results


# ============================================================
# COMMAND LINE
# ============================================================

def parse_arguments():

    parser = argparse.ArgumentParser(
        description=(
            "Extract SIFT features from "
            "undistorted images."
        )
    )

    parser.add_argument(
        "--input",
        required=True,
        help=(
            "Directory containing "
            "undistorted images."
        ),
    )

    parser.add_argument(
        "--output",
        required=True,
        help=(
            "Directory where feature "
            "files will be saved."
        ),
    )

    parser.add_argument(
        "--max-features",
        type=int,
        default=5000,
        help=(
            "Maximum SIFT features "
            "per image. Default: 5000."
        ),
    )

    return parser.parse_args()


# ============================================================
# RUN DIRECTLY
# ============================================================

if __name__ == "__main__":

    args = parse_arguments()

    extract_features_directory(
        input_dir=args.input,
        output_dir=args.output,
        max_features=args.max_features,
    )