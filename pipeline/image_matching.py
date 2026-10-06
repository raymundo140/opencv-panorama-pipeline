import argparse
import json
import re
from pathlib import Path

import numpy as np


# ============================================================
# NATURAL SORTING
# ============================================================

def natural_sort_key(path):
    """
    Sorts filenames naturally.

    Example:

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


# ============================================================
# FIND FEATURE FILES
# ============================================================

def find_feature_files(
    features_dir,
):
    """
    Finds all SIFT feature files created by
    feature_extraction.py.

    Expected format:

        *_features.npz
    """

    features_dir = Path(
        features_dir
    )

    if not features_dir.exists():

        raise RuntimeError(
            "\nFeature directory does not exist:\n"
            f"{features_dir}\n"
        )

    feature_files = list(
        features_dir.glob(
            "*_features.npz"
        )
    )

    feature_files = sorted(
        feature_files,
        key=natural_sort_key,
    )

    return feature_files


# ============================================================
# READ FEATURE INFORMATION
# ============================================================

def read_feature_info(
    feature_file,
):
    """
    Reads basic information from one feature file.

    We do not perform matching here.

    We only check that the file contains:

        image name
        keypoints
        descriptors
    """

    data = np.load(
        feature_file
    )

    required_keys = {
        "image_name",
        "keypoints",
        "descriptors",
    }

    missing = (
        required_keys
        - set(data.files)
    )

    if missing:

        raise RuntimeError(
            f"\nFeature file is missing data:\n"
            f"{feature_file}\n"
            f"Missing keys: {missing}\n"
        )

    image_name = str(
        data["image_name"]
    )

    keypoints = (
        data["keypoints"]
    )

    descriptors = (
        data["descriptors"]
    )

    return {
        "feature_file": (
            str(
                feature_file.resolve()
            )
        ),

        "image_name": (
            image_name
        ),

        "number_keypoints": (
            len(keypoints)
        ),

        "descriptor_shape": (
            list(
                descriptors.shape
            )
        ),
    }


# ============================================================
# CREATE SEQUENTIAL IMAGE PAIRS
# ============================================================

def create_sequential_pairs(
    feature_files,
):
    """
    Creates pairs between consecutive panorama images.

    Example:

        image01 <-> image02
        image02 <-> image03
        image03 <-> image04

    This works because our panorama pictures were
    intentionally captured in sequence with overlap.
    """

    if len(feature_files) < 2:

        raise RuntimeError(
            "At least two images are required "
            "to create panorama pairs."
        )

    image_information = []

    # --------------------------------------------------------
    # Read information about every feature file
    # --------------------------------------------------------

    for feature_file in feature_files:

        info = read_feature_info(
            feature_file
        )

        image_information.append(
            info
        )

    pairs = []

    # --------------------------------------------------------
    # Pair adjacent images
    # --------------------------------------------------------

    for index in range(
        len(image_information) - 1
    ):

        left = (
            image_information[index]
        )

        right = (
            image_information[
                index + 1
            ]
        )

        pair = {

            "pair_id": (
                index
            ),

            "image_a": (
                left["image_name"]
            ),

            "image_b": (
                right["image_name"]
            ),

            "features_a": (
                left["feature_file"]
            ),

            "features_b": (
                right["feature_file"]
            ),

            "keypoints_a": (
                left[
                    "number_keypoints"
                ]
            ),

            "keypoints_b": (
                right[
                    "number_keypoints"
                ]
            ),
        }

        pairs.append(
            pair
        )

    return pairs


# ============================================================
# SAVE PAIRS
# ============================================================

def save_pairs(
    pairs,
    output_file,
):
    """
    Saves image-pair information as JSON.

    JSON is used because it is human-readable
    and easy for later pipeline stages to load.
    """

    output_file = Path(
        output_file
    )

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    data = {
        "pairing_method": (
            "sequential"
        ),

        "number_pairs": (
            len(pairs)
        ),

        "pairs": (
            pairs
        ),
    }

    with open(
        output_file,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            data,
            file,
            indent=4,
        )


# ============================================================
# LOAD PAIRS
# ============================================================

def load_pairs(
    pair_file,
):
    """
    Loads the image-pair JSON file.

    This function will be used by
    feature_matching.py.
    """

    pair_file = Path(
        pair_file
    )

    if not pair_file.exists():

        raise RuntimeError(
            "\nImage-pair file does not exist:\n"
            f"{pair_file}\n"
        )

    with open(
        pair_file,
        "r",
        encoding="utf-8",
    ) as file:

        data = json.load(
            file
        )

    return data["pairs"]


# ============================================================
# MAIN IMAGE MATCHING FUNCTION
# ============================================================

def create_image_pairs(
    features_dir,
    output_file,
):
    """
    Determines which panorama images should
    be compared during feature matching.

    For the current panorama, we use sequential
    image pairing.
    """

    print()
    print("========================================")
    print(" IMAGE MATCHING / PAIR SELECTION")
    print("========================================")
    print()

    feature_files = (
        find_feature_files(
            features_dir
        )
    )

    if not feature_files:

        raise RuntimeError(
            "\nNo feature files found in:\n"
            f"{features_dir}\n"
        )

    print(
        f"Feature files found: "
        f"{len(feature_files)}"
    )

    print()

    # --------------------------------------------------------
    # Display images in detected order
    # --------------------------------------------------------

    print(
        "Detected panorama sequence:"
    )

    print()

    for index, feature_file in enumerate(
        feature_files,
        start=1,
    ):

        info = read_feature_info(
            feature_file
        )

        print(
            f"{index:02d}. "
            f"{info['image_name']} "
            f"({info['number_keypoints']} keypoints)"
        )

    # --------------------------------------------------------
    # Create adjacent pairs
    # --------------------------------------------------------

    pairs = (
        create_sequential_pairs(
            feature_files
        )
    )

    print()
    print("----------------------------------------")
    print(" Candidate image pairs")
    print("----------------------------------------")
    print()

    for pair in pairs:

        print(
            f"Pair {pair['pair_id']:02d}: "
            f"{pair['image_a']} "
            f"<-> "
            f"{pair['image_b']}"
        )

        print(
            f"    Keypoints: "
            f"{pair['keypoints_a']} "
            f"<-> "
            f"{pair['keypoints_b']}"
        )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    save_pairs(
        pairs,
        output_file,
    )

    print()
    print("========================================")
    print(" IMAGE PAIRING COMPLETE")
    print("========================================")
    print()

    print(
        f"Images: "
        f"{len(feature_files)}"
    )

    print(
        f"Candidate pairs: "
        f"{len(pairs)}"
    )

    print()

    print(
        "Pair information saved to:"
    )

    print(
        output_file
    )

    print()

    return pairs


# ============================================================
# COMMAND LINE
# ============================================================

def parse_arguments():

    parser = argparse.ArgumentParser(
        description=(
            "Determine candidate image pairs "
            "for panorama feature matching."
        )
    )

    parser.add_argument(
        "--features",
        required=True,
        help=(
            "Directory containing "
            "SIFT feature NPZ files."
        ),
    )

    parser.add_argument(
        "--output",
        required=True,
        help=(
            "JSON file where image pairs "
            "will be saved."
        ),
    )

    return parser.parse_args()


# ============================================================
# RUN DIRECTLY
# ============================================================

if __name__ == "__main__":

    args = parse_arguments()

    create_image_pairs(
        features_dir=args.features,
        output_file=args.output,
    )