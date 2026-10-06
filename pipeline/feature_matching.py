import argparse
from pathlib import Path

import cv2
import numpy as np

from image_matching import load_pairs
from feature_extraction import load_features


# ============================================================
# CONVERT STORED KEYPOINTS BACK TO OPENCV KEYPOINTS
# ============================================================

def array_to_keypoints(
    keypoint_array,
):
    """
    Reconstructs cv2.KeyPoint objects from the numeric
    keypoint array stored by feature_extraction.py.

    Stored columns:

        0 -> x
        1 -> y
        2 -> size
        3 -> angle
        4 -> response
        5 -> octave
        6 -> class_id
    """

    keypoints = []

    for row in keypoint_array:

        kp = cv2.KeyPoint(
            x=float(row[0]),
            y=float(row[1]),
            size=float(row[2]),
            angle=float(row[3]),
            response=float(row[4]),
            octave=int(row[5]),
            class_id=int(row[6]),
        )

        keypoints.append(
            kp
        )

    return keypoints


# ============================================================
# KNN + LOWE RATIO TEST
# ============================================================

def match_descriptors(
    descriptors_a,
    descriptors_b,
    ratio_threshold=0.75,
):
    """
    Matches SIFT descriptors using KNN.

    For every descriptor in image A:

        nearest match       = m
        second-best match   = n

    Lowe ratio test:

        m.distance < ratio * n.distance

    If the best match is significantly better than the
    second-best match, we keep it.

    Returns the good cv2.DMatch objects.
    """

    if descriptors_a is None:
        return []

    if descriptors_b is None:
        return []

    if len(descriptors_a) == 0:
        return []

    if len(descriptors_b) < 2:
        return []

    # --------------------------------------------------------
    # SIFT descriptors use floating point values.
    #
    # Euclidean / L2 distance is appropriate for SIFT.
    # --------------------------------------------------------

    matcher = cv2.BFMatcher(
        cv2.NORM_L2,
        crossCheck=False,
    )

    raw_matches = matcher.knnMatch(
        descriptors_a.astype(
            np.float32
        ),
        descriptors_b.astype(
            np.float32
        ),
        k=2,
    )

    good_matches = []

    for pair in raw_matches:

        # Occasionally KNN may return fewer than 2 matches.
        if len(pair) < 2:
            continue

        best_match = pair[0]
        second_match = pair[1]

        if (
            best_match.distance
            < ratio_threshold
            * second_match.distance
        ):

            good_matches.append(
                best_match
            )

    # Best matches first.
    good_matches = sorted(
        good_matches,
        key=lambda match: match.distance,
    )

    return good_matches


# ============================================================
# EXTRACT MATCHED XY COORDINATES
# ============================================================

def extract_matched_points(
    keypoints_a,
    keypoints_b,
    matches,
):
    """
    Converts feature matches into corresponding 2D points.

    Output:

        points_a[i] <-> points_b[i]

    These point pairs will be passed to RANSAC when
    estimating the homography.
    """

    points_a = []

    points_b = []

    query_indices = []
    train_indices = []

    distances = []

    for match in matches:

        point_a = (
            keypoints_a[
                match.queryIdx
            ].pt
        )

        point_b = (
            keypoints_b[
                match.trainIdx
            ].pt
        )

        points_a.append(
            point_a
        )

        points_b.append(
            point_b
        )

        query_indices.append(
            match.queryIdx
        )

        train_indices.append(
            match.trainIdx
        )

        distances.append(
            match.distance
        )

    return (
        np.asarray(
            points_a,
            dtype=np.float32,
        ),

        np.asarray(
            points_b,
            dtype=np.float32,
        ),

        np.asarray(
            query_indices,
            dtype=np.int32,
        ),

        np.asarray(
            train_indices,
            dtype=np.int32,
        ),

        np.asarray(
            distances,
            dtype=np.float32,
        ),
    )


# ============================================================
# SAVE MATCH DATA
# ============================================================

def save_matches(
    output_file,
    pair_id,
    image_a,
    image_b,
    feature_file_a,
    feature_file_b,
    points_a,
    points_b,
    query_indices,
    train_indices,
    distances,
    ratio_threshold,
):
    """
    Saves feature correspondences for later homography
    estimation.
    """

    output_file = Path(
        output_file
    )

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.savez_compressed(
        output_file,

        pair_id=np.array(
            pair_id
        ),

        image_a=np.array(
            image_a
        ),

        image_b=np.array(
            image_b
        ),

        feature_file_a=np.array(
            str(feature_file_a)
        ),

        feature_file_b=np.array(
            str(feature_file_b)
        ),

        points_a=points_a,

        points_b=points_b,

        query_indices=query_indices,

        train_indices=train_indices,

        distances=distances,

        ratio_threshold=np.array(
            ratio_threshold
        ),
    )


# ============================================================
# LOAD MATCH DATA
# ============================================================

def load_matches(
    match_file,
):
    """
    Loads a saved feature-match NPZ file.

    This will be used by homography.py.
    """

    data = np.load(
        match_file
    )

    return {
        "pair_id": int(
            data["pair_id"]
        ),

        "image_a": str(
            data["image_a"]
        ),

        "image_b": str(
            data["image_b"]
        ),

        "points_a": (
            data["points_a"]
        ),

        "points_b": (
            data["points_b"]
        ),

        "query_indices": (
            data["query_indices"]
        ),

        "train_indices": (
            data["train_indices"]
        ),

        "distances": (
            data["distances"]
        ),
    }


# ============================================================
# CREATE DEBUG VISUALIZATION
# ============================================================

def create_match_visualization(
    image_a,
    keypoints_a,
    image_b,
    keypoints_b,
    matches,
    max_drawn=100,
):
    """
    Creates a side-by-side visualization of the strongest
    feature correspondences.

    Only a limited number are drawn so the image stays
    readable.
    """

    if len(matches) == 0:

        return np.hstack(
            (
                image_a,
                image_b,
            )
        )

    matches_to_draw = (
        matches[:max_drawn]
    )

    visualization = cv2.drawMatches(
        image_a,
        keypoints_a,
        image_b,
        keypoints_b,
        matches_to_draw,
        None,
        flags=(
            cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        ),
    )

    return visualization


# ============================================================
# MATCH ONE IMAGE PAIR
# ============================================================

def match_image_pair(
    pair,
    output_dir,
    debug_dir,
    ratio_threshold=0.75,
):
    """
    Performs feature matching for one image pair.
    """

    pair_id = pair[
        "pair_id"
    ]

    # --------------------------------------------------------
    # Load features
    # --------------------------------------------------------

    features_a = load_features(
        pair["features_a"]
    )

    features_b = load_features(
        pair["features_b"]
    )

    keypoint_array_a = (
        features_a[
            "keypoints"
        ]
    )

    keypoint_array_b = (
        features_b[
            "keypoints"
        ]
    )

    descriptors_a = (
        features_a[
            "descriptors"
        ]
    )

    descriptors_b = (
        features_b[
            "descriptors"
        ]
    )

    # --------------------------------------------------------
    # Reconstruct OpenCV keypoints
    # --------------------------------------------------------

    keypoints_a = (
        array_to_keypoints(
            keypoint_array_a
        )
    )

    keypoints_b = (
        array_to_keypoints(
            keypoint_array_b
        )
    )

    # --------------------------------------------------------
    # KNN + Lowe ratio
    # --------------------------------------------------------

    good_matches = (
        match_descriptors(
            descriptors_a,
            descriptors_b,
            ratio_threshold=(
                ratio_threshold
            ),
        )
    )

    # --------------------------------------------------------
    # Convert matches into XY point correspondences
    # --------------------------------------------------------

    (
        points_a,
        points_b,
        query_indices,
        train_indices,
        distances,
    ) = extract_matched_points(
        keypoints_a,
        keypoints_b,
        good_matches,
    )

    # --------------------------------------------------------
    # Save match data
    # --------------------------------------------------------

    output_file = (
        output_dir
        / (
            f"pair_{pair_id:02d}"
            f"_matches.npz"
        )
    )

    save_matches(
        output_file=output_file,
        pair_id=pair_id,
        image_a=(
            features_a[
                "image_name"
            ]
        ),
        image_b=(
            features_b[
                "image_name"
            ]
        ),
        feature_file_a=(
            pair["features_a"]
        ),
        feature_file_b=(
            pair["features_b"]
        ),
        points_a=points_a,
        points_b=points_b,
        query_indices=(
            query_indices
        ),
        train_indices=(
            train_indices
        ),
        distances=distances,
        ratio_threshold=(
            ratio_threshold
        ),
    )

    # --------------------------------------------------------
    # Debug visualization
    # --------------------------------------------------------

    image_path_a = Path(
        features_a[
            "image_path"
        ]
    )

    image_path_b = Path(
        features_b[
            "image_path"
        ]
    )

    image_a = cv2.imread(
        str(image_path_a)
    )

    image_b = cv2.imread(
        str(image_path_b)
    )

    if (
        image_a is not None
        and image_b is not None
    ):

        visualization = (
            create_match_visualization(
                image_a,
                keypoints_a,
                image_b,
                keypoints_b,
                good_matches,
            )
        )

        debug_file = (
            debug_dir
            / (
                f"pair_{pair_id:02d}"
                f"_matches.jpg"
            )
        )

        cv2.imwrite(
            str(debug_file),
            visualization,
        )

    return {
        "pair_id": (
            pair_id
        ),

        "image_a": (
            features_a[
                "image_name"
            ]
        ),

        "image_b": (
            features_b[
                "image_name"
            ]
        ),

        "number_matches": (
            len(good_matches)
        ),

        "output_file": (
            output_file
        ),
    }


# ============================================================
# PROCESS ALL IMAGE PAIRS
# ============================================================

def match_all_pairs(
    pair_file,
    output_dir,
    ratio_threshold=0.75,
):
    """
    Runs SIFT feature matching over all candidate image pairs.
    """

    pair_file = Path(
        pair_file
    )

    output_dir = Path(
        output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    debug_dir = (
        output_dir.parent
        / "debug"
        / "matches"
    )

    debug_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Load candidate pairs
    # --------------------------------------------------------

    pairs = load_pairs(
        pair_file
    )

    if not pairs:

        raise RuntimeError(
            "No image pairs found."
        )

    print()
    print("========================================")
    print(" SIFT FEATURE MATCHING")
    print("========================================")
    print()

    print(
        f"Candidate pairs: "
        f"{len(pairs)}"
    )

    print(
        f"Lowe ratio threshold: "
        f"{ratio_threshold}"
    )

    print()

    results = []

    total_matches = 0

    # ========================================================
    # PROCESS EACH PAIR
    # ========================================================

    for index, pair in enumerate(
        pairs,
        start=1,
    ):

        print(
            f"[{index}/{len(pairs)}] "
            f"{pair['image_a']} "
            f"<-> "
            f"{pair['image_b']}"
        )

        result = (
            match_image_pair(
                pair=pair,
                output_dir=(
                    output_dir
                ),
                debug_dir=(
                    debug_dir
                ),
                ratio_threshold=(
                    ratio_threshold
                ),
            )
        )

        number_matches = (
            result[
                "number_matches"
            ]
        )

        total_matches += (
            number_matches
        )

        print(
            f"  Good matches: "
            f"{number_matches}"
        )

        # ----------------------------------------------------
        # Homography needs at least 4 correspondences.
        # ----------------------------------------------------

        if number_matches < 4:

            print(
                "  WARNING: fewer than 4 matches."
            )

            print(
                "  Homography cannot be estimated."
            )

        elif number_matches < 20:

            print(
                "  WARNING: low number of matches."
            )

            print(
                "  Homography may be unstable."
            )

        else:

            print(
                "  Match count looks usable."
            )

        print(
            f"  Saved: "
            f"{result['output_file']}"
        )

        print()

        results.append(
            result
        )

    # ========================================================
    # SUMMARY
    # ========================================================

    print("========================================")
    print(" FEATURE MATCHING COMPLETE")
    print("========================================")
    print()

    print(
        f"Pairs processed: "
        f"{len(results)}"
    )

    print(
        f"Total good matches: "
        f"{total_matches}"
    )

    if results:

        average = (
            total_matches
            / len(results)
        )

        print(
            f"Average matches per pair: "
            f"{average:.1f}"
        )

    print()

    print(
        "Match files:"
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
            "Match SIFT descriptors between "
            "candidate panorama image pairs."
        )
    )

    parser.add_argument(
        "--pairs",
        required=True,
        help=(
            "JSON image-pair file generated "
            "by image_matching.py."
        ),
    )

    parser.add_argument(
        "--output",
        required=True,
        help=(
            "Directory where feature-match "
            "NPZ files will be saved."
        ),
    )

    parser.add_argument(
        "--ratio",
        type=float,
        default=0.75,
        help=(
            "Lowe ratio-test threshold. "
            "Default: 0.75."
        ),
    )

    return parser.parse_args()


# ============================================================
# RUN DIRECTLY
# ============================================================

if __name__ == "__main__":

    args = parse_arguments()

    match_all_pairs(
        pair_file=args.pairs,
        output_dir=args.output,
        ratio_threshold=args.ratio,
    )