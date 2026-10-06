import argparse
import shutil
import sys
import time
import traceback
from pathlib import Path


# ============================================================
# PROJECT PATHS
# ============================================================
#
# Expected repository structure:
#
# M3/
# ├── example1_cerro/
# │   ├── main.py
# │   ├── media/
# │   └── output/
# │
# └── pipeline/
#     ├── calibration.py
#     ├── config.py
#     ├── undistortion.py
#     ├── feature_extraction.py
#     ├── image_matching.py
#     ├── feature_matching.py
#     ├── homography.py
#     ├── warping.py
#     └── stitching.py
#
# Everything is relative to the location of this file.
# No machine-specific absolute paths are used.
# ============================================================

ACTIVITY_DIR = Path(__file__).resolve().parent

PROJECT_DIR = ACTIVITY_DIR.parent

PIPELINE_DIR = (
    PROJECT_DIR
    / "pipeline"
)


# ============================================================
# MAKE PIPELINE IMPORTABLE
# ============================================================

if str(PIPELINE_DIR) not in sys.path:

    sys.path.insert(
        0,
        str(PIPELINE_DIR),
    )


# ============================================================
# PIPELINE IMPORTS
# ============================================================

from config import CALIBRATION_FILE

from calibration import (
    calibrate_camera,
    load_calibration,
)

from undistortion import (
    undistort_directory,
)

from feature_extraction import (
    extract_features_directory,
)

from image_matching import (
    create_image_pairs,
)

from feature_matching import (
    match_all_pairs,
)

from homography import (
    estimate_all_homographies,
)

from warping import (
    warp_panorama_images,
)

from stitching import (
    stitch_panorama,
)


# ============================================================
# ACTIVITY PATHS
# ============================================================

MEDIA_DIR = (
    ACTIVITY_DIR
    / "media"
)

OUTPUT_DIR = (
    ACTIVITY_DIR
    / "output"
)


# ------------------------------------------------------------
# Pipeline stage outputs
# ------------------------------------------------------------

UNDISTORTED_DIR = (
    OUTPUT_DIR
    / "undistorted"
)

FEATURES_DIR = (
    OUTPUT_DIR
    / "features"
)

PAIRS_DIR = (
    OUTPUT_DIR
    / "pairs"
)

PAIRS_FILE = (
    PAIRS_DIR
    / "image_pairs.json"
)

MATCHES_DIR = (
    OUTPUT_DIR
    / "matches"
)

HOMOGRAPHIES_DIR = (
    OUTPUT_DIR
    / "homographies"
)

WARPED_DIR = (
    OUTPUT_DIR
    / "warped"
)

WARPED_MASKS_DIR = (
    OUTPUT_DIR
    / "warped_masks"
)

PANORAMA_FILE = (
    OUTPUT_DIR
    / "panorama.png"
)


# ============================================================
# PIPELINE PARAMETERS
# ============================================================
#
# These are algorithm settings.
# They are not command-line arguments because this example
# should run with a simple, reproducible configuration.
# ============================================================

SIFT_MAX_FEATURES = 5000

LOWE_RATIO = 0.75

RANSAC_THRESHOLD = 3.0

RANSAC_CONFIDENCE = 0.995

RANSAC_MAX_ITERATIONS = 5000

FEATHER_POWER = 1.0


# ============================================================
# TERMINAL OUTPUT
# ============================================================

def print_header(title):
    """
    Prints a large section header.
    """

    print()
    print("=" * 60)
    print(title)
    print("=" * 60)
    print()


def print_stage(
    stage_number,
    total_stages,
    title,
):
    """
    Prints the current panorama pipeline stage.
    """

    print()
    print("=" * 60)

    print(
        f"STAGE "
        f"{stage_number}/"
        f"{total_stages}"
    )

    print(title)

    print("=" * 60)
    print()


# ============================================================
# PROJECT VALIDATION
# ============================================================

def validate_project():
    """
    Checks that the repository structure and activity
    input images exist before starting the pipeline.

    Returns:
        Number of panorama input images.
    """

    if not PIPELINE_DIR.exists():

        raise RuntimeError(
            "\nPipeline directory was not found:\n"
            f"{PIPELINE_DIR}\n"
        )

    if not MEDIA_DIR.exists():

        raise RuntimeError(
            "\nMedia directory was not found:\n"
            f"{MEDIA_DIR}\n\n"
            "Create a media/ directory beside main.py "
            "and place the panorama photographs inside it."
        )

    valid_extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".webp",
    }

    input_images = [
        path

        for path in MEDIA_DIR.iterdir()

        if (
            path.is_file()
            and path.suffix.lower()
            in valid_extensions
        )
    ]

    if len(input_images) < 2:

        raise RuntimeError(
            "\nAt least two panorama images are required.\n"
            f"Images found: {len(input_images)}\n"
            f"Directory: {MEDIA_DIR}\n"
        )

    return len(
        input_images
    )


# ============================================================
# OUTPUT PREPARATION
# ============================================================

def prepare_output_directory():
    """
    Removes output from a previous execution.

    Only the activity output/ directory is deleted.

    The following are never touched:

        media/
        pipeline/
        calibration_data/
        calibration_results/
    """

    if OUTPUT_DIR.exists():

        print(
            "Removing previous pipeline output..."
        )

        shutil.rmtree(
            OUTPUT_DIR
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "Output directory prepared:"
    )

    print(
        OUTPUT_DIR
    )

    print()


# ============================================================
# CALIBRATION HANDLING
# ============================================================

def prepare_calibration(
    calibration_mode,
):
    """
    Either:

        saved -> use camera_calibration.npz

    or:

        run -> execute calibration.py first
    """

    print_header(
        "CAMERA CALIBRATION"
    )

    # ========================================================
    # RUN CALIBRATION AGAIN
    # ========================================================

    if calibration_mode == "run":

        print(
            "Calibration mode:"
        )

        print(
            "RUN NEW CALIBRATION"
        )

        print()

        calibrate_camera()

    # ========================================================
    # USE SAVED CALIBRATION
    # ========================================================

    else:

        print(
            "Calibration mode:"
        )

        print(
            "USE SAVED CALIBRATION"
        )

        print()

        if not CALIBRATION_FILE.exists():

            raise RuntimeError(
                "\nSaved camera calibration was not found:\n"
                f"{CALIBRATION_FILE}\n\n"
                "Run:\n\n"
                "    python3 main.py --calibration run\n"
            )

    # ========================================================
    # VERIFY CALIBRATION
    # ========================================================

    calibration = (
        load_calibration()
    )

    print(
        "Calibration loaded successfully."
    )

    print()

    print(
        "Calibration file:"
    )

    print(
        CALIBRATION_FILE
    )

    print()

    print(
        "Camera matrix K:"
    )

    print(
        calibration[
            "camera_matrix"
        ]
    )

    print()

    print(
        "Distortion coefficients:"
    )

    print(
        calibration[
            "dist_coeffs"
        ]
    )

    print()

    print(
        "Calibration resolution:"
    )

    print(
        f"{calibration['image_size'][0]}"
        f" x "
        f"{calibration['image_size'][1]}"
    )

    print()

    print(
        f"Calibration RMS error: "
        f"{calibration['rms_error']:.6f} px"
    )

    print(
        f"Mean reprojection error: "
        f"{calibration['reprojection_error']:.6f} px"
    )

    print()


# ============================================================
# COMPLETE PIPELINE
# ============================================================

def run_pipeline(
    calibration_mode,
):
    """
    Runs the entire panorama pipeline automatically.

    Stages:

        1. Undistortion
        2. SIFT feature extraction
        3. Sequential image pairing
        4. SIFT feature matching
        5. RANSAC homography estimation
        6. Perspective warping
        7. Feather stitching

    Camera calibration is handled before these stages.
    """

    start_time = (
        time.perf_counter()
    )

    # ========================================================
    # VALIDATE INPUT
    # ========================================================

    number_images = (
        validate_project()
    )

    print_header(
        "2D PANORAMA PIPELINE"
    )

    print(
        "Activity directory:"
    )

    print(
        ACTIVITY_DIR
    )

    print()

    print(
        "Pipeline directory:"
    )

    print(
        PIPELINE_DIR
    )

    print()

    print(
        "Input directory:"
    )

    print(
        MEDIA_DIR
    )

    print()

    print(
        f"Input images: "
        f"{number_images}"
    )

    print()

    # ========================================================
    # CAMERA CALIBRATION
    # ========================================================

    prepare_calibration(
        calibration_mode
    )

    # ========================================================
    # PREPARE OUTPUT
    # ========================================================

    prepare_output_directory()

    total_stages = 7

    # ========================================================
    # STAGE 1
    # UNDISTORTION
    # ========================================================

    print_stage(
        1,
        total_stages,
        "UNDISTORTION",
    )

    undistorted_images = (
        undistort_directory(
            input_dir=(
                MEDIA_DIR
            ),

            output_dir=(
                UNDISTORTED_DIR
            ),

            save_comparisons=True,
        )
    )

    if len(undistorted_images) != number_images:

        raise RuntimeError(
            "\nUndistortion did not process all input images.\n"
            f"Expected: {number_images}\n"
            f"Produced: {len(undistorted_images)}\n"
        )

    # ========================================================
    # STAGE 2
    # SIFT FEATURE EXTRACTION
    # ========================================================

    print_stage(
        2,
        total_stages,
        "SIFT FEATURE EXTRACTION",
    )

    feature_results = (
        extract_features_directory(
            input_dir=(
                UNDISTORTED_DIR
            ),

            output_dir=(
                FEATURES_DIR
            ),

            max_features=(
                SIFT_MAX_FEATURES
            ),
        )
    )

    if len(feature_results) != number_images:

        raise RuntimeError(
            "\nFeature extraction did not process "
            "all images.\n"
            f"Expected: {number_images}\n"
            f"Produced: {len(feature_results)}\n"
        )

    # ========================================================
    # STAGE 3
    # IMAGE PAIRING
    # ========================================================

    print_stage(
        3,
        total_stages,
        "IMAGE MATCHING / PAIR SELECTION",
    )

    pairs = (
        create_image_pairs(
            features_dir=(
                FEATURES_DIR
            ),

            output_file=(
                PAIRS_FILE
            ),
        )
    )

    expected_pairs = (
        number_images - 1
    )

    if len(pairs) != expected_pairs:

        raise RuntimeError(
            "\nUnexpected number of sequential image pairs.\n"
            f"Expected: {expected_pairs}\n"
            f"Created: {len(pairs)}\n"
        )

    # ========================================================
    # STAGE 4
    # FEATURE MATCHING
    # ========================================================

    print_stage(
        4,
        total_stages,
        "SIFT FEATURE MATCHING",
    )

    match_results = (
        match_all_pairs(
            pair_file=(
                PAIRS_FILE
            ),

            output_dir=(
                MATCHES_DIR
            ),

            ratio_threshold=(
                LOWE_RATIO
            ),
        )
    )

    if len(match_results) != expected_pairs:

        raise RuntimeError(
            "\nFeature matching did not process "
            "all image pairs.\n"
            f"Expected: {expected_pairs}\n"
            f"Produced: {len(match_results)}\n"
        )

    # ========================================================
    # STAGE 5
    # HOMOGRAPHY + RANSAC
    # ========================================================

    print_stage(
        5,
        total_stages,
        "RANSAC HOMOGRAPHY ESTIMATION",
    )

    homography_results = (
        estimate_all_homographies(
            matches_dir=(
                MATCHES_DIR
            ),

            image_dir=(
                UNDISTORTED_DIR
            ),

            output_dir=(
                HOMOGRAPHIES_DIR
            ),

            reprojection_threshold=(
                RANSAC_THRESHOLD
            ),

            confidence=(
                RANSAC_CONFIDENCE
            ),

            max_iterations=(
                RANSAC_MAX_ITERATIONS
            ),
        )
    )

    failed_homographies = [
        result

        for result in homography_results

        if not result.get(
            "success",
            False,
        )
    ]

    if failed_homographies:

        print()
        print(
            "Failed homography pairs:"
        )

        for result in failed_homographies:

            print(
                f"  Pair "
                f"{result.get('pair_id')}: "
                f"{result.get('reason')}"
            )

        raise RuntimeError(
            "\nOne or more homographies failed.\n"
            "Warping cannot continue because the "
            "panorama image chain would be broken."
        )

    if len(
        homography_results
    ) != expected_pairs:

        raise RuntimeError(
            "\nUnexpected number of homography results.\n"
            f"Expected: {expected_pairs}\n"
            f"Produced: {len(homography_results)}\n"
        )

    # ========================================================
    # STAGE 6
    # PERSPECTIVE WARPING
    # ========================================================

    print_stage(
        6,
        total_stages,
        "PERSPECTIVE WARPING",
    )

    warped_results = (
        warp_panorama_images(
            homography_dir=(
                HOMOGRAPHIES_DIR
            ),

            image_dir=(
                UNDISTORTED_DIR
            ),

            output_dir=(
                WARPED_DIR
            ),

            # Middle image selected automatically.
            reference_index=None,
        )
    )

    if len(warped_results) != number_images:

        raise RuntimeError(
            "\nPerspective warping did not process "
            "all images.\n"
            f"Expected: {number_images}\n"
            f"Produced: {len(warped_results)}\n"
        )

    # ========================================================
    # STAGE 7
    # STITCHING
    # ========================================================

    print_stage(
        7,
        total_stages,
        "PANORAMA STITCHING",
    )

    panorama = (
        stitch_panorama(
            warped_dir=(
                WARPED_DIR
            ),

            mask_dir=(
                WARPED_MASKS_DIR
            ),

            output_file=(
                PANORAMA_FILE
            ),

            feather_power=(
                FEATHER_POWER
            ),
        )
    )

    if panorama is None:

        raise RuntimeError(
            "Stitching did not produce a panorama."
        )

    if not PANORAMA_FILE.exists():

        raise RuntimeError(
            "\nPipeline finished stitching but the "
            "panorama file was not created:\n"
            f"{PANORAMA_FILE}\n"
        )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    elapsed_time = (
        time.perf_counter()
        - start_time
    )

    panorama_height, panorama_width = (
        panorama.shape[:2]
    )

    print()
    print("=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print()

    print(
        "Pipeline:"
    )

    print(
        "Calibration"
        " -> Undistortion"
        " -> SIFT"
        " -> Image Pairing"
        " -> Feature Matching"
        " -> RANSAC"
        " -> WarpPerspective"
        " -> Stitching"
    )

    print()

    print(
        f"Input images: "
        f"{number_images}"
    )

    print(
        f"Image pairs: "
        f"{expected_pairs}"
    )

    print(
        f"Calibration mode: "
        f"{calibration_mode}"
    )

    print()

    print(
        "Final panorama resolution:"
    )

    print(
        f"{panorama_width} x "
        f"{panorama_height}"
    )

    print()

    print(
        "Final panorama:"
    )

    print(
        PANORAMA_FILE
    )

    print()

    print(
        f"Total execution time: "
        f"{elapsed_time:.2f} seconds"
    )

    print()

    print("=" * 60)
    print()


# ============================================================
# COMMAND-LINE ARGUMENTS
# ============================================================

def parse_arguments():
    """
    The only command-line option controls camera calibration.

    Default:

        python3 main.py

    uses the saved calibration.

    Explicit saved calibration:

        python3 main.py --calibration saved

    Recalibrate first:

        python3 main.py --calibration run
    """

    parser = argparse.ArgumentParser(
        description=(
            "Run the complete 2D panorama pipeline."
        )
    )

    parser.add_argument(
        "--calibration",

        choices=(
            "saved",
            "run",
        ),

        default="saved",

        help=(
            "Use the existing saved calibration "
            "('saved') or execute camera calibration "
            "again ('run'). Default: saved."
        ),
    )

    return parser.parse_args()


# ============================================================
# MAIN
# ============================================================

def main():

    args = (
        parse_arguments()
    )

    try:

        run_pipeline(
            calibration_mode=(
                args.calibration
            )
        )

    except KeyboardInterrupt:

        print()
        print("=" * 60)
        print("PIPELINE INTERRUPTED")
        print("=" * 60)
        print()

        print(
            "Execution interrupted by user."
        )

        print()

        sys.exit(130)

    except Exception:

        print()
        print("=" * 60)
        print("PIPELINE FAILED")
        print("=" * 60)
        print()

        # Full traceback is intentionally shown.
        # This makes debugging failures in any individual
        # pipeline stage much easier.
        traceback.print_exc()

        print()

        print(
            "Partial results, if any, remain in:"
        )

        print(
            OUTPUT_DIR
        )

        print()

        sys.exit(1)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()