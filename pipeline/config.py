from pathlib import Path


PIPELINE_DIR = Path(__file__).resolve().parent

CALIBRATION_IMAGES_DIR = (
    PIPELINE_DIR / "calibration_data"
)

CALIBRATION_RESULTS_DIR = (
    PIPELINE_DIR / "calibration_results"
)

CALIBRATION_DEBUG_DIR = (
    PIPELINE_DIR / "debug" / "calibration"
)

CALIBRATION_FILE = (
    CALIBRATION_RESULTS_DIR
    / "camera_calibration.npz"
)

CALIBRATION_COMPARISON_FILE = (
    CALIBRATION_RESULTS_DIR
    / "calibration_comparison.txt"
)


# 8 x 6 checkerboard squares
# -> 7 x 5 INTERNAL corners
CHECKERBOARD_SIZE = (7, 5)

# 3 cm = 0.03 m
SQUARE_SIZE = 0.03