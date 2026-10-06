import cv2
import numpy as np

from config import (
    CALIBRATION_IMAGES_DIR,
    CALIBRATION_RESULTS_DIR,
    CALIBRATION_DEBUG_DIR,
    CALIBRATION_FILE,
    CALIBRATION_COMPARISON_FILE,
    CHECKERBOARD_SIZE,
    SQUARE_SIZE,
)


# ============================================================
# CHECKERBOARD GEOMETRY
# ============================================================

def create_object_points():
    """
    Creates the known 3D checkerboard coordinates.

    Because the calibration board is planar:

        Z = 0

    Example with square size = 0.03 m:

        (0.00, 0.00, 0)
        (0.03, 0.00, 0)
        (0.06, 0.00, 0)
        ...
    """

    cols, rows = CHECKERBOARD_SIZE

    points = np.zeros(
        (rows * cols, 3),
        dtype=np.float64,
    )

    points[:, :2] = (
        np.mgrid[0:cols, 0:rows]
        .T
        .reshape(-1, 2)
    )

    points *= SQUARE_SIZE

    return points


# ============================================================
# IMAGE SEARCH
# ============================================================

def find_calibration_images():
    """
    Finds calibration images inside calibration_data.
    """

    valid_extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".webp",
    }

    images = []

    for path in CALIBRATION_IMAGES_DIR.iterdir():

        if (
            path.is_file()
            and path.suffix.lower() in valid_extensions
        ):
            images.append(path)

    return sorted(images)


# ============================================================
# POINT NORMALIZATION FOR DLT
# ============================================================

def normalize_points_2d(points):
    """
    Normalizes 2D points so that:

        centroid -> (0, 0)

    and the average distance from the origin is sqrt(2).

    This improves numerical stability of the DLT homography.
    """

    points = np.asarray(
        points,
        dtype=np.float64,
    )

    centroid = np.mean(
        points,
        axis=0,
    )

    shifted = (
        points - centroid
    )

    distances = np.linalg.norm(
        shifted,
        axis=1,
    )

    mean_distance = np.mean(
        distances
    )

    if mean_distance < 1e-12:
        raise RuntimeError(
            "Degenerate point configuration."
        )

    scale = (
        np.sqrt(2.0)
        / mean_distance
    )

    T = np.array(
        [
            [
                scale,
                0.0,
                -scale * centroid[0],
            ],
            [
                0.0,
                scale,
                -scale * centroid[1],
            ],
            [
                0.0,
                0.0,
                1.0,
            ],
        ],
        dtype=np.float64,
    )

    homogeneous = np.column_stack(
        (
            points,
            np.ones(len(points)),
        )
    )

    normalized_h = (
        T @ homogeneous.T
    ).T

    normalized = (
        normalized_h[:, :2]
        / normalized_h[:, 2:3]
    )

    return normalized, T


# ============================================================
# HOMOGRAPHY USING OUR OWN DLT + SVD
# ============================================================

def estimate_homography_dlt(
    plane_points,
    image_points,
):
    """
    Estimates:

        image_point ~ H * plane_point

    using normalized DLT + SVD.

    This does NOT use cv2.findHomography().
    """

    plane_points = np.asarray(
        plane_points,
        dtype=np.float64,
    )

    image_points = np.asarray(
        image_points,
        dtype=np.float64,
    )

    plane_norm, T_plane = (
        normalize_points_2d(
            plane_points
        )
    )

    image_norm, T_image = (
        normalize_points_2d(
            image_points
        )
    )

    A = []

    for (
        (X, Y),
        (u, v),
    ) in zip(
        plane_norm,
        image_norm,
    ):

        A.append(
            [
                -X,
                -Y,
                -1.0,
                0.0,
                0.0,
                0.0,
                u * X,
                u * Y,
                u,
            ]
        )

        A.append(
            [
                0.0,
                0.0,
                0.0,
                -X,
                -Y,
                -1.0,
                v * X,
                v * Y,
                v,
            ]
        )

    A = np.asarray(
        A,
        dtype=np.float64,
    )

    _, _, Vt = np.linalg.svd(A)

    h = Vt[-1]

    H_normalized = h.reshape(
        3,
        3,
    )

    H = (
        np.linalg.inv(T_image)
        @ H_normalized
        @ T_plane
    )

    if abs(H[2, 2]) > 1e-12:

        H /= H[2, 2]

    else:

        H /= np.linalg.norm(H)

    return H


# ============================================================
# ZHANG CONSTRAINT VECTOR
# ============================================================

def zhang_v(H, i, j):
    """
    Creates Zhang's v_ij constraint vector.
    """

    hi = H[:, i]
    hj = H[:, j]

    return np.array(
        [
            hi[0] * hj[0],

            hi[0] * hj[1]
            + hi[1] * hj[0],

            hi[1] * hj[1],

            hi[2] * hj[0]
            + hi[0] * hj[2],

            hi[2] * hj[1]
            + hi[1] * hj[2],

            hi[2] * hj[2],
        ],
        dtype=np.float64,
    )


# ============================================================
# OUR ZHANG INTRINSIC CALIBRATION
# ============================================================

def estimate_intrinsics_zhang(
    homographies,
):
    """
    Computes the intrinsic matrix K using Zhang's method.

    The system uses:

        v12^T b = 0

        (v11 - v22)^T b = 0

    for every homography.

    B is defined as:

        B = K^-T K^-1
    """

    V = []

    for H in homographies:

        v12 = zhang_v(
            H,
            0,
            1,
        )

        v11 = zhang_v(
            H,
            0,
            0,
        )

        v22 = zhang_v(
            H,
            1,
            1,
        )

        V.append(v12)

        V.append(
            v11 - v22
        )

    V = np.asarray(
        V,
        dtype=np.float64,
    )

    _, _, Vt = np.linalg.svd(V)

    b = Vt[-1]

    B = np.array(
        [
            [
                b[0],
                b[1],
                b[3],
            ],
            [
                b[1],
                b[2],
                b[4],
            ],
            [
                b[3],
                b[4],
                b[5],
            ],
        ],
        dtype=np.float64,
    )

    B = (
        B + B.T
    ) / 2.0

    eigenvalues = np.linalg.eigvalsh(B)

    # SVD may return the solution with opposite sign.
    if np.all(eigenvalues < 0):

        B = -B

        eigenvalues = (
            -eigenvalues
        )

    # Numerical protection.
    if np.min(eigenvalues) <= 0:

        eigvals, eigvecs = np.linalg.eigh(B)

        epsilon = (
            np.max(np.abs(eigvals))
            * 1e-10
        )

        eigvals = np.maximum(
            eigvals,
            epsilon,
        )

        B = (
            eigvecs
            @ np.diag(eigvals)
            @ eigvecs.T
        )

    # B = K^-T K^-1
    #
    # Cholesky:
    #
    # B = L L^T
    #
    # L = K^-T
    #
    # K = L^-T

    L = np.linalg.cholesky(B)

    K = np.linalg.inv(L).T

    K /= K[2, 2]

    K[
        np.abs(K) < 1e-10
    ] = 0.0

    return K, B


# ============================================================
# CAMERA POSE FROM HOMOGRAPHY
# ============================================================

def estimate_pose_from_homography(
    H,
    K,
):
    """
    Recovers initial camera pose from:

        H = K [r1 r2 t]
    """

    K_inv = np.linalg.inv(K)

    h1 = H[:, 0]
    h2 = H[:, 1]
    h3 = H[:, 2]

    k_h1 = K_inv @ h1
    k_h2 = K_inv @ h2
    k_h3 = K_inv @ h3

    scale = (
        2.0
        / (
            np.linalg.norm(k_h1)
            + np.linalg.norm(k_h2)
        )
    )

    r1 = scale * k_h1
    r2 = scale * k_h2
    t = scale * k_h3

    # Homography sign ambiguity.
    if t[2] < 0:

        r1 = -r1
        r2 = -r2
        t = -t

    r3 = np.cross(
        r1,
        r2,
    )

    R_approx = np.column_stack(
        (
            r1,
            r2,
            r3,
        )
    )

    # Force an orthonormal rotation matrix.
    U, _, Vt = np.linalg.svd(
        R_approx
    )

    R = U @ Vt

    if np.linalg.det(R) < 0:

        U[:, -1] *= -1

        R = U @ Vt

    return R, t


# ============================================================
# OUR RADIAL DISTORTION ESTIMATION
# ============================================================

def estimate_radial_distortion(
    object_points_list,
    image_points_list,
    rotations,
    translations,
    K,
):
    """
    Estimates only:

        k1
        k2

    using linear least squares.
    """

    fx = K[0, 0]
    skew = K[0, 1]
    fy = K[1, 1]

    cx = K[0, 2]
    cy = K[1, 2]

    A = []
    b = []

    for (
        object_points,
        image_points,
        R,
        t,
    ) in zip(
        object_points_list,
        image_points_list,
        rotations,
        translations,
    ):

        camera_points = (
            R @ object_points.T
            + t.reshape(3, 1)
        ).T

        X = camera_points[:, 0]
        Y = camera_points[:, 1]
        Z = camera_points[:, 2]

        x = X / Z
        y = Y / Z

        r2 = (
            x * x
            + y * y
        )

        r4 = r2 * r2

        u0 = (
            fx * x
            + skew * y
            + cx
        )

        v0 = (
            fy * y
            + cy
        )

        observed = (
            image_points
            .reshape(-1, 2)
        )

        u_obs = observed[:, 0]
        v_obs = observed[:, 1]

        u_factor = (
            u0 - cx
        )

        v_factor = (
            v0 - cy
        )

        for index in range(
            len(object_points)
        ):

            A.append(
                [
                    u_factor[index]
                    * r2[index],

                    u_factor[index]
                    * r4[index],
                ]
            )

            b.append(
                u_obs[index]
                - u0[index]
            )

            A.append(
                [
                    v_factor[index]
                    * r2[index],

                    v_factor[index]
                    * r4[index],
                ]
            )

            b.append(
                v_obs[index]
                - v0[index]
            )

    A = np.asarray(
        A,
        dtype=np.float64,
    )

    b = np.asarray(
        b,
        dtype=np.float64,
    )

    distortion, _, _, _ = (
        np.linalg.lstsq(
            A,
            b,
            rcond=None,
        )
    )

    return (
        distortion[0],
        distortion[1],
    )


# ============================================================
# MANUAL PROJECTION
# ============================================================

def project_points_manual(
    object_points,
    R,
    t,
    K,
    k1,
    k2,
):
    """
    Projects 3D checkerboard points using our
    manually estimated camera parameters.
    """

    camera_points = (
        R @ object_points.T
        + t.reshape(3, 1)
    ).T

    X = camera_points[:, 0]
    Y = camera_points[:, 1]
    Z = camera_points[:, 2]

    x = X / Z
    y = Y / Z

    r2 = (
        x * x
        + y * y
    )

    radial = (
        1.0
        + k1 * r2
        + k2 * r2 * r2
    )

    x_distorted = (
        x * radial
    )

    y_distorted = (
        y * radial
    )

    fx = K[0, 0]
    skew = K[0, 1]
    fy = K[1, 1]

    cx = K[0, 2]
    cy = K[1, 2]

    u = (
        fx * x_distorted
        + skew * y_distorted
        + cx
    )

    v = (
        fy * y_distorted
        + cy
    )

    return np.column_stack(
        (
            u,
            v,
        )
    )


# ============================================================
# MANUAL REPROJECTION ERROR
# ============================================================

def calculate_manual_errors(
    object_points_list,
    image_points_list,
    rotations,
    translations,
    K,
    k1,
    k2,
):
    """
    Returns:

        RMS reprojection error
        mean Euclidean reprojection error
    """

    all_errors = []

    for (
        object_points,
        observed_points,
        R,
        t,
    ) in zip(
        object_points_list,
        image_points_list,
        rotations,
        translations,
    ):

        projected = (
            project_points_manual(
                object_points,
                R,
                t,
                K,
                k1,
                k2,
            )
        )

        observed = (
            observed_points
            .reshape(-1, 2)
        )

        errors = np.linalg.norm(
            observed - projected,
            axis=1,
        )

        all_errors.extend(
            errors
        )

    all_errors = np.asarray(
        all_errors,
        dtype=np.float64,
    )

    rms_error = np.sqrt(
        np.mean(
            all_errors ** 2
        )
    )

    mean_error = np.mean(
        all_errors
    )

    return (
        rms_error,
        mean_error,
    )


# ============================================================
# OPENCV REPROJECTION ERROR
# ============================================================

def calculate_opencv_errors(
    object_points_list,
    image_points_list,
    rvecs,
    tvecs,
    K,
    dist_coeffs,
):
    """
    Independently calculates OpenCV reprojection error.
    """

    all_errors = []

    for i in range(
        len(object_points_list)
    ):

        projected, _ = (
            cv2.projectPoints(
                object_points_list[i],
                rvecs[i],
                tvecs[i],
                K,
                dist_coeffs,
            )
        )

        projected = (
            projected.reshape(-1, 2)
        )

        observed = (
            image_points_list[i]
            .reshape(-1, 2)
        )

        errors = np.linalg.norm(
            observed - projected,
            axis=1,
        )

        all_errors.extend(
            errors
        )

    all_errors = np.asarray(
        all_errors,
        dtype=np.float64,
    )

    rms_error = np.sqrt(
        np.mean(
            all_errors ** 2
        )
    )

    mean_error = np.mean(
        all_errors
    )

    return (
        rms_error,
        mean_error,
    )


# ============================================================
# COMPARISON UTILITY
# ============================================================

def percentage_difference(
    manual_value,
    opencv_value,
):
    if abs(opencv_value) < 1e-12:
        return 0.0

    return (
        abs(
            manual_value
            - opencv_value
        )
        / abs(opencv_value)
        * 100.0
    )


# ============================================================
# LOAD OFFICIAL CALIBRATION
# ============================================================

def load_calibration():
    """
    Loads the OFFICIAL calibration used by
    the rest of the computer-vision pipeline.

    These generic values point to OpenCV's result.
    """

    data = np.load(
        CALIBRATION_FILE
    )

    return {
        "camera_matrix": (
            data["camera_matrix"]
        ),

        "dist_coeffs": (
            data["dist_coeffs"]
        ),

        "image_size": (
            int(data["image_width"]),
            int(data["image_height"]),
        ),

        "rms_error": float(
            data["rms_error"]
        ),

        "reprojection_error": float(
            data["reprojection_error"]
        ),
    }


# ============================================================
# MAIN CALIBRATION FUNCTION
# ============================================================

def calibrate_camera():

    CALIBRATION_RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    CALIBRATION_DEBUG_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    image_paths = (
        find_calibration_images()
    )

    if not image_paths:

        raise RuntimeError(
            "\nNo calibration images found in:\n"
            f"{CALIBRATION_IMAGES_DIR}\n"
        )

    print()
    print("========================================")
    print(" CAMERA CALIBRATION")
    print("========================================")
    print()

    print(
        f"Images found: "
        f"{len(image_paths)}"
    )

    print(
        f"Checkerboard inner corners: "
        f"{CHECKERBOARD_SIZE}"
    )

    print(
        f"Square size: "
        f"{SQUARE_SIZE} m"
    )

    print()

    object_template = (
        create_object_points()
    )

    plane_template = (
        object_template[:, :2]
    )

    all_object_points = []
    all_image_points = []

    homographies = []

    successful_names = []

    image_size = None

    criteria = (
        cv2.TERM_CRITERIA_EPS
        + cv2.TERM_CRITERIA_MAX_ITER,
        30,
        0.001,
    )

    # ========================================================
    # CORNER DETECTION
    # ========================================================

    for image_path in image_paths:

        print(
            f"Processing: "
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

        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY,
        )

        current_size = (
            gray.shape[1],
            gray.shape[0],
        )

        if image_size is None:

            image_size = current_size

            print(
                f"  Resolution: "
                f"{image_size[0]} x "
                f"{image_size[1]}"
            )

        elif current_size != image_size:

            raise RuntimeError(
                "\nCalibration images have "
                "different resolutions.\n"
                f"Expected: {image_size}\n"
                f"Found: {current_size}\n"
                f"Image: {image_path.name}\n"
            )

        found, corners = (
            cv2.findChessboardCorners(
                gray,
                CHECKERBOARD_SIZE,
                None,
            )
        )

        if not found:

            print(
                "  Checkerboard NOT detected"
            )

            continue

        refined = cv2.cornerSubPix(
            gray,
            corners,
            (11, 11),
            (-1, -1),
            criteria,
        )

        image_points = (
            refined
            .reshape(-1, 2)
            .astype(np.float64)
        )

        print(
            "  Checkerboard detected"
        )

        # ----------------------------------------------------
        # Manual homography
        # ----------------------------------------------------

        H = estimate_homography_dlt(
            plane_template,
            image_points,
        )

        homographies.append(H)

        all_object_points.append(
            object_template.copy()
        )

        all_image_points.append(
            refined.astype(
                np.float64
            )
        )

        successful_names.append(
            image_path.name
        )

        # ----------------------------------------------------
        # Debug visualization
        # ----------------------------------------------------

        debug_image = image.copy()

        cv2.drawChessboardCorners(
            debug_image,
            CHECKERBOARD_SIZE,
            refined,
            found,
        )

        cv2.imwrite(
            str(
                CALIBRATION_DEBUG_DIR
                / image_path.name
            ),
            debug_image,
        )

    number_successful = len(
        all_object_points
    )

    print()
    print(
        f"Successful calibration images: "
        f"{number_successful}/"
        f"{len(image_paths)}"
    )

    if number_successful < 3:

        raise RuntimeError(
            "Too few valid calibration images."
        )

    # ========================================================
    # OUR ZHANG CALIBRATION
    # ========================================================

    print()
    print("========================================")
    print(" OUR ZHANG CALIBRATION")
    print("========================================")
    print()

    manual_K, manual_B = (
        estimate_intrinsics_zhang(
            homographies
        )
    )

    manual_rotations = []
    manual_translations = []

    for H in homographies:

        R, t = (
            estimate_pose_from_homography(
                H,
                manual_K,
            )
        )

        manual_rotations.append(R)

        manual_translations.append(t)

    manual_k1, manual_k2 = (
        estimate_radial_distortion(
            all_object_points,
            all_image_points,
            manual_rotations,
            manual_translations,
            manual_K,
        )
    )

    (
        manual_rms,
        manual_mean_error,
    ) = calculate_manual_errors(
        all_object_points,
        all_image_points,
        manual_rotations,
        manual_translations,
        manual_K,
        manual_k1,
        manual_k2,
    )

    manual_dist_coeffs = np.array(
        [
            [
                manual_k1,
                manual_k2,
                0.0,
                0.0,
                0.0,
            ]
        ],
        dtype=np.float64,
    )

    print(
        "Manual camera matrix K:"
    )

    print(manual_K)

    print()

    print(
        "Manual radial distortion:"
    )

    print(
        f"k1 = {manual_k1:.10f}"
    )

    print(
        f"k2 = {manual_k2:.10f}"
    )

    print()

    print(
        f"Manual RMS reprojection error: "
        f"{manual_rms:.6f} pixels"
    )

    print(
        f"Manual mean reprojection error: "
        f"{manual_mean_error:.6f} pixels"
    )

    # ========================================================
    # OPENCV CALIBRATION
    # ========================================================

    print()
    print("========================================")
    print(" OPENCV CALIBRATION")
    print("========================================")
    print()

    # OpenCV wants Point3f / Point2f here.
    opencv_object_points = [
        points.astype(np.float32)
        for points in all_object_points
    ]

    opencv_image_points = [
        points.astype(np.float32)
        for points in all_image_points
    ]

    # Same distortion model used by our manual result:
    #
    # k1 = estimated
    # k2 = estimated
    # p1 = 0
    # p2 = 0
    # k3 = 0

    flags = (
        cv2.CALIB_ZERO_TANGENT_DIST
        | cv2.CALIB_FIX_K3
    )

    (
        opencv_returned_rms,
        opencv_K,
        opencv_dist,
        opencv_rvecs,
        opencv_tvecs,
    ) = cv2.calibrateCamera(
        opencv_object_points,
        opencv_image_points,
        image_size,
        None,
        None,
        flags=flags,
    )

    (
        opencv_rms,
        opencv_mean_error,
    ) = calculate_opencv_errors(
        opencv_object_points,
        opencv_image_points,
        opencv_rvecs,
        opencv_tvecs,
        opencv_K,
        opencv_dist,
    )

    opencv_dist_flat = (
        opencv_dist.flatten()
    )

    opencv_k1 = (
        opencv_dist_flat[0]
    )

    opencv_k2 = (
        opencv_dist_flat[1]
    )

    print(
        "OpenCV camera matrix K:"
    )

    print(opencv_K)

    print()

    print(
        "OpenCV distortion coefficients:"
    )

    print(opencv_dist)

    print()

    print(
        f"k1 = {opencv_k1:.10f}"
    )

    print(
        f"k2 = {opencv_k2:.10f}"
    )

    print(
        "p1 = 0"
    )

    print(
        "p2 = 0"
    )

    print(
        "k3 = 0"
    )

    print()

    print(
        f"OpenCV returned RMS: "
        f"{opencv_returned_rms:.6f} pixels"
    )

    print(
        f"OpenCV independently calculated RMS: "
        f"{opencv_rms:.6f} pixels"
    )

    print(
        f"OpenCV mean reprojection error: "
        f"{opencv_mean_error:.6f} pixels"
    )

    # ========================================================
    # COMPARISON
    # ========================================================

    print()
    print("========================================")
    print(" CALIBRATION COMPARISON")
    print("========================================")
    print()

    comparison_lines = []

    comparison_lines.append(
        "CAMERA CALIBRATION COMPARISON"
    )

    comparison_lines.append(
        "=" * 70
    )

    comparison_lines.append("")

    comparison_lines.append(
        f"Resolution: "
        f"{image_size[0]} x "
        f"{image_size[1]}"
    )

    comparison_lines.append(
        f"Images used: "
        f"{number_successful}"
    )

    comparison_lines.append("")

    header = (
        f"{'Parameter':<12}"
        f"{'Manual Zhang':>18}"
        f"{'OpenCV':>18}"
        f"{'Difference %':>18}"
    )

    print(header)

    print(
        "-" * len(header)
    )

    comparison_lines.append(header)

    comparison_lines.append(
        "-" * len(header)
    )

    parameters = (
        (
            "fx",
            manual_K[0, 0],
            opencv_K[0, 0],
        ),
        (
            "fy",
            manual_K[1, 1],
            opencv_K[1, 1],
        ),
        (
            "cx",
            manual_K[0, 2],
            opencv_K[0, 2],
        ),
        (
            "cy",
            manual_K[1, 2],
            opencv_K[1, 2],
        ),
        (
            "k1",
            manual_k1,
            opencv_k1,
        ),
        (
            "k2",
            manual_k2,
            opencv_k2,
        ),
    )

    for (
        name,
        manual_value,
        opencv_value,
    ) in parameters:

        difference = (
            percentage_difference(
                manual_value,
                opencv_value,
            )
        )

        line = (
            f"{name:<12}"
            f"{manual_value:>18.6f}"
            f"{opencv_value:>18.6f}"
            f"{difference:>17.3f}%"
        )

        print(line)

        comparison_lines.append(
            line
        )

    print()

    comparison_lines.append("")

    error_header = (
        f"{'Error':<30}"
        f"{'Manual':>18}"
        f"{'OpenCV':>18}"
    )

    print(error_header)

    print(
        "-" * len(error_header)
    )

    comparison_lines.append(
        error_header
    )

    comparison_lines.append(
        "-" * len(error_header)
    )

    rms_line = (
        f"{'RMS reprojection [px]':<30}"
        f"{manual_rms:>18.6f}"
        f"{opencv_rms:>18.6f}"
    )

    mean_line = (
        f"{'Mean reprojection [px]':<30}"
        f"{manual_mean_error:>18.6f}"
        f"{opencv_mean_error:>18.6f}"
    )

    print(rms_line)

    print(mean_line)

    comparison_lines.append(
        rms_line
    )

    comparison_lines.append(
        mean_line
    )

    # ========================================================
    # SAVE EVERYTHING
    # ========================================================

    np.savez(
        CALIBRATION_FILE,

        # ====================================================
        # OFFICIAL PIPELINE CALIBRATION
        #
        # These generic names are what ALL later modules
        # should use.
        #
        # They point to OpenCV's calibration.
        # ====================================================

        camera_matrix=opencv_K,

        dist_coeffs=opencv_dist,

        rms_error=opencv_rms,

        reprojection_error=(
            opencv_mean_error
        ),

        calibration_method=np.array(
            "opencv_k1_k2"
        ),

        # ====================================================
        # GENERAL INFORMATION
        # ====================================================

        image_width=image_size[0],

        image_height=image_size[1],

        checkerboard_cols=(
            CHECKERBOARD_SIZE[0]
        ),

        checkerboard_rows=(
            CHECKERBOARD_SIZE[1]
        ),

        square_size=(
            SQUARE_SIZE
        ),

        successful_images=np.asarray(
            successful_names
        ),

        # ====================================================
        # MANUAL ZHANG RESULTS
        # ====================================================

        manual_camera_matrix=(
            manual_K
        ),

        manual_B_matrix=(
            manual_B
        ),

        manual_dist_coeffs=(
            manual_dist_coeffs
        ),

        manual_rms_error=(
            manual_rms
        ),

        manual_mean_error=(
            manual_mean_error
        ),

        manual_rotations=np.asarray(
            manual_rotations
        ),

        manual_translations=np.asarray(
            manual_translations
        ),

        # ====================================================
        # OPENCV RESULTS
        # ====================================================

        opencv_camera_matrix=(
            opencv_K
        ),

        opencv_dist_coeffs=(
            opencv_dist
        ),

        opencv_rms_error=(
            opencv_rms
        ),

        opencv_mean_error=(
            opencv_mean_error
        ),

        opencv_rvecs=np.asarray(
            opencv_rvecs
        ),

        opencv_tvecs=np.asarray(
            opencv_tvecs
        ),
    )

    # ========================================================
    # SAVE TEXT COMPARISON
    # ========================================================

    CALIBRATION_COMPARISON_FILE.write_text(
        "\n".join(
            comparison_lines
        )
        + "\n",
        encoding="utf-8",
    )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print()
    print("========================================")
    print(" OFFICIAL PIPELINE CALIBRATION")
    print("========================================")
    print()

    print(
        "Calibration method:"
    )

    print(
        "OpenCV calibration "
        "(k1 + k2 distortion model)"
    )

    print()

    print(
        "Official camera matrix K:"
    )

    print(opencv_K)

    print()

    print(
        "Official distortion coefficients:"
    )

    print(opencv_dist)

    print()

    print(
        f"Official RMS error: "
        f"{opencv_rms:.6f} pixels"
    )

    print(
        f"Official mean reprojection error: "
        f"{opencv_mean_error:.6f} pixels"
    )

    print()

    print(
        "Calibration saved to:"
    )

    print(
        CALIBRATION_FILE
    )

    print()

    print(
        "Comparison report saved to:"
    )

    print(
        CALIBRATION_COMPARISON_FILE
    )

    print()

    print(
        "Debug checkerboard images saved to:"
    )

    print(
        CALIBRATION_DEBUG_DIR
    )

    print()

    print("========================================")
    print(" CALIBRATION COMPLETE")
    print("========================================")
    print()

    return {
        "camera_matrix": (
            opencv_K
        ),

        "dist_coeffs": (
            opencv_dist
        ),

        "image_size": (
            image_size
        ),

        "rms_error": (
            opencv_rms
        ),

        "reprojection_error": (
            opencv_mean_error
        ),

        "manual": {
            "camera_matrix": (
                manual_K
            ),

            "dist_coeffs": (
                manual_dist_coeffs
            ),

            "rms_error": (
                manual_rms
            ),

            "mean_error": (
                manual_mean_error
            ),
        },

        "opencv": {
            "camera_matrix": (
                opencv_K
            ),

            "dist_coeffs": (
                opencv_dist
            ),

            "rms_error": (
                opencv_rms
            ),

            "mean_error": (
                opencv_mean_error
            ),
        },
    }


# ============================================================
# RUN DIRECTLY
# ============================================================

if __name__ == "__main__":

    calibrate_camera()