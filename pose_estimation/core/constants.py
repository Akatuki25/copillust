"""COCO17 keypoint definitions, skeleton, and color maps."""

# COCO17 keypoint names in canonical order (index 0–16).
COCO17_KEYPOINTS: list[str] = [
    "nose",            # 0
    "left_eye",        # 1
    "right_eye",       # 2
    "left_ear",        # 3
    "right_ear",       # 4
    "left_shoulder",   # 5
    "right_shoulder",  # 6
    "left_elbow",      # 7
    "right_elbow",     # 8
    "left_wrist",      # 9
    "right_wrist",     # 10
    "left_hip",        # 11
    "right_hip",       # 12
    "left_knee",       # 13
    "right_knee",      # 14
    "left_ankle",      # 15
    "right_ankle",     # 16
]

# Name → index lookup.
COCO17_KEYPOINT_INDEX: dict[str, int] = {
    name: idx for idx, name in enumerate(COCO17_KEYPOINTS)
}

# Number of keypoints.
NUM_KEYPOINTS: int = 17

# Face keypoint indices (0–4).
FACE_INDICES: list[int] = [0, 1, 2, 3, 4]

# Upper-body keypoint indices.
UPPER_BODY_INDICES: list[int] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# Lower-body keypoint indices.
LOWER_BODY_INDICES: list[int] = [11, 12, 13, 14, 15, 16]

# COCO skeleton: pairs of keypoint indices that form limb connections.
# Each tuple is (start_index, end_index), 1-indexed in COCO but 0-indexed here.
COCO17_SKELETON: list[tuple[int, int]] = [
    (0, 1),    # nose – left_eye
    (0, 2),    # nose – right_eye
    (1, 3),    # left_eye – left_ear
    (2, 4),    # right_eye – right_ear
    (5, 6),    # left_shoulder – right_shoulder
    (5, 7),    # left_shoulder – left_elbow
    (7, 9),    # left_elbow – left_wrist
    (6, 8),    # right_shoulder – right_elbow
    (8, 10),   # right_elbow – right_wrist
    (5, 11),   # left_shoulder – left_hip
    (6, 12),   # right_shoulder – right_hip
    (11, 12),  # left_hip – right_hip
    (11, 13),  # left_hip – left_knee
    (13, 15),  # left_knee – left_ankle
    (12, 14),  # right_hip – right_knee
    (14, 16),  # right_knee – right_ankle
]

# Left-right swap pairs for flip augmentation and left-right confusion detection.
COCO17_SWAP_PAIRS: list[tuple[int, int]] = [
    (1, 2),    # left_eye – right_eye
    (3, 4),    # left_ear – right_ear
    (5, 6),    # left_shoulder – right_shoulder
    (7, 8),    # left_elbow – right_elbow
    (9, 10),   # left_wrist – right_wrist
    (11, 12),  # left_hip – right_hip
    (13, 14),  # left_knee – right_knee
    (15, 16),  # left_ankle – right_ankle
]

# OKS sigmas for COCO17 (used in AP computation).
COCO17_SIGMAS: list[float] = [
    0.026,  # nose
    0.025,  # left_eye
    0.025,  # right_eye
    0.035,  # left_ear
    0.035,  # right_ear
    0.079,  # left_shoulder
    0.079,  # right_shoulder
    0.072,  # left_elbow
    0.072,  # right_elbow
    0.062,  # left_wrist
    0.062,  # right_wrist
    0.107,  # left_hip
    0.107,  # right_hip
    0.087,  # left_knee
    0.087,  # right_knee
    0.089,  # left_ankle
    0.089,  # right_ankle
]

# Colors for skeleton visualization (RGB).
SKELETON_COLORS: list[tuple[int, int, int]] = [
    (255, 0, 0),      # nose – left_eye
    (255, 0, 0),      # nose – right_eye
    (255, 85, 0),     # left_eye – left_ear
    (255, 85, 0),     # right_eye – right_ear
    (0, 255, 0),      # left_shoulder – right_shoulder
    (0, 255, 85),     # left_shoulder – left_elbow
    (0, 255, 170),    # left_elbow – left_wrist
    (0, 170, 255),    # right_shoulder – right_elbow
    (0, 85, 255),     # right_elbow – right_wrist
    (255, 170, 0),    # left_shoulder – left_hip
    (170, 255, 0),    # right_shoulder – right_hip
    (85, 255, 0),     # left_hip – right_hip
    (255, 255, 0),    # left_hip – left_knee
    (255, 170, 0),    # left_knee – left_ankle
    (170, 0, 255),    # right_hip – right_knee
    (85, 0, 255),     # right_knee – right_ankle
]

# Keypoint colors for visualization (RGB).
KEYPOINT_COLORS: list[tuple[int, int, int]] = [
    (255, 0, 0),      # nose
    (255, 85, 0),     # left_eye
    (255, 170, 0),    # right_eye
    (255, 255, 0),    # left_ear
    (170, 255, 0),    # right_ear
    (85, 255, 0),     # left_shoulder
    (0, 255, 0),      # right_shoulder
    (0, 255, 85),     # left_elbow
    (0, 255, 170),    # right_elbow
    (0, 255, 255),    # left_wrist
    (0, 170, 255),    # right_wrist
    (0, 85, 255),     # left_hip
    (0, 0, 255),      # right_hip
    (85, 0, 255),     # left_knee
    (170, 0, 255),    # right_knee
    (255, 0, 255),    # left_ankle
    (255, 0, 170),    # right_ankle
]
