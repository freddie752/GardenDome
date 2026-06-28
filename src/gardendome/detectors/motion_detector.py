import numpy as np
import cv2


class MotionDetector:
    def __init__(self, motion_threshold, motion_fraction):
        self._motion_threshold = motion_threshold
        self._motion_fraction = motion_fraction

    def detect(self, previous_frame: np.ndarray, current_frame: np.ndarray):
        if previous_frame.ndim != 2 or current_frame.ndim != 2:
            raise ValueError("Expected greyscale (2D) frames")
        diff_frame = cv2.absdiff(src1=previous_frame, src2=current_frame)
        motion_mask = diff_frame > self._motion_threshold
        motion_percentage = np.sum(motion_mask) / motion_mask.size
        return bool(motion_percentage > self._motion_fraction)
