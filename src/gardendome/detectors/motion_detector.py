import numpy as np
import cv2


class MotionDetector:
    def __init__(self, motion_threshold, motion_fraction):
        self._motion_threshold = motion_threshold
        self._motion_fraction = motion_fraction

    def has_motion(self, previous_frame, current_frame):
        motion_mask = self._calculate_motion_mask(previous_frame, current_frame)
        return self._exceeds_motion_fraction(motion_mask)

    def locate_motion(self, previous_frame, current_frame):
        motion_mask = self._calculate_motion_mask(previous_frame, current_frame)
        if not self._exceeds_motion_fraction(motion_mask):
            return None
        contours, _ = cv2.findContours(
            motion_mask.astype(np.uint8) * 255,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        largest = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest)
        return x, y, w, h

    def _calculate_motion_mask(self, previous_frame, current_frame):
        if previous_frame.ndim != 2 or current_frame.ndim != 2:
            raise ValueError("Expected greyscale (2D) frames")
        diff_frame = cv2.absdiff(src1=previous_frame, src2=current_frame)
        return diff_frame > self._motion_threshold

    def _exceeds_motion_fraction(self, motion_mask):
        motion_percentage = np.sum(motion_mask) / motion_mask.size
        return bool(motion_percentage > self._motion_fraction)
