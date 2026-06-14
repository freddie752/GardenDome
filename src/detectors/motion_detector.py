import numpy as np
import cv2


class MotionDetector:
    def __init__(self, motion_threshold, motion_fraction):
        self._motion_threshold = motion_threshold
        self._motion_fraction = motion_fraction

    def detect(self, previous_frame: np.ndarray, current_frame: np.ndarray):
        """Detects motion by comparing pixel differences of two frames."""

        # Compute absolute difference between two frames
        diff_frame = cv2.absdiff(src1=previous_frame, src2=current_frame)

        # Calculate pixels where difference is above threshold
        motion_mask = (diff_frame > self._motion_threshold).astype(np.uint8) * 255

        # Calculate the percentage of changed pixels
        motion_percentage = np.sum(motion_mask) / (motion_mask.shape[0] * motion_mask.shape[1] * 255)
        
        # Return True if more than 1% of pixels changed
        return motion_percentage > self._motion_fraction  
