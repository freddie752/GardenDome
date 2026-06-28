import cv2
import numpy as np


class Picamera2Feed:
    def __init__(self, camera):
        self._camera = camera

    def get_frame(self) -> np.ndarray:
        img_rgb = self._camera.capture_array()
        frame = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        frame = cv2.GaussianBlur(src=frame, ksize=(5, 5), sigmaX=0)

        return frame


# TODO: Options to get frome live feed or pre-recorded files (images and videos?)
