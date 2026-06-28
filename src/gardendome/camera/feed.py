import cv2


class Picamera2Feed:
    def __init__(self, camera):
        self._camera = camera

    def get_grey_frame(self):
        rgb_frame = self._get_frame()
        grey_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2GRAY)
        blurred_frame = cv2.GaussianBlur(src=grey_frame, ksize=(5, 5), sigmaX=0)

        return blurred_frame

    def get_colour_frame(self):
        rgb_frame = self._get_frame()
        bgr_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)

        return bgr_frame

    def _get_frame(self):
        frame = self._camera.capture_array()
        return cv2.rotate(frame, cv2.ROTATE_180)


# TODO: Options to get frome live feed or pre-recorded files (images and videos?)
