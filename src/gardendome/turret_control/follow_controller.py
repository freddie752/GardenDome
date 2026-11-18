class FollowController:
    def __init__(self, fov_h, fov_v):
        self._fov_h = fov_h
        self._fov_v = fov_v

    def calculate_target_position(
        self, target_centre, frame_shape, current_pan, current_tilt
    ):
        frame_h, frame_w = frame_shape[:2]
        target_x, target_y = target_centre

        x_offset = target_x / frame_w - 0.5
        pan_offset = x_offset * self._fov_h
        target_pan = current_pan - pan_offset

        y_offset = target_y / frame_h - 0.5
        tilt_offset = y_offset * self._fov_v
        target_tilt = current_tilt + tilt_offset

        return target_pan, target_tilt
