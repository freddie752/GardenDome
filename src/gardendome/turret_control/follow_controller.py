class FollowController:
    def __init__(self, fov_h, fov_v, gain, max_step):
        self._fov_h = fov_h
        self._fov_v = fov_v
        self._gain = gain
        self._max_step = max_step

    def _calculate_target_position(
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

    def _calculate_next_position(
        self, target_pan, target_tilt, current_pan, current_tilt
    ):
        pan_diff = target_pan - current_pan
        pan_move = max(min(self._gain * pan_diff, self._max_step), -self._max_step)
        next_pan = current_pan + pan_move

        tilt_diff = target_tilt - current_tilt
        tilt_move = max(min(self._gain * tilt_diff, self._max_step), -self._max_step)
        next_tilt = current_tilt + tilt_move

        return next_pan, next_tilt

    def update(self, target_centre, frame_shape, current_pan, current_tilt):
        target_pan, target_tilt = self._calculate_target_position(
            target_centre=target_centre,
            frame_shape=frame_shape,
            current_pan=current_pan,
            current_tilt=current_tilt,
        )
        return self._calculate_next_position(
            target_pan=target_pan,
            target_tilt=target_tilt,
            current_pan=current_pan,
            current_tilt=current_tilt,
        )
