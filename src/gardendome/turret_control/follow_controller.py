from dataclasses import dataclass

from gardendome.turret_control.angles import Angles


@dataclass
class FollowConfig:
    fov_h: float
    fov_v: float
    gain: float
    max_step: float
    start_threshold: float
    stop_threshold: float


class FollowController:
    def __init__(self, follow_config):
        self._fov_h = follow_config.fov_h
        self._fov_v = follow_config.fov_v
        self._gain = follow_config.gain
        self._max_step = follow_config.max_step
        self._start_threshold = follow_config.start_threshold
        self._stop_threshold = follow_config.stop_threshold
        self._moving = False

    def _calculate_target_position(self, target_centre, frame_shape, current):
        frame_h, frame_w = frame_shape[:2]
        target_x, target_y = target_centre

        x_offset = target_x / frame_w - 0.5
        pan_offset = x_offset * self._fov_h
        target_pan = current.pan - pan_offset

        y_offset = target_y / frame_h - 0.5
        tilt_offset = y_offset * self._fov_v
        target_tilt = current.tilt + tilt_offset

        return Angles(pan=target_pan, tilt=target_tilt)

    def _calculate_next_angle(self, error, current_angle):
        step = max(min(self._gain * error, self._max_step), -self._max_step)
        return current_angle + step

    def update(self, target_centre, frame_shape, current):
        target = self._calculate_target_position(
            target_centre=target_centre,
            frame_shape=frame_shape,
            current=current,
        )
        error = Angles(pan=target.pan - current.pan, tilt=target.tilt - current.tilt)

        threshold = self._stop_threshold if self._moving else self._start_threshold
        if abs(error.pan) < threshold and abs(error.tilt) < threshold:
            self._moving = False
            return None

        self._moving = True
        next_pan = self._calculate_next_angle(
            error=error.pan, current_angle=current.pan
        )
        next_tilt = self._calculate_next_angle(
            error=error.tilt, current_angle=current.tilt
        )
        return Angles(pan=next_pan, tilt=next_tilt)
