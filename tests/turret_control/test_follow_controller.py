from gardendome.turret_control.angles import Angles
from gardendome.turret_control.follow_controller import FollowConfig, FollowController

FRAME_SHAPE = (160, 240, 3)
CURRENT_PAN = 90
CURRENT_TILT = 120
NO_STEP_LIMIT = float("inf")


def update(centre, gain, max_step):
    follow_config = FollowConfig(
        fov_h=60,
        fov_v=40,
        gain=gain,
        max_step=max_step,
        start_threshold=0,
        stop_threshold=0,
    )
    controller = FollowController(follow_config)
    return controller.update(
        target_centre=centre,
        frame_shape=FRAME_SHAPE,
        current=Angles(pan=CURRENT_PAN, tilt=CURRENT_TILT),
    )


def test_target_at_centre_returns_current_position():
    result = update((120, 80), gain=1, max_step=NO_STEP_LIMIT)
    assert result == (CURRENT_PAN, CURRENT_TILT)


def test_target_at_right_edge_lowers_pan_by_half_fov():
    result = update((240, 80), gain=1, max_step=NO_STEP_LIMIT)
    assert result == (CURRENT_PAN - 30, CURRENT_TILT)


def test_target_at_left_edge_raises_pan_by_half_fov():
    result = update((0, 80), gain=1, max_step=NO_STEP_LIMIT)
    assert result == (CURRENT_PAN + 30, CURRENT_TILT)


def test_target_at_top_edge_lowers_tilt_by_half_fov():
    result = update((120, 0), gain=1, max_step=NO_STEP_LIMIT)
    assert result == (CURRENT_PAN, CURRENT_TILT - 20)


def test_target_at_bottom_edge_raises_tilt_by_half_fov():
    result = update((120, 160), gain=1, max_step=NO_STEP_LIMIT)
    assert result == (CURRENT_PAN, CURRENT_TILT + 20)


def test_step_moves_gain_fraction_of_small_error():
    result = update((104, 72), gain=0.5, max_step=4)
    assert result == (CURRENT_PAN + 2, CURRENT_TILT - 1)


def test_step_caps_large_positive_error_at_max_step():
    result = update((40, 160), gain=0.5, max_step=4)
    assert result == (CURRENT_PAN + 4, CURRENT_TILT + 4)


def test_step_caps_large_negative_error_at_max_step():
    result = update((200, 0), gain=0.5, max_step=4)
    assert result == (CURRENT_PAN - 4, CURRENT_TILT - 4)


def test_step_caps_each_axis_independently():
    result = update((40, 88), gain=0.5, max_step=4)
    assert result == (CURRENT_PAN + 4, CURRENT_TILT + 1)


def make_hysteresis_controller():
    follow_config = FollowConfig(
        fov_h=60,
        fov_v=40,
        gain=1,
        max_step=NO_STEP_LIMIT,
        start_threshold=3,
        stop_threshold=1,
    )
    return FollowController(follow_config)


def update_controller(controller, centre):
    return controller.update(
        target_centre=centre,
        frame_shape=FRAME_SHAPE,
        current=Angles(pan=CURRENT_PAN, tilt=CURRENT_TILT),
    )


def test_idle_error_below_start_threshold_returns_none():
    controller = make_hysteresis_controller()
    assert update_controller(controller, (112, 80)) is None


def test_idle_error_above_start_threshold_moves():
    controller = make_hysteresis_controller()
    result = update_controller(controller, (104, 80))
    assert result == (CURRENT_PAN + 4, CURRENT_TILT)


def test_idle_negative_error_above_start_threshold_moves():
    controller = make_hysteresis_controller()
    result = update_controller(controller, (136, 80))
    assert result == (CURRENT_PAN - 4, CURRENT_TILT)


def test_idle_tilt_error_alone_above_start_threshold_moves():
    controller = make_hysteresis_controller()
    result = update_controller(controller, (120, 96))
    assert result == (CURRENT_PAN, CURRENT_TILT + 4)


def test_moving_error_between_thresholds_keeps_moving():
    controller = make_hysteresis_controller()
    update_controller(controller, (104, 80))
    result = update_controller(controller, (112, 80))
    assert result == (CURRENT_PAN + 2, CURRENT_TILT)


def test_moving_error_below_stop_threshold_returns_none():
    controller = make_hysteresis_controller()
    update_controller(controller, (104, 80))
    assert update_controller(controller, (118, 80)) is None


def test_stopped_error_between_thresholds_returns_none():
    controller = make_hysteresis_controller()
    update_controller(controller, (104, 80))
    update_controller(controller, (118, 80))
    assert update_controller(controller, (112, 80)) is None
