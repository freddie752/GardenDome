from gardendome.turret_control.follow_controller import FollowController

FRAME_SHAPE = (100, 200, 3)
CURRENT_PAN = 90
CURRENT_TILT = 120


def make_controller():
    return FollowController(fov_h=60, fov_v=40)


def calculate(centre):
    return make_controller().calculate_target_position(
        target_centre=centre,
        frame_shape=FRAME_SHAPE,
        current_pan=CURRENT_PAN,
        current_tilt=CURRENT_TILT,
    )


def test_target_at_centre_returns_current_position():
    assert calculate((100, 50)) == (CURRENT_PAN, CURRENT_TILT)


def test_target_at_right_edge_lowers_pan_by_half_fov():
    assert calculate((200, 50)) == (CURRENT_PAN - 30, CURRENT_TILT)


def test_target_at_left_edge_raises_pan_by_half_fov():
    assert calculate((0, 50)) == (CURRENT_PAN + 30, CURRENT_TILT)


def test_target_at_top_edge_lowers_tilt_by_half_fov():
    assert calculate((100, 0)) == (CURRENT_PAN, CURRENT_TILT - 20)


def test_target_at_bottom_edge_raises_tilt_by_half_fov():
    assert calculate((100, 100)) == (CURRENT_PAN, CURRENT_TILT + 20)
