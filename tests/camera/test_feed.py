import pytest
from unittest.mock import Mock, patch
import numpy as np
import cv2
from src.camera.feed import Picamera2Feed

@pytest.fixture
def mock_camera():
    return Mock()

@pytest.fixture
def picamera2_feed(mock_camera):
    return Picamera2Feed(mock_camera)

@patch("cv2.cvtColor")
@patch("cv2.GaussianBlur")
def test_get_frame(mock_gaussian_blur, mock_cvt_color, mock_camera, picamera2_feed):
    # Mock camera capture_array to return a dummy image
    dummy_bgr_image = np.zeros((100, 100, 3), dtype=np.uint8)
    mock_camera.capture_array.return_value = dummy_bgr_image

    # Mock cv2.cvtColor to return specific processed images
    mock_rgb_image = np.zeros((100, 100, 3), dtype=np.uint8)
    mock_gray_image = np.zeros((100, 100), dtype=np.uint8)
    mock_cvt_color.side_effect = [mock_rgb_image, mock_gray_image]
    
    # Mock cv2.GaussianBlur to return a final frame
    mock_gaussian_blur.return_value = np.zeros((100, 100), dtype=np.uint8)

    frame = picamera2_feed.get_frame()

    # Assertions
    mock_camera.capture_array.assert_called_once()
    assert mock_cvt_color.call_count == 2
    mock_cvt_color.assert_any_call(src=dummy_bgr_image, code=cv2.COLOR_BGR2RGB)
    mock_cvt_color.assert_any_call(mock_rgb_image, cv2.COLOR_BGR2GRAY)
    mock_gaussian_blur.assert_called_once_with(src=mock_gray_image, ksize=(5, 5), sigmaX=0)
    assert isinstance(frame, np.ndarray)