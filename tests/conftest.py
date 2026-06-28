from unittest.mock import MagicMock
import sys

sys.modules["picamera2"] = MagicMock()
sys.modules["picamera2.outputs"] = MagicMock()
sys.modules["picamera2.encoders"] = MagicMock()

sys.modules["gardendome.turret_control.PCA9685"] = MagicMock()