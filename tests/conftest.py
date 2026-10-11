import sys
from unittest.mock import MagicMock

sys.modules["picamera2"] = MagicMock()
sys.modules["picamera2.outputs"] = MagicMock()
sys.modules["picamera2.encoders"] = MagicMock()

sys.modules["gardendome.turret_control.PCA9685"] = MagicMock()

sys.modules["ai_edge_litert"] = MagicMock()
sys.modules["ai_edge_litert.interpreter"] = MagicMock()