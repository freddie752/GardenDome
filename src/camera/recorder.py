from datetime import datetime
from picamera2.outputs import FileOutput
from picamera2.encoders import H264Encoder


class Picamera2Recorder:
    def __init__(self, logger, camera, recording_dir, bitrate):
        self._logger = logger
        self._camera = camera
        self._recording_dir = recording_dir
        self._encoder = H264Encoder(bitrate=bitrate)
        self._is_recording = False
        self._current_file = None

    @property
    def is_recording(self):
        return self._is_recording

    def start(self, prefix: str):
        if self._is_recording:
            raise RuntimeError("Cannot start recording: Recorder is already active.")
        self._is_recording = True
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._current_file = f"{prefix}_{timestamp}.h264"
        file_output = FileOutput(f"{self._recording_dir}{self._current_file}")
        self._camera.start_recording(self._encoder, file_output)
        self._logger.info(f"Recording started. Storing at {self._current_file}")
        
    def stop(self):
        if not self._is_recording:
            raise RuntimeError("Cannot stop recording: Recorder is not active.")
        self._camera.stop_recording()
        self._logger.video(self._recording_dir, self._current_file)
        self._current_file = None
        self._is_recording = False
        self._logger.info("Recording stopped.")
        
