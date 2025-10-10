import cv2
import numpy as np
from PIL import ImageGrab
from picamera2 import Picamera2
from picamera2.encoders import H264Encoder
from picamera2.outputs import FileOutput
from datetime import datetime

RECORDING_DIR = "data/recording/"    
BITRATE = 2000000
MOTION_THRESHOLD = 10  # Pixel intensity difference threshold to detect motion

class MotionDetectorRecorder:
    def __init__(self, recording_dir, bitrate, motion_threshold):
        self.recording_dir = recording_dir
        self.bitrate = bitrate
        self.motion_threshold = motion_threshold
        self.picam2 = Picamera2()
        self.camera_config = self.picam2.create_video_configuration()
        self.picam2.configure(self.camera_config)
        self.picam2.start()
        self.encoder = H264Encoder(bitrate=self.bitrate)

    def detect_motion_record(self):
        """Detects motion and records video when motion is detected."""
        previous_frame = None
        previous_motion = False

        while True:
            current_frame = self.get_frame()

            if previous_frame is None:
                previous_frame = current_frame
                continue

            current_motion = self.detect_motion(previous_frame, current_frame)

            if current_motion and not previous_motion:
                # Start recording
                filename = datetime.now().strftime("motion_%Y%m%d_%H%M%S.h264")
                file_output = FileOutput(f"{self.recording_dir}{filename}")
                self.picam2.start_recording(self.encoder, file_output)    
                print(f"Motion detected. Recording to {filename}")
                
                #TODO: Live object recognition around contours
                
            elif not current_motion and previous_motion:
                # Stop recording
                print("Motion stopped.")
                self.picam2.stop_recording()
                self.picam2.start()

            previous_frame = current_frame 
            previous_motion = current_motion

    def detect_motion(self, previous_frame, current_frame):
        """Detects motion by comparing pixel differences of two frames."""

        # Compute absolute difference between two frames
        diff_frame = cv2.absdiff(src1=previous_frame, src2=current_frame)

        # Calculate pixels where difference is above threshold
        motion_mask = (diff_frame > self.motion_threshold).astype(np.uint8) * 255

        # Calculate the percentage of changed pixels
        motion_percentage = np.sum(motion_mask) / (motion_mask.shape[0] * motion_mask.shape[1] * 255)
        
        # Return True if more than 1% of pixels changed
        return motion_percentage > 0.01  

    #TODO: Options to get frome live feed or pre-recorded files (images and videos?)
    def get_frame(self):
        # Load image from picam live feed and convert it to RGB
        img_brg = self.picam2.capture_array()
        img_rgb = cv2.cvtColor(src=img_brg, code=cv2.COLOR_BGR2RGB)

        #Grayscale and blur image
        current_frame = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
        current_frame = cv2.GaussianBlur(src=current_frame, ksize=(5, 5), sigmaX=0)

        return current_frame

if __name__ == "__main__":
    motion_detector_recorder = MotionDetectorRecorder(RECORDING_DIR, BITRATE, MOTION_THRESHOLD)
    motion_detector_recorder.detect_motion_record()