import cv2
from picamera2 import Picamera2

picam2 = Picamera2()
picam2.start()
cv2.namedWindow("Live Feed", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Live Feed", 640, 480)

while True:
    frame = picam2.capture_array()
    cv2.imshow("Live Feed", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()


