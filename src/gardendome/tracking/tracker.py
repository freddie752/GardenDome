import cv2

class Tracker:
    def draw_box(frame, bbox):
        x, y, w, h = bbox
        annotated = frame.copy()
        cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)
        return annotated

    def calculate_centre(bbox):
        x, y, w, h = bbox
        return x + 0.5 * w, y + 0.5 * h
