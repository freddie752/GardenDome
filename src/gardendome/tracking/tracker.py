import cv2


class Tracker:
    def calculate_centre(self, bbox):
        x, y, w, h = bbox
        return x + w // 2, y + h // 2

    def draw_box(self, frame, bbox):
        x, y, w, h = bbox
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 5)
        return frame

    def draw_centre(self, frame, centre):
        cx, cy = int(centre[0]), int(centre[1])
        cv2.circle(frame, (cx, cy), 5, (255, 255, 255), -1)
        return frame
