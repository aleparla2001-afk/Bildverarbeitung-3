import cv2


def open_camera(camera_index=0):
    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        raise RuntimeError("Kamera konnte nicht geöffnet werden.")

    return cap


def read_frame(cap, flip=True):
    ret, frame = cap.read()

    if not ret:
        return None

    if flip:
        frame = cv2.flip(frame, 1)

    return frame


def close_camera(cap):
    cap.release()
    cv2.destroyAllWindows()