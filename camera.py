import cv2


def open_camera(camera_index=0):
    """Open and return a configured video capture device."""
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Kamera {camera_index} konnte nicht geöffnet werden.")
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


def read_frame(cap, flip=True):
    """Read a single frame from the camera and optionally mirror it."""
    if cap is None or not hasattr(cap, "read"):
        raise RuntimeError("Ungültiges Kameraobjekt.")

    ret, frame = cap.read()
    if not ret or frame is None:
        return None

    if flip:
        frame = cv2.flip(frame, 1)

    return frame


def close_camera(cap):
    """Release camera resources and close OpenCV windows."""
    if cap is not None:
        try:
            cap.release()
        except Exception:
            pass
    cv2.destroyAllWindows()