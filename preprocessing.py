import cv2


def preprocess_frame(frame, width=320, height=240):
    """
    Vorverarbeitung für späteren Einsatz auf dem Raspberry Pi.
    """
    resized = cv2.resize(frame, (width, height))
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    return resized, gray, blurred