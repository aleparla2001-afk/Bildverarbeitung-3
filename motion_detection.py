import cv2


def detect_motion(previous_frame, current_frame, min_area=500):
    """
    Vergleicht zwei vorverarbeitete Frames und erkennt Bewegungsbereiche.
    """
    diff = cv2.absdiff(previous_frame, current_frame)
    _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
    thresh = cv2.dilate(thresh, None, iterations=2)

    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    boxes = []

    for contour in contours:
        area = cv2.contourArea(contour)

        if area < min_area:
            continue

        x, y, w, h = cv2.boundingRect(contour)
        boxes.append((x, y, w, h))

    return thresh, boxes