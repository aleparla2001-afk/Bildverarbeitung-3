import cv2

from camera import open_camera, read_frame, close_camera
from preprocessing import preprocess_frame
from motion_detection import detect_motion


def main():
    try:
        cap = open_camera(0)
    except RuntimeError as e:
        print(f"Fehler: {e}")
        return

    print("Kamera läuft.")
    print("Drücke 'q' zum Beenden.")
    """
    Zentrale Konfigurationsdatei für den Wurfcoach.
    Alle einstellbaren Parameter sind hier definiert.
    """

    # Kamera-Einstellungen
    CAMERA_INDEX = 0
    FRAME_WIDTH = 640
    FRAME_HEIGHT = 480
    CAMERA_FPS = 30

    # Preprocessing
    GAUSSIAN_BLUR_KERNEL = (5, 5)
    GAUSSIAN_BLUR_SIGMA = 0

    # Motion Detection
    THRESHOLD_VALUE = 30
    DILATE_KERNEL_SIZE = 5
    DILATE_ITERATIONS = 2
    MIN_MOTION_AREA = 100

    # Throw Detection
    THROW_HISTORY_LENGTH = 10
    THROW_MIN_UPWARD_DELTA = 30  # Pixel, die sich nach oben bewegen müssen
    THROW_COOLDOWN_FRAMES = 30  # Frames bevor ein neuer Wurf erkannt wird
    THROW_ZONE_HEIGHT_RATIO = 0.3  # Obere 30% des Bildes als Wurfzone

    # UI/Anzeige
    SHOW_GRAY_WINDOW = False
    SHOW_MASK_WINDOW = False
    SHOW_THROW_ZONE_LINE = True
    WINDOW_ORIGINAL = "Wurfcoach"
    WINDOW_GRAY = "Graustufen"
    WINDOW_MASK = "Bewegungsmaske"

    # Debug
    DEBUG_MODE = False

    previous_blurred = None

    while True:
        frame = read_frame(cap)

        if frame is None:
            print("Fehler: Kein Bild von der Kamera erhalten.")
            break

        resized, gray, blurred = preprocess_frame(frame)

        motion_view = resized.copy()

        if previous_blurred is not None:
            motion_mask, boxes = detect_motion(previous_blurred, blurred)

            for (x, y, w, h) in boxes:
                cv2.rectangle(motion_view, (x, y), (x + w, y + h), (0, 255, 0), 2)

            cv2.imshow("Bewegungsmaske", motion_mask)

        cv2.imshow("Original", resized)
        cv2.imshow("Graustufen", gray)
        cv2.imshow("Erkannte Bewegung", motion_view)

        previous_blurred = blurred

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    close_camera(cap)


if __name__ == "__main__":
    main()