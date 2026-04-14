import cv2

from camera import close_camera, open_camera, read_frame
from config import (
    CAMERA_INDEX,
    DILATE_ITERATIONS,
    DRAW_THROW_ZONE,
    EVALUATION_GOOD_DELTA,
    EVALUATION_HISTORY_LENGTH,
    FRAME_HEIGHT,
    FRAME_WIDTH,
    MIN_MOTION_AREA,
    SHOW_GRAY_WINDOW,
    SHOW_MASK_WINDOW,
    THRESHOLD_VALUE,
    THROW_COOLDOWN_FRAMES,
    THROW_HISTORY_LENGTH,
    THROW_MIN_UPWARD_DELTA,
    THROW_ZONE_END_RATIO,
    THROW_ZONE_RATIO,
    THROW_ZONE_START_RATIO,
)
from motion_detection import detect_motion
from preprocessing import preprocess_frame
from throw_detection import ThrowDetector
from evaluation import ThrowEvaluator


def draw_status(frame, status_text: str, alert: bool = False, line: int = 1) -> None:
    """Draw status text on the output frame."""
    color = (0, 255, 0) if not alert else (0, 0, 255)
    y = 25 + (line - 1) * 30
    cv2.putText(
        frame,
        status_text,
        (10, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color,
        2,
        cv2.LINE_AA,
    )


def draw_boxes(frame, boxes):
    """Draw motion boxes on the output frame."""
    for x, y, w, h in boxes:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)


def main():
    cap = None

    try:
        cap = open_camera(CAMERA_INDEX)
        print("Kamera läuft.")
        print("Drücke 'q' zum Beenden.")

        previous_blurred = None
        throw_detector = ThrowDetector(
            history_length=THROW_HISTORY_LENGTH,
            min_upward_delta=THROW_MIN_UPWARD_DELTA,
            cooldown_frames=THROW_COOLDOWN_FRAMES,
            zone_start_ratio=THROW_ZONE_START_RATIO,
            zone_end_ratio=THROW_ZONE_END_RATIO,
        )
        throw_evaluator = ThrowEvaluator(
            history_length=EVALUATION_HISTORY_LENGTH,
            good_delta=EVALUATION_GOOD_DELTA,
            zone_end_ratio=THROW_ZONE_END_RATIO,
        )

        while True:
            frame = read_frame(cap)
            if frame is None:
                print("Fehler: Kein Bild von der Kamera erhalten.")
                break

            resized, gray, blurred = preprocess_frame(frame, FRAME_WIDTH, FRAME_HEIGHT)
            motion_view = resized.copy()

            if previous_blurred is not None:
                motion_mask, boxes = detect_motion(
                    previous_blurred,
                    blurred,
                    min_area=MIN_MOTION_AREA,
                    threshold_value=THRESHOLD_VALUE,
                    dilate_iterations=DILATE_ITERATIONS,
                )
                draw_boxes(motion_view, boxes)

                throw_detected, status_text = throw_detector.update(boxes, FRAME_HEIGHT)
                evaluation_label, evaluation_status = throw_evaluator.update(
                    boxes,
                    FRAME_HEIGHT,
                    throw_detected=throw_detected,
                )

                if DRAW_THROW_ZONE:
                    zone_y = int(FRAME_HEIGHT * THROW_ZONE_RATIO)
                    cv2.line(
                        motion_view,
                        (0, zone_y),
                        (FRAME_WIDTH, zone_y),
                        (255, 0, 0),
                        2,
                    )

                if throw_detected:
                    draw_status(
                        motion_view,
                        evaluation_label,
                        alert=(evaluation_label != "Guter Wurf"),
                        line=1,
                    )
                    draw_status(
                        motion_view,
                        evaluation_status,
                        alert=(evaluation_label != "Guter Wurf"),
                        line=2,
                    )
                else:
                    draw_status(motion_view, status_text)

                if SHOW_MASK_WINDOW:
                    cv2.imshow("Bewegungsmaske", motion_mask)

            if SHOW_GRAY_WINDOW:
                cv2.imshow("Graustufen", gray)

            cv2.imshow("Kamera", motion_view)
            previous_blurred = blurred

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except RuntimeError as error:
        print(f"Fehler: {error}")

    finally:
        if cap is not None:
            close_camera(cap)


if __name__ == "__main__":
    main()