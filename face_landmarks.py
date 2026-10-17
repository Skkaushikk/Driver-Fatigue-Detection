
import cv2
import mediapipe as mp
import os
import math
import time
import winsound
import threading

# Step 1: Locate the face landmark model
model_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "face_landmarker.task"
)

if not os.path.exists(model_path):
    print("Error: face_landmarker.task not found!")
    raise SystemExit

# Step 2: Configure MediaPipe
BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=RunningMode.VIDEO,
    num_faces=1
)

# Eye landmark indices
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]


def calculate_ear(landmarks, eye_indices, width, height):
    points = []

    for index in eye_indices:
        point = landmarks[index]
        points.append((
            int(point.x * width),
            int(point.y * height)
        ))

    vertical_1 = math.dist(points[1], points[5])
    vertical_2 = math.dist(points[2], points[4])
    horizontal = math.dist(points[0], points[3])

    if horizontal == 0:
        return 0.0

    return (vertical_1 + vertical_2) / (2.0 * horizontal)


# Step 3: Start webcam
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Error: Webcam could not be opened.")
    raise SystemExit

print("Driver Fatigue Prototype Started! Press Q to quit.")

EAR_THRESHOLD = 0.21
CLOSED_SECONDS_REQUIRED = 2.0

eyes_closed_since = None
alert_triggered = False
last_timestamp = -1


def play_warning_sound():
    try:
        winsound.Beep(1000, 2000)
    except RuntimeError:
        print("Could not play warning sound.")


try:
    with FaceLandmarker.create_from_options(options) as landmarker:
        while True:
            success, frame = camera.read()

            if not success:
                print("Could not read webcam frame.")
                break

            frame = cv2.flip(frame, 1)
            height, width = frame.shape[:2]

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb_frame
            )

            timestamp_ms = max(
                int(time.monotonic() * 1000),
                last_timestamp + 1
            )
            last_timestamp = timestamp_ms

            results = landmarker.detect_for_video(
                mp_image, timestamp_ms
            )

            current_time = time.monotonic()

            if results.face_landmarks:
                landmarks = results.face_landmarks[0]

                # Draw all facial landmarks in green
                for point in landmarks:
                    x = int(point.x * width)
                    y = int(point.y * height)

                    if 0 <= x < width and 0 <= y < height:
                        cv2.circle(
                            frame, (x, y), 1, (0, 255, 0), -1
                        )

                # Calculate Eye Aspect Ratio (EAR)
                left_ear = calculate_ear(
                    landmarks, LEFT_EYE, width, height
                )
                right_ear = calculate_ear(
                    landmarks, RIGHT_EYE, width, height
                )
                ear = (left_ear + right_ear) / 2.0

                eyes_closed = ear < EAR_THRESHOLD

                # Track continuous eye closure
                if eyes_closed:
                    if eyes_closed_since is None:
                        eyes_closed_since = current_time

                    closed_duration = current_time - eyes_closed_since
                else:
                    eyes_closed_since = None
                    closed_duration = 0.0
                    alert_triggered = False

                # Show EAR and eye status
                eye_status = "Eyes Closed" if eyes_closed else "Eyes Open"
                eye_color = (0, 0, 255) if eyes_closed else (0, 255, 0)

                cv2.putText(
                    frame,
                    f"EAR: {ear:.2f} | {eye_status}",
                    (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    eye_color,
                    2
                )

                # Highlight selected eye landmarks in cyan
                for index in LEFT_EYE + RIGHT_EYE:
                    point = landmarks[index]
                    x = int(point.x * width)
                    y = int(point.y * height)
                    cv2.circle(
                        frame, (x, y), 3, (255, 255, 0), -1
                    )

                # Show the closure timer
                if eyes_closed:
                    cv2.putText(
                        frame,
                        f"Eyes closed for: {closed_duration:.1f}s",
                        (20, 60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        eye_color,
                        2
                    )

                # Choose exactly one main status message
                if closed_duration >= CLOSED_SECONDS_REQUIRED:
                    cv2.putText(
                        frame,
                        "YOU'RE NOT GOOD TO GO!",
                        (20, 95),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        (0, 0, 255),
                        2
                    )

                    # Play one beep per continuous closure event
                    if not alert_triggered:
                        alert_triggered = True
                        threading.Thread(
                            target=play_warning_sound,
                            daemon=True
                        ).start()

                elif eyes_closed:
                    cv2.putText(
                        frame,
                        "EYES CLOSED - WAITING...",
                        (20, 95),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        (0, 165, 255),
                        2
                    )

                else:
                    cv2.putText(
                        frame,
                        "YOU'RE GOOD TO GO!",
                        (20, 95),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        (0, 255, 0),
                        2
                    )

            else:
                # Reset if face disappears
                eyes_closed_since = None
                alert_triggered = False

                cv2.putText(
                    frame,
                    "NO FACE DETECTED",
                    (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2
                )

            # Display output
            cv2.imshow(
                "Driver Fatigue - Face and Eye Tracking",
                frame
            )

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

finally:
    camera.release()
    cv2.destroyAllWindows()

print("Webcam closed successfully.")
