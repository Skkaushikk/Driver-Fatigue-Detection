
import cv2
import os

# Step 1: Load the YuNet face detection model
model_path = "face_detection_yunet_2026may.onnx"

if not os.path.exists(model_path):
    print("Error: YuNet model file not found!")
    print("Keep the model file in the same folder as main.py.")
    raise SystemExit

face_detector = cv2.FaceDetectorYN.create(
    model_path,
    "",
    (320, 320),
    0.6,
    0.3,
    5000
)

# Step 2: Start the webcam
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Error: Webcam could not be opened.")
    raise SystemExit

print("Face Detection Started! Press Q to quit.")

# Step 3: Read video frames
while True:
    success, frame = camera.read()

    if not success:
        print("Could not read webcam frame.")
        break

    # Step 4: Set the webcam frame size
    height, width = frame.shape[:2]
    face_detector.setInputSize((width, height))

    # Step 5: Detect faces using YuNet
    _, faces = face_detector.detect(frame)

    # Step 6: Draw a rectangle around each detected face
    if faces is not None:
        for face in faces:
            x, y, w, h = face[:4].astype(int)

            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                "Face Detected",
                (x, max(y - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

    # Step 7: Display the video
    cv2.imshow("Driver Fatigue Detection", frame)

    # Press Q to exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Step 8: Release the webcam
camera.release()
cv2.destroyAllWindows()

print("Webcam closed successfully.")
