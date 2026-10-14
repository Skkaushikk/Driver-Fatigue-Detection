
import cv2

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Error: Webcam could not be opened.")
else:
    print("Webcam started successfully!")

while camera.isOpened():
    success, frame = camera.read()

    if not success:
        print("Could not read webcam frame.")
        break

    cv2.imshow("Driver Fatigue Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()
