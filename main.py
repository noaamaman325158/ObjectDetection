import cv2
import numpy as np
import datetime
import os

snapshot_dir = r"/home/noaa/PycharmProjects/ObjectDetection/snapshot_directory"  # Replace with your desired path
os.makedirs(snapshot_dir, exist_ok=True)

# Load pre-trained model and classes
net = cv2.dnn.readNetFromCaffe('deploy.prototxt', 'mobilenet_iter_73000.caffemodel')
classes = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
           "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]


def detect_dogs_and_people(frame):
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)

    net.setInput(blob)
    detections = net.forward()
    current_time = datetime.datetime.now()

    for i in np.arange(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.2:  # Confidence threshold
            idx = int(detections[0, 0, i, 1])
            if classes[idx] not in ["dog", "person"]:
                continue
            detected_class = classes[idx]
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
            label = "{}: {:.2f}%".format(detected_class, confidence * 100)
            cv2.putText(frame, label, (startX, startY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Write detection time to file and take a snapshot
            snapshot_filename = f"{snapshot_dir}/snapshot_{detected_class}_{current_time.strftime('%Y%m%d_%H%M%S')}.png"
            cv2.imwrite(snapshot_filename, frame)  # Save the current frame as an image
            with open("detections.txt", "a") as file:
                file.write(
                    f"{detected_class.capitalize()} detected at {current_time.strftime('%Y-%m-%d %H:%M:%S')}, snapshot saved as {snapshot_filename}\n")
            break  # Stop after the first detection to avoid multiple logs and snapshots for the same object in one frame


# Load a video
cap = cv2.VideoCapture('dog_poo.mp4')
while True:
    ret, frame = cap.read()
    if not ret:
        break
    resized_frame = cv2.resize(frame, (640, 480))
    detect_dogs_and_people(resized_frame)
    cv2.imshow('Frame', resized_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
