import cv2
import numpy as np
import datetime
import os
import psycopg2
from dotenv import load_dotenv

load_dotenv()

# Database connection parameters
db_name = os.getenv('DB_NAME')
db_user = os.getenv('DB_USER')
db_pass = os.getenv('DB_PASS')
db_host = os.getenv('DB_HOST')
db_port = os.getenv('DB_PORT')

# Directory for snapshots
snapshot_dir = r"/home/noaa/PycharmProjects/ObjectDetection/snapshot_directory"
os.makedirs(snapshot_dir, exist_ok=True)


# Function to connect to the database
def connect_db():
    return psycopg2.connect(
        database=db_name,
        user=db_user,
        password=db_pass,
        host=db_host,
        port=db_port
    )


# Function to insert detection data into the database
def insert_detection_data(timestamp, object_type, confidence, location, bounding_box, image_path, additional_metadata):
    conn = connect_db()
    cur = conn.cursor()
    sql = """
    INSERT INTO object_detections (timestamp, object_type, confidence_score, location, bounding_box, image_snapshot, additional_metadata)
    VALUES (%s, %s, %s, %s, %s, %s, %s)
    """
    with open(image_path, 'rb') as file:
        image_data = file.read()

    # Explicitly convert confidence to a Python float
    confidence = float(confidence)

    cur.execute(sql, (timestamp, object_type, confidence, location, bounding_box, image_data, additional_metadata))
    conn.commit()
    cur.close()
    conn.close()


# Load pre-trained model and classes
net = cv2.dnn.readNetFromCaffe('deep_learning_model_setup/deploy.prototxt', 'deep_learning_model_setup'
                                                                            '/mobilenet_iter_73000.caffemodel')
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
        if confidence > 0.2:
            idx = int(detections[0, 0, i, 1])
            if classes[idx] not in ["dog", "person", "cat"]:
                continue
            detected_class = classes[idx]
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
            label = "{}: {:.2f}%".format(detected_class, confidence * 100)
            cv2.putText(frame, label, (startX, startY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            snapshot_filename = f"{snapshot_dir}/snapshot_{detected_class}_{current_time.strftime('%Y%m%d_%H%M%S')}.png"
            cv2.imwrite(snapshot_filename, frame)
            insert_detection_data(
                timestamp=current_time,
                object_type=detected_class,
                confidence=confidence,
                location="YourCameraLocation",  # Replace with actual camera location
                bounding_box=str([startX, startY, endX, endY]),
                image_path=snapshot_filename,
                additional_metadata="{}"  # Replace with actual metadata if available
            )
            break


if __name__ == "__main__":
    # Load a video
    cap = cv2.VideoCapture('tests_videos/dog_poo.mp4')

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
