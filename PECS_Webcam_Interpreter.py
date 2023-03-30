import cv2
import numpy as np
from keras.models import load_model

# Load the trained model
model = load_model("object_recognition_model.h5")

# Load the class indices file
class_indices = np.load('class_indices.npy', allow_pickle=True).item()

# Map class indices to corresponding labels
labels = {v: k for k, v in class_indices.items()}

# Open the default camera
cap = cv2.VideoCapture(1)

# Set minimum confidence level for detection
confidence_threshold = 0.5

while True:
    # Read a frame from the camera
    ret, frame = cap.read()

    # Get frame dimensions
    height, width, _ = frame.shape

    # Preprocess the frame for the model
    preprocessed_frame = cv2.resize(frame, (256, 256))
    preprocessed_frame = preprocessed_frame.astype('float32') / 255.0
    preprocessed_frame = np.expand_dims(preprocessed_frame, axis=0)

    # Use the model to predict the classes of objects in the frame
    predictions = model.predict(preprocessed_frame)

    # Filter out predictions with low confidence
    detections = []
    if len(predictions) > 0:
        for index, confidence in enumerate(predictions[0]):
            if confidence > confidence_threshold:
                detections.append((index, confidence))

    # Draw bounding boxes around detected objects
    for detection in detections:
        # Get the class label of the detected object
        class_index = detection[0]
        class_label = labels[class_index]

        # Get the coordinates of the detected object
        x, y, w, h = detection[1:5] * np.array([width, height, width, height])
        left = int(x - w/2)
        top = int(y - h/2)
        right = int(x + w/2)
        bottom = int(y + h/2)

        # Draw the bounding box and label of the detected object
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, class_label, (left, top - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow('Object Detection', frame)

    # Wait for a key press and check if it is the 'q' key
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

# Release the camera and close the window
cap.release()
cv2.destroyAllWindows()
