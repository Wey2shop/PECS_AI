import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('Model/pecs_recognition_model.h5')

# Load the class indices
class_indices = np.load('Model/class_indices.npy', allow_pickle=True).item()

# Define the labels for the classes
class_labels = {v: k for k, v in class_indices.items()}

# Create a window to display the camera feed
cv2.namedWindow('Object Classification')

# Open the default camera
cap = cv2.VideoCapture(1)

while True:
    # Read a frame from the camera
    ret, frame = cap.read()

    # Preprocess the frame
    img = cv2.resize(frame, (256, 256))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    # Use the trained model to predict the class
    predictions = model.predict(img)
    class_index = np.argmax(predictions[0])
    
    # Check if the prediction confidence is greater than 80%
    if predictions[0][class_index] > 0.8:
        class_label = f"{class_labels[class_index]}: {predictions[0][class_index]*100:.2f}%"
    else:
        class_label = "Waiting for detection"

    # Display the class label on the camera feed
    cv2.putText(frame, class_label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Display the camera feed
    cv2.imshow('Object Classification', frame)

    # Press 'q' to quit the program
    if cv2.waitKey(1) == ord('q'):
        break

# Release the camera and close the window
cap.release()
cv2.destroyAllWindows()
