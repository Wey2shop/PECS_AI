import cv2

# Open the default camera
cap = cv2.VideoCapture(1)

# Initialize frame count
count = 0

# Start a loop to read frames from the video stream
while True:
    # Read a frame from the video stream
    ret, frame = cap.read()

    # Increment frame count
    count += 1

    # Check if the frame was read successfully
    if ret:
        # If the frame count is divisible by 10, save the frame
        if count % 10 == 0:
            filename = f"{count:06}.jpg"  # Format the file name with 6 digits
            cv2.imwrite(filename, frame)   # Save the frame as a JPEG image
            print(f"Saved frame {count}")  # Print a message indicating that the frame has been saved

        # Show the frame
        cv2.imshow("Video Stream", frame)

        # Wait for a key press to exit the program
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        # If the frame was not read successfully, break the loop
        break

# Release the video stream and close all windows
cap.release()
cv2.destroyAllWindows()
