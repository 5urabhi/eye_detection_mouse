import csv
import time
import cv2
import numpy as np
from pynput.mouse import Listener

# Initialize mouse position
mouse_pos = None

# Function to handle mouse move events
def on_move(x, y):
    global mouse_pos
    mouse_pos = (x, y)

# Function to handle mouse click events
def on_click(x, y, button, pressed):
    if pressed:
        # Record mouse position and timestamp
        data.append((time.time(), mouse_pos[0], mouse_pos[1]))

# Create list to store data
data = []

# Start mouse listener
with Listener(on_move=on_move, on_click=on_click) as listener:
    # Initialize the webcam
    cap = cv2.VideoCapture(0)

    # Load the Haar cascade for eye detection
    eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

    # Initialize variables for eye tracking
    eye_locations = []
    frame_count = 0

    # Set the duration of the recording in seconds
    duration = 60

    # Start the timer
    start_time = time.time()

    # Loop until the duration has passed
    while time.time() - start_time < duration:
        # Read a frame from the webcam
        ret, frame = cap.read()

        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect eyes in the frame
        eyes = eye_cascade.detectMultiScale(gray, 1.3, 5)

        # Save eye locations every 5 frames
        if frame_count == 5:
            eye_locations.append(eyes)
            frame_count = 0

        # Draw rectangles around the eyes
        for (x, y, w, h) in eyes:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Display the frame
        cv2.imshow('Eye Tracking', frame)

        # Exit if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_count += 1

    # Release the webcam and close the window
    cap.release()
    cv2.destroyAllWindows()

    # Save eye locations to CSV file
    with open('eye_data.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['x', 'y', 'w', 'h'])
        for eyes in eye_locations:
            for (x, y, w, h) in eyes:
                writer.writerow([x, y, w, h])

    # Stop mouse listener
    listener.stop()

    # Save mouse data to CSV file
    with open('mouse_data.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['timestamp', 'x', 'y'])
        writer.writerows(data)

    # Print a message to indicate that the data has been saved
    print('Data saved successfully!')
