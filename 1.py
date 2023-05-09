import csv
import time
import cv2
import numpy as np
import pyautogui
from pynput.mouse import Listener

# Initialize mouse position
mouse_pos = None

# Function to handle mouse move events
def on_move(x, y):
    global mouse_pos
    mouse_pos = (x, y)

# Function to handle mouse click events
def on_click(x, y, button, pressed):
    pass

# Create list to store data
data = []

# Initialize variables for eye tracking
eye_locations = []
frame_count = 0

# Get screen resolution
screen_width, screen_height = pyautogui.size()

# Define calibration points
calibration_points = [
    (screen_width // 2, screen_height // 2), # center
    (screen_width // 4, screen_height // 4), # top-left
    (screen_width * 3 // 4, screen_height // 4), # top-right
    (screen_width * 3 // 4, screen_height * 3 // 4), # bottom-right
    (screen_width // 4, screen_height * 3 // 4), # bottom-left
    (screen_width // 2, screen_height // 4), # top-center
    (screen_width // 2, screen_height * 3 // 4), # bottom-center
    (screen_width // 4, screen_height // 2), # middle-left
    (screen_width * 3 // 4, screen_height // 2), # middle-right
]

# Define duration in seconds
duration = 60

# Start mouse listener
with Listener(on_move=on_move, on_click=on_click) as listener:
    # Initialize the webcam
    cap = cv2.VideoCapture(0)

    # Load the Haar cascade for eye detection
    eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

    # Show a black screen in full screen mode
    cv2.namedWindow('Eye Tracking', cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty('Eye Tracking', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    screen = np.zeros((screen_height, screen_width), dtype=np.uint8)

    # Loop through the calibration points
    for point in calibration_points:
        # Show a dot at the current point for 2 seconds
        screen.fill(0)
        cv2.circle(screen, point, 50, (255, 255, 255), -1)
        cv2.imshow('Eye Tracking', screen)
        cv2.waitKey(2000)

        # Save eye locations for the next 5 seconds
        end_time = time.time() + 5
        while time.time() < end_time:
            # Read a frame from the webcam
            ret, frame = cap.read()
            print(frame.shape)

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
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

            # Show the frame
            cv2.imshow('Eye Tracking', frame)

            # Increment the frame count
            frame_count += 1

            # Check if
            # Save data
            data.append([point, eye_locations])

            # Clear eye locations for next calibration point
            eye_locations = []

            # Reset frame count
            frame_count = 0

        # Show a blank screen for 2 seconds
        screen.fill(0)
        cv2.imshow('Eye Tracking', screen)
        cv2.waitKey(2000)

        # Show a message to indicate the start of the test
        screen.fill(0)
        cv2.putText(screen, 'Test starting in 3 seconds', (screen_width // 2 - 150, screen_height // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow('Eye Tracking', screen)
        cv2.waitKey(3000)

        # Start the test
        end_time = time.time() + duration
        while time.time() < end_time:
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
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Show the frame
            cv2.imshow('Eye Tracking', frame)

            # Increment frame count
            frame_count += 1

        # Release the webcam
        cap.release()

        # Save the data to a CSV file
        with open('eye_tracking_data.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Point', 'Eye Locations'])
            for row in data:
                writer.writerow(row)

        # Show a message to indicate the end of the test
        screen.fill(0)
        cv2.putText(screen, 'Test complete!', (screen_width // 2 - 100, screen_height // 2), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow('Eye Tracking', screen)
        cv2.waitKey(3000)
        cv2.destroyAllWindows()