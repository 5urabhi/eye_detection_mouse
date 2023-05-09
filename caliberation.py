import csv
import time
import cv2
import numpy as np
from pynput.mouse import Listener
from tkinter import *
import tkinter.font as font
import pyautogui

# Function to handle mouse move events
def on_move(x, y):
    pass

# Function to handle mouse click events
def on_click(x, y, button, pressed):
    pass

# Function to handle mouse scroll events
def on_scroll(x, y, dx, dy):
    pass

# Initialize mouse listener
mouse_listener = Listener(on_move=on_move, on_click=on_click, on_scroll=on_scroll)
mouse_listener.start()

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Load the Haar cascade for eye detection
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

# Get the screen resolution
root = Tk()
root.withdraw()
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()

# Initialize variables for eye tracking and calibration
calibration_positions = [
    (int(screen_width/2), int(screen_height/2)), # Center
    (int(screen_width/4), int(screen_height/2)), # Left
    (int(screen_width/4*3), int(screen_height/2)), # Right
    (int(screen_width/2), int(screen_height/4)), # Top
    (int(screen_width/2), int(screen_height/4*3)) # Bottom
]
calibration_dots = []

# Draw calibration dots
for position in calibration_positions:
    dot = np.zeros((screen_height, screen_width, 3), dtype=np.uint8)
    cv2.circle(dot, position, 10, (0, 255, 0), -1)
    calibration_dots.append(dot)

# Show calibration dots one by one and record eye locations
eye_locations = []
for dot in calibration_dots:
    # Show dot on full screen
    cv2.namedWindow('Calibration', cv2.WINDOW_NORMAL)
    cv2.setWindowProperty('Calibration', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow('Calibration', dot)
    cv2.waitKey(3000)

    # Record eye locations
    for i in range(5):
        # Read a frame from the webcam
        ret, frame = cap.read()

        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect eyes in the frame
        eyes = eye_cascade.detectMultiScale(gray, 1.3, 5)

        # Save eye locations
        eye_locations.append(eyes)

    # Close the window
    cv2.destroyAllWindows()

# Release the webcam
cap.release()

# Save eye locations to CSV file
with open('eye_data.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['x', 'y', 'w', 'h'])
    for eyes in eye_locations:
        for (x, y, w, h) in eyes:
            writer.writerow([x, y, w, h])

# Stop mouse listener
mouse_listener.stop()

print('Eye tracking data saved to eye_data.csv')
