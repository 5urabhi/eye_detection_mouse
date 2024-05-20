

---


# Eye-Controlled Virtual Keyboard

This repository contains a Python implementation of an eye-controlled virtual keyboard. The project leverages computer vision and machine learning techniques to track the right eye, detect blinks, and project the eye's centroid onto a virtual keyboard. Inputs are taken using blink detection, and the resulting text is displayed on the screen and can be saved to a file.

## Overview

The main logic flow of the project is as follows:

1. **Face Detection and Eye Tracking**: Find the face and track the right eye.
2. **Calibration**: Calibrate the proper range of allowed space for comfortable movement.
3. **Frame-by-Frame Processing**:
    - Track the right eye and project its centroid on a virtual keyboard in a "working window".
    - Detect blinks to "press" keys.
    - Update a text string and print it on the screen (and optionally save it to a file).

## Requirements

- Python 2 or Python 3
- OpenCV
- NumPy
- dlib (with the trained model `shape_predictor_68_face_landmarks.dat`)

## Installation

1. **Clone the repository**:
   ```sh
   git clone https://github.com/your-username/Eye-Controlled-Virtual-Keyboard.git
   cd Eye-Controlled-Virtual-Keyboard
   ```

2. **Install dependencies**:
   ```sh
   pip install opencv-python numpy dlib cmake
   ```

3. **Install dlib**:
   Download the latest version of dlib from [this link](https://pypi.org/simple/dlib/). Extract the contents and open `setup.py` to get the commands for installation.

4. **Download the trained model**:
   Download the `shape_predictor_68_face_landmarks.dat` file from [here](https://github.com/davisking/dlib-models/blob/master/shape_predictor_68_face_landmarks.dat.bz2). Extract the file and place it in the project directory.

## Running the Code

Execute the main script:
```sh
python eye_key.py
```
## Result

![eye_mouse](https://github.com/5urabhi/eye_detection_mouse/assets/104481755/c5621683-ab59-41e8-a845-c565b38a44ab)

## Additional Resources

- **Facial mapping (landmarks) with Dlib + Python**:
  - [Towards Data Science article](https://towardsdatascience.com/facial-mapping-landmarks-with-dlib-python-160abcf7d672)
  - [PyImageSearch article](https://www.pyimagesearch.com/2017/04/03/facial-landmarks-dlib-opencv-python/)

- **Trained Model File**:
  - Download the `shape_predictor_68_face_landmarks.dat` from [dlib-models repository](https://github.com/davisking/dlib-models/blob/master/shape_predictor_68_face_landmarks.dat.bz2)

- **YouTube Video**:
  - [Demonstration Video](https://www.youtube.com/watch?v=LCfCTPDiFnc)
 

## Acknowledgments

This project combines several codes and resources:
- Facial mapping and landmarks detection using Dlib and Python.
- The trained model for facial landmarks from the dlib models repository.

Feel free to explore and contribute to this project by submitting issues or pull requests.

---

