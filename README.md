# Face Detector.
## Overview
This project implements a Face Detection system using OpenCV. 
The system detects the face of a person in real-time from a webcam feed and returns an output.

## Features
- Real-time face detection from a webcam feed.
- Detects faces in static images.
- Draws bounding boxes around detected faces.

## Installation
### 1. Clone the repository
```
git clone https://github.com/Samuel-Njoroge/face-detector
cd face-detector
```
### 2. Install dependencies
Create a virtual environment for the project:
```
python -m venv venv
```
Activate the virtual environment.
```
Linux : source venv/bin/activate
Windows : venv\Scripts\activate
```

Install the required Python libraries:

```
pip install -r requirements.txt
```

### 3. Download Haar Cascade Classifier
The default classifier file (haarcascade_frontalface_default.xml) is required to run the system. You can find it in the `xml_files` directory.

### Usage
#### Real-Time Face Detection (Webcam Input)
1. Run the following command to start face detection from your webcam:

```
python face_detector.py
```
2. The webcam window will open.
3. The system will attempt to detect faces in real-time.
4. If faces are detected, bounding boxes will appear around them.
5. Press 'q' to exit the program.
6. Detect Faces in an Image

### Code Structure
```bash
face-detector/
│
├── face_detector.py         # Main script for running face detection (handles webcam & image input)
├── xml_files/face_data.xml  # Pre-trained Haar Cascade Classifier model for face detection
├── requirements.txt         # List of required Python libraries (OpenCV, NumPy)
├── README.md                # Project documentation (how to set up, use, and contribute)
│
├── train_data/                # Folder to store image files for testing
    ├── image11.jpg            # Example image 1 for testing face detection
    └── image12.jpg            # Example image 2 for testing face detection
```

### How It Works
- Loading Haar Cascade Classifier: The system loads a pre-trained Haar Cascade Classifier model using OpenCV's cv2.CascadeClassifier method.
- Reading Input: The system reads a live feed from the webcam.
- Converting to Grayscale: Since face detection works better on grayscale images, the input is converted to grayscale using OpenCV's cv2.cvtColor() method.
- Face Detection: The detectMultiScale() method is used to detect faces in the image. This method returns the coordinates of bounding boxes around the detected faces.
- Displaying the Output: If the input is from a webcam, the system displays the webcam feed with the detected faces in real-time. For images, the modified image is displayed with bounding boxes.

### License
This project is licensed under the [MIT License](https://opensource.org/license/mit).
