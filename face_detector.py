import os
import cv2
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import pickle

# Define paths
DATA_DIR = "./train_data"
MODEL_FILE = "face_recognition_model.pkl"

# Step 1: Preprocess images to create a dataset
def preprocess_images():
    if not os.path.exists(DATA_DIR):
        print(f"Error: Data directory '{DATA_DIR}' does not exist.")
        return []

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    X, y = [], []

    for idx, filename in enumerate(os.listdir(DATA_DIR)):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            img_path = os.path.join(DATA_DIR, filename)
            img = cv2.imread(img_path)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            for (x, y_, w, h) in faces:
                face = gray[y_:y_ + h, x:x + w]
                face_resized = cv2.resize(face, (100, 100))  # Resize to consistent size
                X.append(face_resized.flatten())
                y.append(1)  # Label all faces as "1" for matching

    return np.array(X), np.array(y)

# Step 2: Train the face recognition model
def train_model():
    X, y = preprocess_images()
    if len(X) == 0:
        print("No images processed for training.")
        return

    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X, y)

    with open(MODEL_FILE, "wb") as f:
        pickle.dump(knn, f)

    print("Model trained and saved.")

# Step 3: Capture image and perform recognition
def capture_and_recognize():
    if not os.path.exists(MODEL_FILE):
        print("Error: Model file does not exist. Train the model first.")
        return

    with open(MODEL_FILE, "rb") as f:
        knn = pickle.load(f)

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    cap = cv2.VideoCapture(0)

    print("Press 'c' to capture your image.")

    while True:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        cv2.imshow("Face Detection", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('c'):
            if len(faces) == 0:
                print("No face detected. Try again.")
            else:
                for (x, y, w, h) in faces:
                    face = gray[y:y + h, x:x + w]
                    face_resized = cv2.resize(face, (100, 100)).flatten()
                    prediction = knn.predict([face_resized])[0]

                    if prediction == 1:
                        print("Face match found : Hello Sam!")
                    else:
                        print("No match found : Hello, Stranger!")

                break

        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Main script
if __name__ == "__main__":
    train_model()
    capture_and_recognize()
    
