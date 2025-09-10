# train_image.py
import os
import time
import cv2
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

import numpy as np
from PIL import Image

def ensure_dirs():
    os.makedirs("TrainingImage", exist_ok=True)
    os.makedirs("TrainingImageLabel", exist_ok=True)

def getImagesAndLabels(path):
    imagePaths = [os.path.join(path, f) for f in os.listdir(path) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    faces = []
    Ids = []
    for imagePath in imagePaths:
        pilImage = Image.open(imagePath).convert('L')  # grayscale
        imageNp = np.array(pilImage, 'uint8')
        try:
            Id = int(os.path.split(imagePath)[-1].split(".")[1])
        except Exception:
            continue
        faces.append(imageNp)
        Ids.append(Id)
    return faces, Ids

def TrainImages():
    """
    Train LBPH recognizer and save model to TrainingImageLabel/Trainner.yml
    Returns (True, message) or (False, message)
    """
    ensure_dirs()
    faces, Ids = getImagesAndLabels("TrainingImage")
    if len(faces) == 0:
        return False, "No images found in TrainingImage."

    # create LBPH recognizer (requires opencv-contrib)
    try:
        recognizer = cv2.face.LBPHFaceRecognizer_create()
    except Exception as e:
        return False, f"cv2.face.LBPHFaceRecognizer_create not found: {e}\nInstall opencv-contrib-python."

    recognizer.train(faces, np.array(Ids))
    recognizer.save(f"TrainingImageLabel{os.sep}Trainner.yml")

    # optional progress
    for i in range(len(faces)):
        print(f"Trained on {i+1}/{len(faces)} images", end="\r")
        time.sleep(0.01)

    return True, "Training completed and model saved to TrainingImageLabel/Trainner.yml"
