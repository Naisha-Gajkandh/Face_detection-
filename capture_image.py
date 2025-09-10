import csv
import cv2
import os

def ensure_dirs():
    os.makedirs("TrainingImage", exist_ok=True)
    os.makedirs("StudentDetails", exist_ok=True)

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass
    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass
    return False

def takeImages(Id=None, name=None, samples=100):
    """
    Capture face images from webcam and save to TrainingImage folder.
    If Id or name are None, will ask via console (but Kivy will pass them).
    Returns (True, message) or (False, error_message).
    """
    ensure_dirs()

    if Id is None:
        Id = input("Enter Your Id: ").strip()
    if name is None:
        name = input("Enter Your Name: ").strip()

    if not (is_number(Id) and name.isalpha()):
        msg = ""
        if not is_number(Id):
            msg += "Enter Numeric ID. "
        if not name.isalpha():
            msg += "Enter Alphabetical Name."
        return False, msg

    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        return False, "Cannot open webcam."

    detector = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    sampleNum = 0

    try:
        while True:
            ret, img = cam.read()
            if not ret:
                break
            # ✅ convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(gray, 1.3, 5, minSize=(20, 20))

            for (x, y, w, h) in faces:
                sampleNum += 1
                print(f"[{sampleNum}] Face detected - saving image")
                cv2.rectangle(img, (x, y), (x+w, y+h), (10, 159, 255), 2)
                filepath = f"TrainingImage{os.sep}{name}.{Id}.{sampleNum}.jpg"
                cv2.imwrite(filepath, gray[y:y+h, x:x+w])
                cv2.imshow('Capturing Faces - press q to stop', img)

            if cv2.waitKey(100) & 0xFF == ord('q'):
                break
            if sampleNum >= samples:
                break
    finally:
        cam.release()
        cv2.destroyAllWindows()

    # ✅ Save student info to CSV
    csv_path = f"StudentDetails{os.sep}StudentDetails.csv"
    header = ["Id", "Name"]
    row = [int(Id), name]
    if os.path.isfile(csv_path):
        with open(csv_path, 'a', newline='') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerow(row)
    else:
        with open(csv_path, 'w', newline='') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerow(header)
            writer.writerow(row)

    return True, f"Images saved for ID: {Id} Name: {name}"
