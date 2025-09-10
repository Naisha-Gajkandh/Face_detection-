import os
import time
import datetime
import cv2
import pandas as pd

def recognize_attendance():
    """
    Run realtime recognition and save attendance instantly when a face passes threshold.
    """
    # Ensure Attendance directory exists
    os.makedirs("Attendance", exist_ok=True)

    # Student details file
    csv_students = f"StudentDetails{os.sep}StudentDetails.csv"
    if not os.path.isfile(csv_students):
        return False, "StudentDetails.csv not found. Capture faces first."

    # Model file
    model_path = f"TrainingImageLabel{os.sep}Trainner.yml"
    if not os.path.isfile(model_path):
        return False, "Trainner.yml not found. Run training first."

    # Load recognizer
    try:
        recognizer = cv2.face.LBPHFaceRecognizer_create()
    except Exception as e:
        return False, f"cv2.face.LBPHFaceRecognizer_create not found: {e}"
    recognizer.read(model_path)

    # Face detector
    faceCascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    # Read student details with Id as int
    try:
        df = pd.read_csv(csv_students, dtype={'Id': int})
    except Exception as e:
        return False, f"Failed to read StudentDetails.csv: {e}"

    if 'Id' not in df.columns or 'Name' not in df.columns:
        return False, "StudentDetails.csv is missing required columns 'Id' and 'Name'."

    font = cv2.FONT_HERSHEY_SIMPLEX
    col_names = ['Id', 'Name', 'Date', 'Time']

    # Prepare today's attendance file
    today_date = datetime.date.today().strftime('%Y-%m-%d')  
    file_name = f"Attendance{os.sep}Attendance_{today_date}.csv"

    if os.path.exists(file_name):
        attendance = pd.read_csv(file_name, dtype={'Id': int})
    else:
        attendance = pd.DataFrame(columns=col_names)

    # Open camera
    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        return False, "Cannot open webcam."

    cam.set(3, 640)  # Width
    cam.set(4, 480)  # Height
    minW = 0.1 * cam.get(3)
    minH = 0.1 * cam.get(4)

    try:
        while True:
            ret, im = cam.read()
            if not ret:
                break
            gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            faces = faceCascade.detectMultiScale(
                gray, 1.2, 5,
                minSize=(int(minW), int(minH)),
                flags=cv2.CASCADE_SCALE_IMAGE
            )

            for (x, y, w, h) in faces:
                cv2.rectangle(im, (x, y), (x+w, y+h), (10, 159, 255), 2)
                Id, conf = recognizer.predict(gray[y:y+h, x:x+w])
                conf_val = 100 - conf

                if conf_val > 50:  # Threshold
                    # Match ID to name
                    name_arr = df.loc[df['Id'] == Id]['Name'].values
                    name = name_arr[0] if len(name_arr) > 0 else "Unknown"
                    tt = f"{Id}-{name}"
                    ts = time.time()
                    date = datetime.date.today().strftime('%Y-%m-%d')
                    timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')

                    if name != "Unknown" and Id != "Unknown":
                        if not ((attendance['Id'] == Id) & (attendance['Date'] == date)).any():
                            attendance.loc[len(attendance)] = [Id, name, date, timeStamp]
                            attendance.to_csv(file_name, index=False)
                            print(f"[SAVED] {name} ({Id}) at {timeStamp} with {round(conf_val)}% confidence")

                    display_text = f"{tt} [Pass]"
                else:
                    name = 'Unknown'
                    display_text = "Unknown"

                cv2.putText(im, display_text, (x+5, y-5), font, 1, (255, 255, 255), 2)
                color = (0, 255, 0) if conf_val > 50 else ((0, 255, 255) if conf_val > 40 else (0, 0, 255))
                confstr = f"{round(conf_val)}%"
                cv2.putText(im, confstr, (x + 5, y + h - 5), font, 1, color, 1)

            cv2.imshow('Attendance - press q to stop', im)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cam.release()
        cv2.destroyAllWindows()

    return True, f"Attendance saved to {file_name}"
