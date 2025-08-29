import cv2
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk

# Haarcascade classifiers
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye_tree_eyeglasses.xml")
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_smile.xml")


class FaceDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Face | Eyes | Smile Detection")
        self.root.geometry("800x600")
        self.root.configure(bg="#2b2b2b")

        # UI Elements
        self.label = tk.Label(self.root, bg="#2b2b2b")
        self.label.pack()

        self.start_btn = tk.Button(root, text="Start Camera", command=self.start_camera,
                                   bg="#4CAF50", fg="white", font=("Arial", 12, "bold"))
        self.start_btn.pack(side="left", padx=20, pady=10)

        self.stop_btn = tk.Button(root, text="Stop Camera", command=self.stop_camera,
                                  bg="#f44336", fg="white", font=("Arial", 12, "bold"))
        self.stop_btn.pack(side="right", padx=20, pady=10)

        # Variables
        self.cap = None
        self.running = False
        self.frame_count = 0
        self.smile_counter = 0  # for continuous smile check

    def start_camera(self):
        if not self.running:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                messagebox.showerror("Error", "Cannot open camera")
                return
            self.running = True
            self.update_frame()

    def stop_camera(self):
        if self.running:
            self.running = False
            self.cap.release()
            self.label.config(image='')

    def detect_features(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=6, minSize=(80, 80))

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 3)
            cv2.putText(frame, "Face", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            roi_gray = gray[y:y + h, x:x + w]
            roi_color = frame[y:y + h, x:x + w]

            # Eyes detection
            eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=12, minSize=(25, 25))
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
                cv2.putText(roi_color, "Eye", (ex, ey - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Smile detection (every 3rd frame + continuous check)
            if self.frame_count % 3 == 0:
                smiles = smile_cascade.detectMultiScale(
                    roi_gray,
                    scaleFactor=1.8,
                    minNeighbors=35,
                    minSize=(60, 60)
                )
                if len(smiles) > 0:
                    self.smile_counter += 1
                else:
                    self.smile_counter = 0

                # Show "Smile" only if detected for >8 frames (~0.3 sec)
                if self.smile_counter > 8:
                    for (sx, sy, sw, sh) in smiles:
                        cv2.rectangle(roi_color, (sx, sy), (sx + sw, sy + sh), (255, 255, 0), 2)
                        cv2.putText(roi_color, "Smile", (sx, sy - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

        return frame

    def update_frame(self):
        if self.running:
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.flip(frame, 1)
                self.frame_count += 1

                # Detect features
                frame = self.detect_features(frame)

                # FPS Display
                fps = int(self.cap.get(cv2.CAP_PROP_FPS))
                cv2.putText(frame, f"FPS: {fps}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                # Convert to Tkinter format
                img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(img)
                imgtk = ImageTk.PhotoImage(image=img)
                self.label.imgtk = imgtk
                self.label.configure(image=imgtk)

            self.root.after(10, self.update_frame)


if __name__ == "__main__":
    root = tk.Tk()
    app = FaceDetectionApp(root)
    root.mainloop()
