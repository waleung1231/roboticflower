import tensorflow as tf
import cv2
import numpy as np
import xarm
import time
import os

# Set display variable for Raspberry Pi
os.environ['DISPLAY'] = ':0'

# Suppress TensorFlow warnings
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

try:
    # Load the trained model
    print("Loading model...")
    model = tf.keras.models.load_model('smile_detector.h5')
    print("Model loaded successfully!")

    # Initialize XArm and webcam
    print("Initializing XArm...")
    arm = xarm.Controller('USB')
    print("XArm initialized!")

    print("Initializing camera...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise Exception("Could not open camera")
    print("Camera initialized!")

    # Initialize face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    if face_cascade.empty():
        raise Exception("Error loading face cascade classifier")
    print("Face detection initialized!")

    # Initial arm positions
    a, b, c, d, e = 800, -90.25, 51.49999999999999, -81.75, 19.749999999999993
    arm.setPosition([[1,800],[2,-90.25],[3,51.49],[4,-81.75],[5,8.25]], wait=True)
    print("Arm positioned to initial position")

    def predict_smile(face_img):
        """Predict if the face is smiling using our trained model"""
        try:
            # Preprocess the image
            face_img = cv2.resize(face_img, (64, 64))
            face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
            face_img = face_img / 255.0
            face_img = face_img.reshape(1, 64, 64, 1)
            
            # Make prediction
            prediction = model.predict(face_img, verbose=0)[0][0]  # Added verbose=0 to suppress progress bar
            return prediction > 0.5
        except Exception as e:
            print(f"Error in smile prediction: {e}")
            return False

    print("Starting main loop. Press 'q' to quit.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            continue
        
        try:
            # Detect faces
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            
            for (x, y, w, h) in faces:
                face_roi = frame[y:y+h, x:x+w]
                is_smiling = predict_smile(face_roi)
                
                if not is_smiling:
                    if (a<800 and c<51.49 and d>-81.75 and e<25.75):
                        a += 40*2
                        c += 2.57*2
                        d -= 4.08*2
                        e += 0.98*2
                        arm.setPosition([[1, a], [2, b], [3, c], [4, d], [5, e]], wait=False)
                    else:
                        arm.setPosition([[1,800],[2,-90.25],[3,51.49],[4,-81.75],[5,25.7499]], wait=False)
                    rect_color = (0, 0, 255)  # Red when not smiling
                else:
                    if (a>200 and c>-2.25 and d<2.25 and e>8.25):
                        a -= 40*2
                        c -= 2.57*2
                        d += 4.08*2
                        e -= 0.98*2
                        arm.setPosition([[1, a], [2, b], [3, c], [4, d], [5, e]], wait=False)
                    else:
                        arm.setPosition([[1,200],[2,-90.25],[3,-2.75],[4,2.25],[5,8.250]], wait=False)
                    rect_color = (0, 255, 0)  # Green when smiling
                
                cv2.rectangle(frame, (x, y), (x+w, y+h), rect_color, 2)
                time.sleep(0.15)
            
            # Show the frame
            cv2.imshow('Smile Detection', frame)
            
        except Exception as e:
            print(f"Error in main loop: {e}")
            continue
        
        # Check for quit command
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Quit command received")
            break

except Exception as e:
    print(f"Fatal error: {e}")
finally:
    print("Cleaning up...")
    try:
        cap.release()
        cv2.destroyAllWindows()
    except:
        pass
    print("Program terminated")
