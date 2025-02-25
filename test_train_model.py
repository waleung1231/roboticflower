import tensorflow as tf
import cv2
import numpy as np
import os

# Set environment variables for display
os.environ['DISPLAY'] = ':0'
os.environ['QT_QPA_PLATFORM'] = 'xcb'

# Suppress TensorFlow warnings
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

def predict_smile(model, face_img):
    """Predict if the face is smiling using our trained model"""
    try:
        # Preprocess the image
        face_img = cv2.resize(face_img, (64, 64))
        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        face_img = face_img / 255.0
        face_img = face_img.reshape(1, 64, 64, 1)
        
        # Make prediction
        prediction = model.predict(face_img, verbose=0)[0][0]
        confidence = prediction if prediction > 0.5 else 1 - prediction
        return prediction > 0.5, confidence
    except Exception as e:
        print(f"Error in smile prediction: {e}")
        return False, 0.0

def main():
    try:
        # Load the trained model
        print("Loading model...")
        model = tf.keras.models.load_model('smile_detector.h5')
        print("Model loaded successfully!")

        # Initialize webcam
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

        print("Starting test. Press 'q' to quit.")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                continue
            
            # Detect faces
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            
            # Draw text indicating if faces are found
            if len(faces) == 0:
                cv2.putText(frame, "No face detected", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            for (x, y, w, h) in faces:
                face_roi = frame[y:y+h, x:x+w]
                is_smiling, confidence = predict_smile(model, face_roi)
                
                # Set rectangle color based on smile detection
                rect_color = (0, 255, 0) if is_smiling else (0, 0, 255)
                
                # Draw rectangle around face
                cv2.rectangle(frame, (x, y), (x+w, y+h), rect_color, 2)
                
                # Add text labels
                status = "Smiling" if is_smiling else "Not Smiling"
                conf_text = f"Confidence: {confidence*100:.1f}%"
                
                cv2.putText(frame, status, (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, rect_color, 2)
                cv2.putText(frame, conf_text, (x, y+h+25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, rect_color, 2)
            
            # Show the frame
            cv2.imshow('Smile Detector Test', frame)
            
            # Check for quit command
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Quit command received")
                break

    except Exception as e:
        print(f"Error: {e}")
    finally:
        print("Cleaning up...")
        cap.release()
        cv2.destroyAllWindows()
        print("Test completed")

if __name__ == "__main__":
    main()