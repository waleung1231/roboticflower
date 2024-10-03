import xarm
import cv2
import time

arm = xarm.Controller('USB')

smile_threshold = 10

# Load Haar Cascade classifiers
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

# Initialize the webcam
cap = cv2.VideoCapture(0)

a = 800
b = -90.25
c = 51.49999999999999
d = -81.75
e = 19.749999999999993
arm.setPosition([[1,800],[2,-90.25],[3,51.49],[4,-81.75],[5,8.25]], wait=True)

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        smiles = smile_cascade.detectMultiScale(roi_gray, scaleFactor=1.8, minNeighbors=20, minSize=(25, 25))
        
        if type(smiles) is tuple:
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

    cv2.imshow('Smile Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

