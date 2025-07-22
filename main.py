import cv2
from deepface import DeepFace
#cv2 is the OpenCV library used for image and video processing,DeepFace is a library for facial recognition and analysis.

#CascadeClassifier is a class in OpenCV used for object detection,haarcascade_frontalface_default.xml is a pre-trained model for detecting faces.
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# opens the default camera 
cap = cv2.VideoCapture(0)

#This starts an infinite loop to continuously process video frames.
while True:

#cap.read() captures a frame from the camera,ret is a boolean indicating if the frame was read correctly.frame contains the actual image data of the frame.
    ret, frame = cap.read()

    #Converts the frame from color (BGR) to grayscale. This simplifies processing and improves performance for face detection.
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rgb_frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2RGB)

# detects faces in the grayscale image.
    # scaleFactor compensates for faces appearing at different sizes.
    # minNeighbors specifies how many neighbors each candidate rectangle should have to retain it.
    # minSize specifies the minimum size of the detected faces.
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
 
    #Iterates over each detected face's bounding box.
    for (x, y, w, h) in faces:
      

#face_roi: extracts the region of interest (the face) from the RGB frame.
# DeepFace:analyze analyzes the face for emotions.
# result :contains the analysis results, and emotion extracts the dominant emotion.

        face_roi = rgb_frame[x:x + h, y:y + w]
        result = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)
        emotion = result[0]['dominant_emotion']

 #cv2.rectangle draws a rectangle around the detected face.
    # cv2.putText adds text above the rectangle to show the detected emotion.
    #Displays the current frame with detected faces and emotions in a window titled "Real-time Emotion Detection".

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    cv2.imshow('Real-time Emotion Detection', frame)
   #cv2.waitKey(1) waits for a key event for 1 millisecond.
   # & 0xFF ensures compatibility across different platforms.
   # If the key 'q' is pressed, the loop breaks.
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

#cap.release() releases the camera.
# cv2.destroyAllWindows() closes all OpenCV windows.
cap.release()
cv2.destroyAllWindows()