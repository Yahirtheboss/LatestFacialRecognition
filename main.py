import face_recognition
import os, sys
import cv2
import numpy as np
import math

import pyrebase

import time
import datetime
from datetime import datetime
import pytz


import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore

firebaseMediaConfig = {

}

text_cred = credentials.Certificate(r"C:\Users\qax458\OneDrive - The University of Texas-Rio Grande Valley\Desktop\Facial Recognition\firebaseKey.json")
firebase_admin.initialize_app(text_cred)
db = firestore.client()

firebaseMedia = pyrebase.initialize_app(firebaseMediaConfig)
storage = firebaseMedia.storage()



# Need to try and understand this alittle more, its confidence
def face_confidence(face_distance, face_match_threshold=0.6):
    range = (1.0 - face_match_threshold)
    linear_value = (1.0 - face_distance) / (range * 2.0)

    if face_distance > face_match_threshold:
        return str(round(linear_value * 100, 2)) + '%'
    else:
        value = (linear_value + ((1.0 - linear_value) * math.pow((linear_value - 0.5) * 2, 0.2))) * 100
        return str(round(value, 2)) + '%'
    
    

class FaceRecognition:
    face_locations = []
    face_encodings = []
    face_names = []
    known_face_encodings = []
    known_face_names = []
    process_current_frame = True
    
    def __init__(self):
        self.encode_faces()
    
    def encode_faces(self):
        # Checks the "faces" directory and loads the images that are in that directory
        for image in os.listdir('faces'):
            try:
                face_image = cv2.imread(f'faces/{image}')
                if face_image is None:
                    print(f"Could not read image: {image}")
                    continue
                
                # Convert the image to RGB
                face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
                face_encoding = face_recognition.face_encodings(face_image)[0]

                self.known_face_encodings.append(face_encoding)
                self.known_face_names.append(image)
            except IndexError:
                print(f"No faces found in the image: {image}")
            except Exception as e:
                print(f"Error processing image {image}: {e}")

        # Used for testing purposes to see if the images are actually pulled into the array    
        print(self.known_face_names)

    def run_recognition(self):
        # Using OpenCV to use camera, index 0 depending on the number of cameras connected to computer.
        # Code Editor needs to have permission of the camera.
        video_capture = cv2.VideoCapture(0)
        person_detected = False
        last_person_detected = None

        # These are default values
        confidence = 'Unknown'
        name = 'Unknown'

        while True:
            # result = False if there are no frames to process.
            result, frame =  video_capture.read()
            current_time = datetime.now(pytz.timezone("US/Eastern"))
            formatted_time = current_time.strftime("%B %d,%Y %H:%M:%S %p")

            if not result:
                print("Failed to grab frame!!")
                break

            # This will get rid of the percentage for the image name of known people
            if name != 'Unknown':
                name = name[:-9]

            # This will process every second frame.
            if self.process_current_frame:
                # Resizes the current frame to 1/4 to save computer resources.
                small_frame = cv2.resize(frame, (0, 0), fx = 0.25, fy = 0.25)
                # OpenCV uses BGR format we want RGB format
                rgb_small_frame = small_frame[:, :, ::-1]
                
                # Find all the faces in the current frame
                self.face_locations = face_recognition.face_locations(rgb_small_frame)
                self.face_encodings = face_recognition.face_encodings(rgb_small_frame, self.face_locations)
                
                # Image recognition
                self.face_names = []
                for face_encoding in self.face_encodings:
                    matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
                    
                    face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                    best_match_index = np.argmin(face_distances)
                    
                    # This will give labels
                    if matches[best_match_index]:
                        name = self.known_face_names[best_match_index][:-4]
                        # This removes the last 4 characters of the file name
                        confidence = face_confidence(face_distances[best_match_index])[:-1]
                        # Sets formatted_confidence
                        if (confidence == "Unknown"):
                            formatted_confidence = 0
                        else:
                            formatted_confidence = confidence

                        date_time = formatted_time[:27]

                        if person_detected:
                            if name != last_detected_person:  # Check for new person
                                # Update picture for the new person
                                file_path = os.path.join("temp_capture.png")
                                cv2.imwrite(file_path, frame)
                                # Filename that will be uploaded, name of the picture you have
                                storage.child(date_time + ".png").put("temp_capture.png")
                                print(f"New image saved and uploaded for: {name} found in {file_path}")
                                # This will format and upload the log to Firebase
                                if (float(formatted_confidence) >= 85.0):
                                    data = {
                                        'Person':  f"{name} {confidence} Detected"
                                    }

                                    doc_ref = db.collection('Facial Recognition').document(date_time)
                                    doc_ref.set(data)
                                    print('A new log has been created')

                                last_detected_person = name  # Update last detected person
                        else:
                            # First time detecting a person
                            person_detected = True
                            last_detected_person = name
                            file_path = os.path.join("temp_capture.png")
                            time.sleep(0.5)
                            cv2.imwrite(file_path, frame)
                            print(f"Image saved at: {file_path}")

                    self.face_names.append(f'{name} ({confidence})') 
                    
            # This will happen every frame to save computer resources
            self.process_current_frame = not self.process_current_frame
            
            # Display Annotations                   # Zip put these 2 things together and pulls from them
            for (top, right, bottom, left), name in zip(self.face_locations, self.face_names):
                # This brings back the normal dimensions
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4
                
                # Setting frame with color and thickness (BGR)
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                # This will fill the square instead of giving a thickness (The area where the name is going to be)
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), -1)
                # Put Text 6 pixels more to right and 6 pixels higher with the OPENCV font, font size, color, thickness
                cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 0, 0), 2)
                
            cv2.imshow('Face Recognition', frame)
            
            # Break Character for 1 millisecond
            if cv2.waitKey(1) == ord('q'):
                break

        last_detected_person = None
        video_capture.release()
        cv2.destroyAllWindows()
        
# TODO: Distance metric
# TODO: Set the proper time, RIGHT NOW its one hour ahead.

if __name__ == '__main__':
    fr = FaceRecognition()   
    fr.run_recognition()