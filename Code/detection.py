import cv2 
import numpy as np
import mtcnn
from architecture import *
from train_v2 import normalize,l2_normalizer
from scipy.spatial.distance import cosine
from tensorflow.python.keras.models import load_model
import pickle
import sqlite3
from datetime import datetime


############

conn = sqlite3.connect('Data/college.db')

conn.execute('''CREATE TABLE IF NOT EXISTS faces(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            Name varchar(50) NOT NULL,
            Accuracy varchar(50) NOT NULL,
            Time varchar(50) NOT NULL) ''')

conn.commit()
############

confidence_t=0.99
recognition_t=0.4
required_size = (160,160)
time_list = []
mm_time = []


def get_face(img, box):
    x1, y1, width, height = box
    x1, y1 = abs(x1), abs(y1)
    x2, y2 = x1 + width, y1 + height
    face = img[y1:y2, x1:x2]
    return face, (x1, y1), (x2, y2)

def get_encode(face_encoder, face, size):
    face = normalize(face)
    face = cv2.resize(face, size)
    encode = face_encoder.predict(np.expand_dims(face, axis=0))[0]
    return encode


def load_pickle(path):
    with open(path, 'rb') as f:
        encoding_dict = pickle.load(f)
    return encoding_dict

def detect(img ,detector,encoder,encoding_dict):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = detector.detect_faces(img_rgb)
    for res in results:
        if res['confidence'] < confidence_t:
            continue
        face, pt_1, pt_2 = get_face(img_rgb, res['box'])
        encode = get_encode(encoder, face, required_size)
        encode = l2_normalizer.transform(encode.reshape(1, -1))[0]
        name = 'unknown'

        distance = float("inf")
        for db_name, db_encode in encoding_dict.items():
            dist = cosine(db_encode, encode)
            if dist < recognition_t and dist < distance:
                name = db_name
                distance = dist

        if name == 'unknown':
            cv2.rectangle(img, pt_1, pt_2, (0, 0, 255), 2)
            cv2.putText(img, name, pt_1, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)


            now = datetime.now()
            current_time = now.strftime("%H:%M:%S")
            time_list.append(current_time)
            time_list.sort() 
            min_time = time_list[0]
            time_list.clear()
            print(name, distance, min_time)
            # sql = "INSERT INTO `rsb`.`faces` (`name`, `time`, `accuracy`) VALUES (%s, %s, %s);" 
            # sql = "INSERT INTO rsb.faces (name, time, accuracy ) \ VALUES (%s, %s, %s)"
            # val = (str(name), min_time, "0")
            
            # conn.execute(sql,val)
            # conn.commit()

            conn.execute("INSERT INTO faces (Name,Time,Accuracy) VALUES (?, ?, ?)", (str(name), min_time, str(distance)))
            conn.commit()
            


        else:            
            cv2.rectangle(img, pt_1, pt_2, (0, 255, 0), 2)
            cv2.putText(img, name + f'__{distance:.2f}', (pt_1[0], pt_1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 200, 200), 2)

            now = datetime.now()
            current_time = now.strftime("%H:%M:%S")
            time_list.append(current_time)
            time_list.sort()
            min_time = time_list[0]
            time_list.clear()
            # found_name = name
            # break
        
            print(name, distance, min_time)
            # sql = "INSERT INTO `rsb`.`faces` (`name`, `time`, `accuracy`) VALUES (%s, %s, %s);"
            # sql = "INSERT INTO rsb.faces (name, time, accuracy ) \ VALUES (%s, %s, %s)"
            # val = (str(name), min_time, str(distance))
            
            # conn.execute(sql,val)
            # conn.commit()
            conn.execute("INSERT INTO faces (Name,Time,Accuracy) VALUES (?, ?, ?)", (str(name), min_time, str(distance)))
            conn.commit()

    return img 



# if __name__ == "__main__":
required_shape = (160,160)
face_encoder = InceptionResNetV2()
path_m = "Data/facenet_keras_weights.h5"
face_encoder.load_weights(path_m)
encodings_path = 'Data/encodings/encodings.pkl'
face_detector = mtcnn.MTCNN()
encoding_dict = load_pickle(encodings_path)

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret,frame = cap.read()

    if not ret:
        print("CAM NOT OPEND") 
        break
    
    frame= detect(frame , face_detector , face_encoder , encoding_dict)

    cv2.imshow('camera', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

conn.close()
cv2.destroyAllWindows()
# cap.release()
    


