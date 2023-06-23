import cv2
import numpy as np
import os


# KNN Code
def distance(v1, v2):
    # Eucledian
    return np.sqrt(((v1-v2)**2).sum())


def knn(train, test, k=5):
    dist = []

    for i in range(train.shape[0]):
        # to get vector and label
        ix = train[i, :-1]
        iy = train[i, -1]
        # compute the distance from test points
        d = distance(test, ix)
        dist.append([d, iy])

    dk = sorted(dist, key=lambda x: x[0])[:k]

    labels = np.array(dk)[:, -1]

    # get feq of each labels
    output = np.unique(labels, return_counts=True)

    # max freq and coresponding labels
    index = np.argmax(output[1])
    return output[0][index]

# ----------------------------------------------------------------


# INIIT Camera
cap = cv2.VideoCapture(0)

# Face detection file
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")

skip = 0
dataset_path = './data/'
face_data = []
label = []

class_id = 0  # Labels for the given file
names = {}  # Mapping btw id - name


# data preparation

for fx in os.listdir(dataset_path):
    if fx.endswith('.npy'):
        # create a mapping btw class_id and name
        names[class_id] = fx[:-4]
        print("Loaded file "+fx)
        data_item = np.load(dataset_path+fx)
        face_data.append(data_item)

        # creat labels for the class
        target = class_id*np.ones((data_item.shape[0],))
        class_id += 1
        label.append(target)

face_dataset = np.concatenate(face_data, axis=0)
face_labels = np.concatenate(label, axis=0).reshape((-1, 1))

print(face_dataset.shape)
print(face_labels.shape)


trainset = np.concatenate((face_dataset, face_labels), axis=1)
print(trainset.shape)


# testing

while True:
    ret, frame = cap.read()
    if ret == False:
        continue

    faces = face_cascade.detectMultiScale(frame, 1.3, 5)
    if (len(faces) == 0):
        continue

    for face in faces:
        x, y, w, h = face

        offset = 10
        face_section = frame[y-offset:y+h+offset, x-offset:x+w+offset]
        face_section = cv2.resize(face_section, (100, 100))

        out = knn(trainset, face_section.flatten())

        # display on the screen the name and rectangle around it
        pred_name = names[int(out)]
        cv2.putText(frame, pred_name, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)

    cv2.imshow("Faces", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
