import os
import cv2
import numpy as np
from keras.utils.np_utils import to_categorical
from keras.layers import MaxPooling2D, Dense, Flatten, Conv2D
from keras.models import Sequential, model_from_json
import pickle

def getIrisFeatures(image):
    img = cv2.imread(image, 0)
    img = cv2.medianBlur(img, 5)
    cimg = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 10, param1=63, param2=70, minRadius=0, maxRadius=0)
    if circles is not None:
        mask = np.zeros_like(img)
        for i in circles[0, :]:
            cv2.circle(mask, (i[0], i[1]), int(i[2]), (255, 255, 255), thickness=-1)
            break  # Use only the first circle detected
        masked_data = cv2.bitwise_and(img, img, mask=mask)
        x, y, w, h = cv2.boundingRect(mask)
        crop = masked_data[y:y+h, x:x+w]
        return 'success', cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
    return 'failed', None

def getID(name, labels):
    if name not in labels:
        labels.append(name)
    return labels.index(name)

path = 'CASIA1'
labels = []
X_train = []
Y_train = []

for root, dirs, files in os.walk(path):
    for file in files:
        if 'Thumbs.db' not in file:
            status, img = getIrisFeatures(os.path.join(root, file))
            if status == 'success':
                img = cv2.resize(img, (64, 64))
                X_train.append(img)
                ids = getID(os.path.basename(root), labels)
                Y_train.append(ids)

X_train = np.asarray(X_train).astype('float32') / 255.0
Y_train = to_categorical(np.asarray(Y_train), num_classes=len(labels))

os.makedirs('model', exist_ok=True)
np.save('model/X.txt', X_train)
np.save('model/Y.txt', Y_train)

if os.path.exists('model/model.json'):
    with open('model/model.json', "r") as json_file:
        model = model_from_json(json_file.read())
    model.load_weights("model/model_weights.h5")
    model._make_predict_function()
    with open('model/history.pckl', 'rb') as f:
        data = pickle.load(f)
    accuracy = data['accuracy'][-1] * 100
    print(f"Training Model Accuracy = {accuracy:.2f}%")
else:
    model = Sequential([
        Conv2D(32, (3, 3), input_shape=(64, 64, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(256, activation='relu'),
        Dense(len(labels), activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    hist = model.fit(X_train, Y_train, batch_size=16, epochs=60, shuffle=True, verbose=2)
    model.save_weights('model/model_weights.h5')
    with open("model/model.json", "w") as json_file:
        json_file.write(model.to_json())
    with open('model/history.pckl', 'wb') as f:
        pickle.dump(hist.history, f)
    accuracy = hist.history['accuracy'][-1] * 100
    print(f"Training Model Accuracy = {accuracy:.2f}%")
