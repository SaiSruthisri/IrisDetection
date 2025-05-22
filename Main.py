import tkinter as tk
from tkinter import filedialog, messagebox, Text, Label, Button, Scrollbar
import numpy as np
from tensorflow.keras.models import model_from_json
import pickle
import cv2
import os

# Initialize main window
main = tk.Tk()
main.title("Iris Recognition using Machine Learning Technique")
main.geometry("1300x1200")

global filename
global model

def getIrisFeatures(image):
    img = cv2.imread(image, 0)
    if img is None:
        messagebox.showerror("Error", "Image not found or unable to read")
        return None
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
        cv2.imwrite("test.png", crop)
        return cv2.imread("test.png")
    else:
        messagebox.showwarning("Warning", "No eye iris is found")
        return None

def uploadDataset():
    global filename
    filename = filedialog.askdirectory(initialdir=".")
    if not filename:
        return
    text.delete('1.0', tk.END)
    text.insert(tk.END, f"{filename} loaded\n\n")

def loadModel():
    global model
    text.delete('1.0', tk.END)
    try:
        X_train = np.load('model/X.txt.npy')
        Y_train = np.load('model/Y.txt.npy')
        text.insert(tk.END, f'Dataset contains total {X_train.shape[0]} iris images from {Y_train.shape[1]} classes\n')
    except FileNotFoundError:
        messagebox.showerror("Error", "Training data not found")
        return

    if os.path.exists('model/model.json'):
        with open('model/model.json', "r") as json_file:
            model = model_from_json(json_file.read())
        model.load_weights("model/model_weights.h5")
        print(model.summary())
        try:
            with open('model/history.pckl', 'rb') as f:
                data = pickle.load(f)
            accuracy = data['accuracy'][-1] * 100
            text.insert(tk.END, f"CNN Model Prediction Accuracy = {accuracy:.2f}%\n\n")
            text.insert(tk.END, "See Black Console to view CNN layers\n")
        except FileNotFoundError:
            messagebox.showerror("Error", "History file not found")
    else:
        messagebox.showerror("Error", "Model not found. Please train the model using train.py")

def predictChange():
    filename = filedialog.askopenfilename(initialdir="testSamples")
    if not filename:
        return
    image = getIrisFeatures(filename)
    if image is None:
        return
    img = cv2.resize(image, (64, 64))
    img = np.expand_dims(img, axis=0).astype('float32') / 255.0
    preds = model.predict(img)
    predict = np.argmax(preds) + 1
    messagebox.showinfo("Prediction", f'Iris found! Person ID predicted: {predict}')
    img_display = cv2.imread(filename)
    img_display = cv2.resize(img_display, (600, 400))
    img1 = cv2.imread('test.png')
    img1 = cv2.resize(img1, (400, 200))
    cv2.putText(img_display, f'Person ID Predicted from Iris Recognition is: {predict}', (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    cv2.imshow(f'Person ID Predicted from Iris Recognition is: {predict}', img_display)
    cv2.imshow('Iris features extracted from image', img1)
    cv2.waitKey(0)

def exit():
    main.destroy()

font1 = ('times', 14, 'bold')
title = Label(main, text='Iris Recognition using Machine Learning Technique')
title.config(bg='darkviolet', fg='gold')
title.config(font=font1)
title.config(height=3, width=120)
title.place(x=5, y=5)

font2 = ('times', 12, 'bold')
uploadButton = Button(main, text="Upload CASIA Iris Image Dataset", command=uploadDataset)
uploadButton.place(x=50, y=100)
uploadButton.config(font=font2)

pathlabel = Label(main)
pathlabel.config(bg='darkviolet', fg='white')
pathlabel.config(font=font2)
pathlabel.place(x=50, y=150)

predictButton = Button(main, text="Upload Test Image & Predict Person", command=predictChange)
predictButton.place(x=50, y=200)
predictButton.config(font=font2)

exitButton = Button(main, text="Exit", command=exit)
exitButton.place(x=50, y=250)
exitButton.config(font=font2)

font3 = ('times', 12, 'bold')
text = Text(main, height=20, width=150)
scroll = Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=10, y=300)
text.config(font=font3)

main.config(bg='darkviolet')
main.mainloop()
