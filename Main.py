
from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
import matplotlib.pyplot as plt
import numpy as np
from tkinter import ttk
from tkinter import filedialog
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from numpy import dot
from numpy.linalg import norm
import math
import operator

# Create the main window
main = Tk()
main.title("Personalized Travel Planning System")
main.geometry("1300x1200")

# Global variables
global filename
global X, Y
global user_db, content_db
global vector

# Function to upload dataset


def uploadDataset():
    global filename
    text.delete('1.0', END)
    filename = filedialog.askdirectory(initialdir=".")
    text.insert(END, filename + " loaded\n")

# Function to process the dataset


def processDataset():
    text.delete('1.0', END)
    global filename
    global user_db, content_db
    user_db = pd.read_csv("Dataset/User.csv",
                          usecols=['Age', 'Sex', 'Category', 'Places'])
    content_db = pd.read_csv("Dataset/data_content.csv")
    user_db.fillna(0, inplace=True)
    content_db.fillna(0, inplace=True)
    text.insert(END, str(user_db.head()) + "\n\n")
    text.insert(END, str(content_db.head()) + "\n\n")
    content_db = content_db.values
    user_db = user_db.values

# Function to build the collaborative model


def collaborativeModel():
    global X, Y, user_db, content_db
    text.delete('1.0', END)
    X = []
    Y = []
    for i in range(len(user_db)):
        age = str(user_db[i, 0]).strip()
        sex = user_db[i, 1].strip().lower()
        category = user_db[i, 2].strip().lower()
        places = user_db[i, 3].strip().lower()
        content = age + " " + sex + " " + category + " " + places
        X.append(content)
        Y.append(category + "," + places)
    text.insert(END, "Model generated")

# Function to train the KNN algorithm


def trainKNN():
    global X, Y, vector
    vector = TfidfVectorizer()
    X = vector.fit_transform(X).toarray()
    text.insert(END, "KNN trained on below dataset vector\n\n")
    text.insert(END, str(X))

# Function to predict recommendations


def predict():
    text.delete('1.0', END)
    user_recommend = []
    global X, Y, vector, content_db

    query = tf1.get().lower()
    testArray = vector.transform([query]).toarray()
    testArray = testArray[0]
    for i in range(len(X)):
        recommend = dot(X[i], testArray) / (norm(X[i]) * norm(testArray))
        if recommend > 0:
            user_recommend.append([Y[i], recommend])
    user_recommend.sort(key=operator.itemgetter(1), reverse=True)
    top_recommend = []
    for index in range(0, 5):
        top_recommend.append(user_recommend[index][0])
    top = max(top_recommend, key=top_recommend.count)
    arr = top.split(",")
    text.insert(END, "Recommended Tourist Destination: " +
                str(arr[1]) + "\n\n")
    text.insert(
        END, "Below are the nearby places of the recommended destination\n\n")
    for i in range(len(content_db)):
        if arr[0] == str(content_db[i, 0]).strip().lower():
            distance = str(content_db[i, 1]).strip()
            duration = str(content_db[i, 2]).strip()
            nearby = str(content_db[i, 4]).strip()
            rating = str(content_db[i, 6]).strip()
            text.insert(END, "Distance = " + distance + "\n")
            text.insert(END, "Duration = " + duration + "\n")
            text.insert(END, "Nearby Places = " + nearby + "\n")
            text.insert(END, "Rating = " + rating + "\n\n")


# GUI setup
font = ('times', 15, 'bold')
title = Label(main, text='Personalized Travel Planning System')
title.config(font=font)
title.config(height=3, width=120)
title.place(x=0, y=5)

font1 = ('times', 13, 'bold')
ff = ('times', 12, 'bold')

uploadButton = Button(main, text="Upload Travel Dataset",
                      command=uploadDataset)
uploadButton.place(x=20, y=100)
uploadButton.config(font=ff)

processButton = Button(main, text="Preprocess Dataset", command=processDataset)
processButton.place(x=20, y=150)
processButton.config(font=ff)

buildButton = Button(
    main, text="Build Collaborative & Clustering Model", command=collaborativeModel)
buildButton.place(x=20, y=200)
buildButton.config(font=ff)

knnButton = Button(main, text="Train KNN Algorithm", command=trainKNN)
knnButton.place(x=20, y=250)
knnButton.config(font=ff)

l1 = Label(main, text='Input Your Requirements')
l1.config(font=font1)
l1.place(x=20, y=300)

tf1 = Entry(main, width=50)
tf1.config(font=font1)
tf1.place(x=20, y=350)

predictButton = Button(main, text="Predict Recommendation", command=predict)
predictButton.place(x=20, y=400)
predictButton.config(font=ff)

font1 = ('times', 12, 'bold')
text = Text(main, height=30, width=85)
scroll = Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=550, y=100)
text.config(font=font1)

# Start the GUI event loop
main.config()
main.mainloop()
