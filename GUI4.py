# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'GUI.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from fileinput import filename
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QDialog, QApplication, QFileDialog


import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
import random
from tensorflow import keras
import datetime


from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
# more info on callbakcs: https://keras.io/callbacks/ model saver is cool too.

from keras.callbacks import TensorBoard

import openpyxl
from openpyxl import Workbook

import pickle
import time
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.metrics import plot_confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
import seaborn as sns

global model

global CATEGORIES

global fname
fname = 'a'


class Ui_MainWindow(object):






    def button_clicked(self):
        print("you pressed the button")

        self.label_2.setText("Loading Model")

        global model


        model = tf.keras.models.load_model("64x3-CNN.model")
        #prediction = model.predict([prepare('C:/Users/Abbad/Desktop/Sommer22/Project/doggo.jpg')])


        
        self.label_2.setText("Model Loaded")




    def button_clicked2(self):

        def make_confusion_matrix(cf,
                          group_names=None,
                          categories='auto',
                          count=True,
                          percent=True,
                          cbar=True,
                          xyticks=True,
                          xyplotlabels=True,
                          sum_stats=True,
                          figsize=None,
                          cmap='Blues',
                          title=None):
   


            # CODE TO GENERATE TEXT INSIDE EACH SQUARE
            blanks = ['' for i in range(cf.size)]

            if group_names and len(group_names)==cf.size:
                group_labels = ["{}\n".format(value) for value in group_names]
            else:
                group_labels = blanks

            if count:
                group_counts = ["{0:0.0f}\n".format(value) for value in cf.flatten()]
            else:
                group_counts = blanks

            if percent:
                group_percentages = ["{0:.2%}".format(value) for value in cf.flatten()/np.sum(cf)]
            else:
                group_percentages = blanks

            box_labels = [f"{v1}{v2}{v3}".strip() for v1, v2, v3 in zip(group_labels,group_counts,group_percentages)]
            box_labels = np.asarray(box_labels).reshape(cf.shape[0],cf.shape[1])


            # CODE TO GENERATE SUMMARY STATISTICS & TEXT FOR SUMMARY STATS
            if sum_stats:
                #Accuracy is sum of diagonal divided by total observations
                accuracy  = np.trace(cf) / float(np.sum(cf))

                #if it is a binary confusion matrix, show some more stats
                if len(cf)==2:
                    #Metrics for Binary Confusion Matrices
                    precision = cf[1,1] / sum(cf[:,1])
                    recall    = cf[1,1] / sum(cf[1,:])

                    global true_postive, false_negative, true_negative, false_positive , true_postive_rate, false_negative_rate, true_negative_rate, false_positive_rate, f1_score

                    true_postive_rate = cf[1,1] /sum(cf[1,:])
                    true_postive = cf[1,1]
                    print("true_postive_rate",true_postive_rate)

                    false_negative_rate = cf[1,0] /sum(cf[1,:])
                    false_negative = cf[1,0]
                    print("false_negative_rate",false_negative_rate)

                    true_negative_rate = cf[0,0] /sum(cf[0,:])
                    true_negative = cf[0,0]
                    print("true_negative_rate",true_negative_rate)

                    false_positive_rate = cf[0,1] /sum(cf[0,:])
                    false_positive = cf[0,1]
                    print("false_positive_rate",false_positive_rate)


                    f1_score  = 2*precision*recall / (precision + recall)

                    stats_text = "\nTrue Postive Rate={:0.3f}  False Negative Rate={:0.3f}\nTrue Negative Rate={:0.3f}  False Positive Rate={:0.3f}\nAccuracy={:0.3f}\nPrecision={:0.3f}\nRecall={:0.3f}\nF1 Score={:0.3f}".format(
                        true_postive_rate,false_negative_rate,true_negative_rate,false_positive_rate,accuracy,precision,recall,f1_score)
                else:
                    stats_text = "\n\nAccuracy={:0.3f}".format(accuracy)
            else:
                stats_text = ""


            # SET FIGURE PARAMETERS ACCORDING TO OTHER ARGUMENTS
            if figsize==None:
                #Get default figure size if not set
                figsize = plt.rcParams.get('figure.figsize')

            if xyticks==False:
                #Do not show categories if xyticks is False
                categories=False




            # MAKE THE HEATMAP VISUALIZATION
            plt.figure(figsize=figsize)
            sns.heatmap(cf,annot=box_labels,fmt="",cmap=cmap,cbar=cbar,xticklabels=categories,yticklabels=categories)

            if xyplotlabels:
                plt.ylabel('Actual Values')
                plt.xlabel('Predicted Values' + stats_text)
            else:
                plt.xlabel(stats_text)
            
            if title:
                plt.title(title)
        
        print("you pressed the button2")

        wb = Workbook()

        # grab the active worksheet
        ws = wb.active

        # Data can be assigned directly to cells
        #ws['A1'] = 42

        # Rows can also be appended
        ws.append(["File Name", "Category 1 Name", "Category 2 Name", "Predicted Value", "Predicted Int", "Actual Int" , "Total Files",  "TP", "TN", "FP", "FN", "TPR",
                 "TNR","FPR","FNR", "F1-Score", "ROC"])

        def prepare(filepath):
            IMG_SIZE = 100  # 50 in txt-based
            img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
            new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
            return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)


      
        directory_in_str = fname

        directory = os.fsencode(directory_in_str)
            
        for file in os.listdir(directory):
            filename = os.fsdecode(file)
            print("filename=", filename)
            path = fname + "/" + filename
            print("path=", path)

            if (filename.startswith('c')) or (filename.startswith('C')):
                actual_value = 1
            else:
                actual_value = 0

            prediction = model.predict([prepare(path)])

            Predicted_Value = CATEGORIES[int(prediction[0][0])]

            Predicted_Int = int(prediction[0][0])

            print("Predicted_Int ",Predicted_Int)


            ws.append([filename, CATEGORIES[0], CATEGORIES[1], Predicted_Value, Predicted_Int, actual_value])

        # Save the file

        wb.save("sample.xlsx")



        df = pd.read_excel(r'sample.xlsx', header=None, converters={'2':int,'2':int}) # can also index sheet by name or fetch all sheets

        y = df[4].tolist()
        X = df[5].tolist()

        X = X[1:]

        y = y[1:]

        print( 'X',X)
        print('Y',y)

        cf_matrix = confusion_matrix(X, y)
        print(cf_matrix)

        

        labels = ['True Negative','False Positive','False Negative','True Positive']
       
        make_confusion_matrix(cf_matrix, 
                            group_names=labels,
                            categories=CATEGORIES,
                            title= "Confusion Matrix",
                            sum_stats = True)

        print("prediction", prediction)


        print([int(prediction[0][0])])  # will be a list in a list.

        print(CATEGORIES[int(prediction[0][0])])

        self.label_2.setText(CATEGORIES[int(prediction[0][0])])

        ws.append(["All Files", "NULL", "NULL", "NULL", "NULL", "NULL", len(X), true_postive, true_negative, false_positive , false_negative,  true_postive_rate,  true_negative_rate, false_positive_rate, false_negative_rate, f1_score ])

        # Save the file

        wb.save("sample.xlsx")


        self.label_2.setText("Report Generated")

        plt.show()




    def button_clicked4(self):
        
        print("you pressed the button4")
        global fname

        fname = QFileDialog.getExistingDirectory(None, 'Select file')
        print(fname)
        self.label_2.setText("Folder Selected")

    def button_clicked5(self):
            
        print("you pressed the button5")

        text1 = self.lineEdit.text()
        text2 = self.lineEdit_2.text()

        print(text1)
        print(text2)

        global CATEGORIES

        try:

            CATEGORIES = [text1, text2]
            self.label_2.setText("Categories Saved")


        except:
            print("No Categories")











    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(800, 600)
        MainWindow.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(80, 130, 111, 31))
        self.pushButton.setObjectName("pushButton")
        self.pushButton.clicked.connect(self.button_clicked)
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(90, 20, 631, 91))
        font = QtGui.QFont()
        font.setFamily("Arial Rounded MT Bold")
        font.setPointSize(16)
        font.setBold(False)
        font.setWeight(50)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.pushButton_2 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_2.setGeometry(QtCore.QRect(80, 190, 151, 28))
        self.pushButton_2.setObjectName("pushButton_2")
        self.pushButton_2.clicked.connect(self.button_clicked2)
        self.pushButton_3 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_3.setGeometry(QtCore.QRect(80, 300, 93, 28))
        self.pushButton_3.setObjectName("pushButton_3")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(410, 210, 271, 31))
        self.label_2.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.label_2.setText("")
        self.label_2.setObjectName("label_2")
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(580, 430, 211, 141))
        self.label_3.setText("")
        self.label_3.setPixmap(QtGui.QPixmap("FRA-UAS_Logo_rgb.jpg"))
        self.label_3.setScaledContents(True)
        self.label_3.setWordWrap(True)
        self.label_3.setObjectName("label_3")
        self.pushButton_4 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_4.setGeometry(QtCore.QRect(80, 240, 151, 31))
        self.pushButton_4.setObjectName("pushButton_4")
        self.pushButton_4.clicked.connect(self.button_clicked4)

        self.label_4 = QtWidgets.QLabel(self.centralwidget)
        self.label_4.setGeometry(QtCore.QRect(90, 400, 161, 21))
        font = QtGui.QFont()
        font.setPointSize(9)
        self.label_4.setFont(font)
        self.label_4.setObjectName("label_4")
        self.label_5 = QtWidgets.QLabel(self.centralwidget)
        self.label_5.setGeometry(QtCore.QRect(90, 450, 161, 21))
        font = QtGui.QFont()
        font.setPointSize(9)
        self.label_5.setFont(font)
        self.label_5.setObjectName("label_5")
        self.lineEdit = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit.setGeometry(QtCore.QRect(280, 400, 113, 22))
        self.lineEdit.setObjectName("lineEdit")
        self.lineEdit_2 = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit_2.setGeometry(QtCore.QRect(280, 450, 113, 22))
        self.lineEdit_2.setObjectName("lineEdit_2")
        self.pushButton_5 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_5.setGeometry(QtCore.QRect(460, 420, 111, 28))
        self.pushButton_5.setObjectName("pushButton_5")
        self.pushButton_5.clicked.connect(self.button_clicked5)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 808, 26))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.pushButton.setText(_translate("MainWindow", "Load Algorithm"))
        self.label.setText(_translate("MainWindow", "Test System For Pretrained Decision Algorithms"))
        self.pushButton_2.setText(_translate("MainWindow", "Read Data / Print Result"))
        self.pushButton_3.setText(_translate("MainWindow", "Print Report"))
        self.pushButton_4.setText(_translate("MainWindow", "Select Test Data Folder"))
        self.label_4.setText(_translate("MainWindow", "Classification Category 1"))
        self.label_5.setText(_translate("MainWindow", "Classification Category 2"))
        self.pushButton_5.setText(_translate("MainWindow", "Save Categories"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
