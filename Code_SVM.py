from calendar import c
from pyexpat import model
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
import random
from tensorflow import keras

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
# more info on callbakcs: https://keras.io/callbacks/ model saver is cool too.

from keras.callbacks import TensorBoard

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC


import pickle
import time



## SVM


# dir = "C:/Users/Abbad/Desktop/Sommer22/Project/kagglecatsanddogs_5340/PetImages"
# Categories =  ['Cat', 'Dog']

# data = []

# for category in Categories :
#     path = os.path.join (dir, category)
#     label = Categories.index(category)


#     for img in os.listdir (path) :
#         imgpath = os.path.join(path, img)
#         pet_img=cv2.imread(imgpath, 0)
#         try:
#             pet_img=cv2.resize (pet_img, (50, 50))
#             image = np.array (pet_img).flatten()
#             data.append ([image, label])


#         except Exception as e:
#             pass

# print(len(data))

# pick_in = open("data1.pickle","wb")
# pickle.dump(data,pick_in)
# pick_in.close()


pick_in = open("data1.pickle","rb")
data = pickle.load(pick_in)
pick_in.close()

random.shuffle(data)
features = []
labels = []

for feature, label in data:
    features.append(feature)
    labels.append(label)

xtrain, xtest, ytrain, ytest = train_test_split(features, labels, test_size=0.98)  # Training 75 % of data #test_sixe = 0.25

# model = SVC( C=1, kernel='poly',  gamma='auto')

# model.fit(xtrain, ytrain)

# pick = open('svm_model', 'wb')

# pickle.dump(model,pick)

# pick.close()

pick = open('svm_model', 'rb')

model = pickle.load(pick)

pick.close()

prediction = model.predict(xtest)

accuracy = model.score(xtest, ytest)

Categories =  ['Cat', 'Dog']

print("accuracy = ", accuracy)
print("prediction =", Categories[prediction[0]])

mypet = xtest[0].reshape(50,50)

plt.imshow(mypet, cmap='gray')
plt.show()




## Part 6

# pickle_in = open("X.pickle","rb")
# X = pickle.load(pickle_in)

# pickle_in = open("y.pickle","rb")
# y = pickle.load(pickle_in)
# y=np.array(y)

# X = X/255.0

# dense_layers = [0]
# layer_sizes = [64]
# conv_layers = [3]

# for dense_layer in dense_layers:
#     for layer_size in layer_sizes:
#         for conv_layer in conv_layers:
#             NAME = "{}-conv-{}-nodes-{}-dense-{}".format(conv_layer, layer_size, dense_layer, int(time.time()))
#             print(NAME)

#             model = Sequential()

#             model.add(Conv2D(layer_size, (3, 3), input_shape=X.shape[1:]))
#             model.add(Activation('relu'))
#             model.add(MaxPooling2D(pool_size=(2, 2)))

#             for l in range(conv_layer-1):
#                 model.add(Conv2D(layer_size, (3, 3)))
#                 model.add(Activation('relu'))
#                 model.add(MaxPooling2D(pool_size=(2, 2)))

#             model.add(Flatten())

#             for _ in range(dense_layer):
#                 model.add(Dense(layer_size))
#                 model.add(Activation('relu'))

#             model.add(Dense(1))
#             model.add(Activation('sigmoid'))

#             tensorboard = TensorBoard(log_dir="logs/{}".format(NAME))

#             model.compile(loss='binary_crossentropy',
#                           optimizer='adam',
#                           metrics=['accuracy'],
#                           )

#             model.fit(X, y,
#                       batch_size=24,
#                       epochs=10,
#                       validation_split=0.3,
#                       callbacks=[tensorboard])

# model.save('64x3-CNN.model')




# CATEGORIES = ["Dog", "Cat"]


# def prepare(filepath):
#     IMG_SIZE = 100  # 50 in txt-based
#     img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
#     new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
#     return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)


# model = tf.keras.models.load_model("64x3-CNN.model")

# prediction = model.predict([prepare('C:/Users/Abbad/Desktop/Sommer22/Project/doggo.jpg')])
# print(type(prediction))
# print(prediction)  # will be a list in a list.
# print(CATEGORIES[int(prediction[0][0])])


## Part 3 

# pickle_in = open("X.pickle","rb")
# X = pickle.load(pickle_in)

# pickle_in = open("y.pickle","rb")
# y = pickle.load(pickle_in)
# y=np.array(y)

# X = X/255.0

# model = Sequential()

# model.add(Conv2D(256, (3, 3), input_shape=X.shape[1:]))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))

# model.add(Conv2D(256, (3, 3)))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))

# model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors

# model.add(Dense(64))

# model.add(Dense(1))
# model.add(Activation('sigmoid'))

# model.compile(loss='binary_crossentropy',
#               optimizer='adam',
#               metrics=['accuracy'])

# model.fit(X, y, batch_size=24, epochs=3, validation_split=.1)



## Part 2


# DATADIR = "C:/Users/Abbad/Desktop/Sommer22/Project/kagglecatsanddogs_5340/PetImages"

# CATEGORIES = ["Dog", "Cat"]

# for category in CATEGORIES:  # do dogs and cats
#     path = os.path.join(DATADIR,category)  # create path to dogs and cats
#     for img in os.listdir(path):  # iterate over each image per dogs and cats
#         img_array = cv2.imread(os.path.join(path,img) ,cv2.IMREAD_GRAYSCALE)  # convert to array
#         plt.imshow(img_array, cmap='gray')  # graph it
#         plt.show()  # display!

#         break  # we just want one for now so break
#     break  #...and one more!

# print(img_array)

# print(img_array.shape)

# IMG_SIZE = 100

# new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
# plt.imshow(new_array, cmap='gray')
# plt.show()

# training_data = []

# def create_training_data():
#     for category in CATEGORIES:  # do dogs and cats

#         path = os.path.join(DATADIR,category)  # create path to dogs and cats
#         class_num = CATEGORIES.index(category)  # get the classification  (0 or a 1). 0=dog 1=cat

#         for img in os.listdir(path):  # iterate over each image per dogs and cats
#             try:
#                 img_array = cv2.imread(os.path.join(path,img) ,cv2.IMREAD_GRAYSCALE)  # convert to array
#                 new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))  # resize to normalize data size
#                 training_data.append([new_array, class_num])  # add this to our training_data
#             except Exception as e:  # in the interest in keeping the output clean...
#                 pass
#             #except OSError as e:
#             #    print("OSErrroBad img most likely", e, os.path.join(path,img))
#             #except Exception as e:
#             #    print("general exception", e, os.path.join(path,img))

# create_training_data()

# print(len(training_data))

# random.shuffle(training_data)

# for sample in training_data[:10]:
#     print(sample[1])


# X = []
# y = []

# for features,label in training_data:
#     X.append(features)
#     y.append(label)

# print(X[0].reshape(-1, IMG_SIZE, IMG_SIZE, 1))

# X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)


# import pickle

# pickle_out = open("X.pickle","wb")
# pickle.dump(X, pickle_out)
# pickle_out.close()

# pickle_out = open("y.pickle","wb")
# pickle.dump(y, pickle_out)
# pickle_out.close()

# pickle_in = open("X.pickle","rb")
# X = pickle.load(pickle_in)

# pickle_in = open("y.pickle","rb")
# y = pickle.load(pickle_in)

# print(X[1])

## Part 1

# mnist = tf.keras.datasets.mnist

# (x_train, y_train), (x_test, y_test) = mnist.load_data() # 0 through 9. It's 28x28 images of these hand-written digits

# x_train = tf.keras.utils.normalize(x_train, axis=1)
# x_test = tf.keras.utils.normalize(x_test, axis=1)

# model = tf.keras.models.Sequential()
# model.add(tf.keras.layers.Flatten())
# model.add(tf.keras.layers.Dense(128, activation = tf.nn.relu))
# model.add(tf.keras.layers.Dense(128, activation = tf.nn.relu))
# model.add(tf.keras.layers.Dense(10, activation = tf.nn.softmax)) #10 because dataset is numbers from 0 - 9

# model.compile(optimizer='adam',  # Good default optimizer to start with
#               loss='sparse_categorical_crossentropy',  # how will we calculate our "error." Neural network aims to minimize loss.
#               metrics=['accuracy'])  # what to track


# model.fit(x_train, y_train, epochs=3)  # train the model

# val_loss, val_acc = model.evaluate(x_test, y_test)  # evaluate the out of sample data with model

# print(val_loss)  # model's loss (error)
# print(val_acc)  # model's accuracy

# model.save('epic_num_reader.model')

# new_model = tf.keras.models.load_model('epic_num_reader.model')

# predictions = new_model.predict(x_test)

# print(predictions)


# print(np.argmax(predictions[1]))

# plt.imshow(x_test[1],cmap=plt.cm.binary)

# plt.show()

# # print(x_train[0])

# # plt.imshow(x_train[0])
# # plt.show()