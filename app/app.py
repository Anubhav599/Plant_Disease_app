# -*- coding: utf-8 -*-


import kagglehub

# Download latest version
path = kagglehub.dataset_download("abdallahalidev/plantvillage-dataset")

print("Path to dataset files:", path)

"""## Seeding for reproducibility"""

import random
random.seed(0)

import numpy as np
np.random.seed(0)

import tensorflow as tf
tf.random.set_seed(0)

"""##importing the dependencies"""

import os
import json
from zipfile import ZipFile
from PIL import Image

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers , models

"""##Data Curation"""

! pip install kaggle

kaggle_credentials = json.load(open("kaggle.json"))

os.environ['KAGGLE_USERNAME'] = kaggle_credentials['username']
os.environ['KAGGLE_KEY'] = kaggle_credentials['key']

!kaggle datasets download -d abdallahalidev/plantvillage-dataset

!ls

with ZipFile("plantvillage-dataset.zip","r") as zip_ref:
  zip_ref.extractall()

print(os.listdir("plantvillage dataset"))
print(len(os.listdir("plantvillage dataset/segmented")))
print(os.listdir("plantvillage dataset/segmented") [:5])

print(len(os.listdir("plantvillage dataset/color")))
print(os.listdir("plantvillage dataset/color") [:5])

print(len(os.listdir("plantvillage dataset/grayscale")))
print(os.listdir("plantvillage dataset/grayscale") [:5])

"""##Number of Classes = 38"""

print(os.listdir("plantvillage dataset/color"))
print(os.listdir("plantvillage dataset/color/Grape___healthy")[:5])

"""##Data Preprocessing"""

base_dir = "plantvillage dataset/color"

image_path = '/content/plantvillage dataset/color/Apple___Apple_scab/00075aa8-d81a-4184-8541-b692b78d398a___FREC_Scab 3335.JPG'

img = mpimg.imread(image_path)

print(img.shape)
plt.imshow(img)
plt.axis('off')
plt.show()

image_path = "/content/plantvillage dataset/color/Apple___Black_rot/00e909aa-e3ae-4558-9961-336bb0f35db3___JR_FrgE.S 8593.JPG"

img = mpimg.imread(image_path)

print(img)

img_size = 224
batch_size = 32

"""##Train Test Split"""

data_gen = ImageDataGenerator(
    rescale = 1./255,
    validation_split = 0.2
)

train_generator = data_gen.flow_from_directory(
    base_dir,
    target_size = (img_size,img_size),
    batch_size = batch_size,
    subset = 'training',
    class_mode='categorical'
)

validation_generator = data_gen.flow_from_directory(
    base_dir,
    target_size = (img_size,img_size),
    batch_size = batch_size,
    subset = 'validation',
    class_mode='categorical'
)

"""##Convolutional Neural Network (CNN)"""

model = models.Sequential()

model.add(layers.Conv2D(32,(3,3), activation='relu', input_shape=(img_size,img_size,3)))
model.add(layers.MaxPooling2D(2,2))

model.add(layers.Conv2D(64,(3,3), activation='relu'))
model.add(layers.MaxPooling2D(2,2))

model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(train_generator.num_classes, activation='softmax'))

model.summary()

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

"""##Model Training"""

history = model.fit(
    train_generator,
    steps_per_epoch = train_generator.samples // batch_size,
    epochs = 5,
    validation_data = validation_generator,
    validation_steps = validation_generator.samples // batch_size
)

"""#Model Evaluation"""

print("Evaluation model...")
val_loss , val_accuracy = model.evaluate(validation_generator, steps= validation_generator.samples // batch_size)
print(f"Validation Accuracy:  {val_accuracy *100:.2f}")

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Test'], loc = 'upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Test'], loc = 'upper left')
plt.show()

"""##Building a Predictive System"""

def load_and_preprocess_image(image_path, target_size=(224, 224)):
    img = Image.open(image_path)
    img = img.resize(target_size)
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array.astype('float32')/255
    return img_array

def predict_image_class(model, image_path,class_indices):
  preprocessed_img = load_and_preprocess_image(image_path)
  prediction = model.predict(preprocessed_img)
  predicted_class_index = np.argmax(prediction , axis=1)[0]
  predicted_class_name = class_indices[predicted_class_index]
  return predicted_class_name

class_indices = {v: k for k, v in train_generator.class_indices.items()}

class_indices

json.dump(class_indices, open("class_indices.json","w"))

from google.colab import drive
drive.mount('/content/drive')

image_path = '/content/plantvillage dataset/color/Apple___Black_rot/02859ed3-f56a-4315-82a5-c1cb72717225___JR_FrgE.S 8584.JPG'
predicted_class_name = predict_image_class(model, image_path,class_indices)
print("Predicted Class Name:", predicted_class_name)

model.save('plant_disease_model.h5')

from google.colab import files
files.download('plant_disease_model.h5')
