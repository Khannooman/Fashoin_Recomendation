import tensorflow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input 
import numpy as np
from numpy.linalg import norm
import os
from tqdm import tqdm
import pickle
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
import cv2

model = ResNet50(weights = 'imagenet', include_top = False, input_shape = (224, 224,3))
model.trainable = False

model = tensorflow.keras.Sequential([model,
                    GlobalMaxPooling2D()])

feature_list = pickle.load(open('embeddings.pkl', 'rb'))
feature_list = np.array(feature_list)
filenames = pickle.load(open('filenames.pkl', 'rb'))

 
img = image.load_img('Sample\Sample_images.webp', target_size = (224, 224))
img_array = image.img_to_array(img)
extended_img_array = np.expand_dims(img_array, axis = 0)
preprocess_image = preprocess_input(extended_img_array)
result = model.predict(preprocess_image).flatten()
normalized = result/norm(result)
 
neighbors = NearestNeighbors(n_neighbors=5, algorithm='brute', metric = "euclidean")
neighbors.fit(feature_list)

distances, indices = neighbors.kneighbors([normalized])
for i in indices[0][1:6]:
    temp_image = cv2.imread(filenames[i])
    cv2.imshow('output', temp_image)
    cv2.waitKey(0)