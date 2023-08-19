import streamlit as st
import os
st.title('Fashion Recomemender System')
from PIL import Image
import tensorflow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input 
import numpy as np
from numpy.linalg import norm
import os
from tqdm import tqdm
import pickle
from sklearn.neighbors import NearestNeighbors
 
model = ResNet50(weights = 'imagenet', include_top = False, input_shape = (224, 224,3))
model.trainable = False

model = tensorflow.keras.Sequential([model,
                    GlobalMaxPooling2D()])

feature_list = pickle.load(open('embeddings.pkl', 'rb'))
feature_list = np.array(feature_list)
filenames = pickle.load(open('filenames.pkl', 'rb'))


def extract_features(img_path, model):
    img = image.load_img(img_path, target_size = (224, 224))
    img_array = image.img_to_array(img)
    extended_img_array = np.expand_dims(img_array, axis = 0)
    preprocess_image = preprocess_input(extended_img_array)
    result = model.predict(preprocess_image).flatten()
    normalized = result/norm(result)
    return normalized


def recomend(features, feature_list):
    neighbors = NearestNeighbors(n_neighbors=5, algorithm='brute', metric = "euclidean")
    neighbors.fit(feature_list)
    distances, indices = neighbors.kneighbors([features])
    return indices

# file upload and save
def save_upload_file(uploaded_file):
    try:
        with open(os.path.join('uploads', uploaded_file.name), 'wb') as f:
            f.write(uploaded_file.getbuffer())
        return 1
    except:
        return 0

uploaded_file = st.file_uploader("Choose an image")
if uploaded_file is not None:
    if save_upload_file(uploaded_file):
        display_image = Image.open(uploaded_file)
        st.image(display_image)

        features = extract_features(os.path.join('uploads', uploaded_file.name),model)
        # st.text(features)
        indices = recomend(features,feature_list)
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.image(filenames[indices[0][0]])
        with col2:
            st.image(filenames[indices[0][1]])
        with col3:
            st.image(filenames[indices[0][2]])
        with col4:
            st.image(filenames[indices[0][3]])
        with col5:
            st.image(filenames[indices[0][4]])
    else:
        st.header("Some error occured in file upload")