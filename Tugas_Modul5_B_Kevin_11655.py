import streamlit as st
import pickle
import numpy as np
from PIL import Image
import os

##model_directory = r'C:\Users\kevst\OneDrive\Documents\Kuliah\Sem 5\Pembelajaran Mesin dan Pembelajaran Mendalam\Introduction to Deep Learning (Praktek)\Introduction to Deep Learning (Praktek)'
model_path =  'best_model.pkl'

if os.path.exists(model_path):
    try:
        with open(model_path, 'rb') as model_file:
            model = pickle.load(model_file)
            
        class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
        
        def preprocess_image(image):
            image = image.resize((28,28))
            image = image.convert('L')
            image_array = np.array(image)
            image_array = image_array.reshape(1, -1)
            return image_array
            
        st.title("Fashion MNIST Image Classifier")
        st.write("Unggah beberapa gambar item fashion (misalnya sepatu, tas, baju), dan model akan memprediksi kelas masing-masing.")
        
        uploaded_files = st.file_uploader("Pilih gambar...", type=["jpg", "jpeg", "png"], accept_multiple_files = True)
        
        with st.sidebar:
            st.write("## Navigator")
            predict_button = st.button("Predict")

            if uploaded_files and predict_button:
                st.write("### Hasil Prediksi")
                
                for uploaded_file in uploaded_files:
                    image = Image.open(uploaded_file)
                    processed_image = preprocess_image(image)
                    predictions = model.predict_proba(processed_image)
                    predicted_class = np.argmax(predictions)
                    confidence = np.max(predictions) * 100

                    st.write(f"**Nama File:** {uploaded_file.name}")
                    st.write(f"Kelas Prediksi:** {class_names[predicted_class]}**")
                    st.write(f"Confidence:** {confidence:.2f}%**")
                    st.write("---")
        if uploaded_files:
            for uploaded_file in uploaded_files:
                image = Image.open(uploaded_file)
                st.image(image, caption=f"Gambar: {uploaded_file.name}", use_column_width=True)
    except Exception as e:
        st.error(f"Error: {str(e)}")
else:
    st.error("File Model Tidak ditemukan!")
                    
