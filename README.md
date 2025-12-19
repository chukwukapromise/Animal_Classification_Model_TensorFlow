# Animal classifier

Animal Classifier is a deep learning–powered web application that identifies animals from images.
Users can upload a JPG, PNG, JPEG image, and the system automatically analyzes it using a trained MobileNetV2-based convolutional neural network, returning the most likely animal class from a predefined set of categories.

The model has been trained on a diverse dataset covering 90 different animal species, enabling accurate and fast predictions. The application is designed with a simple, user-friendly interface and is optimized for real-time inference.


# How the model works

1. Upload an image containing an animal

2. The image is preprocessed and resized to match the model’s input requirements

3. A deep learning model predicts the animal class

4. The predicted animal name is displayed instantly with confidence score.


# Key Features

1. Image-based animal recognition

2. Supports 90 animal categories

3. Fast inference using TensorFlow Lite & MobileNetV2

4. Clean and intuitive Streamlit web interface

5. Real-time prediction with visual feedback


# Tech Stack

Python, TensorFlow / Keras, MobileNetV2, OpenCV, NumPy, Streamlit.
