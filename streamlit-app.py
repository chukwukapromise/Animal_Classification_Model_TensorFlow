import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_v2_preprocess_input

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path="model/animal_class.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Streamlit 
st.title("Animal Classifier")
st.text(
    "This model has been trained on a diverse dataset covering 90 different animal species, "
    "enabling accurate and fast predictions. The application is designed with a simple, "
    "user-friendly interface and is optimized for real-time inference."
)

uploaded_file = st.file_uploader(
    "Choose an image file to get started",
    type=["png", "jpg", "jpeg"]
)

# Label mapping
label_dict = {
    0: 'Antelope', 1: 'Badger', 2: 'Bat', 3: 'Bear',
    4: 'Bee', 5: 'Beetle', 6: 'Bison', 7: 'Boar',
    8: 'Butterfly', 9: 'Cat', 10: 'Caterpillar',
    11: 'Chimpanzee', 12: 'Cockroach', 13: 'Cow',
    14: 'Coyote', 15: 'Crab', 16: 'Crow', 17: 'Deer',
    18: 'Dog', 19: 'Dolphin', 20: 'Donkey',
    21: 'Dragon fly', 22: 'Duck', 23: 'Eagle',
    24: 'Elephant', 25: 'Flamingo', 26: 'Fly',
    27: 'Fox', 28: 'Goat', 29: 'Goldfish',
    30: 'Goose', 31: 'Gorilla', 32: 'Grasshopper',
    33: 'Hamster', 34: 'Hare', 35: 'Hedgehog',
    36: 'Hippopotamus', 37: 'Hornbill', 38: 'Horse',
    39: 'Hummingbird', 40: 'Hyena', 41: 'Jellyfish',
    42: 'Kangaroo', 43: 'Koala', 44: 'Ladybugs',
    45: 'Leopard', 46: 'Lion', 47: 'Lizard',
    48: 'Lobster', 49: 'Mosquito', 50: 'Moth',
    51: 'Mouse', 52: 'Octopus', 53: 'Okapi',
    54: 'Orangutan', 55: 'Otter', 56: 'Owl',
    57: 'Ox', 58: 'Oyster', 59: 'Panda',
    60: 'Parrot', 61: 'Pelecaniformes', 62: 'Penguin',
    63: 'Pig', 64: 'Pigeon', 65: 'Porcupine',
    66: 'Possum', 67: 'Raccoon', 68: 'Rat',
    69: 'Reindeer', 70: 'Rhinoceros', 71: 'Sandpiper',
    72: 'Seahorse', 73: 'Seal', 74: 'Shark',
    75: 'Sheep', 76: 'Snake', 77: 'Sparrow',
    78: 'Squid', 79: 'Squirrel', 80: 'Star',
    81: 'Swan', 82: 'Tiger', 83: 'Turkey',
    84: 'Turtle', 85: 'Whale', 86: 'Wolf',
    87: 'Wombat', 88: 'Woodpecker', 89: 'Zebra'
}

# Prediction
if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    opencv_image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)

    st.image(opencv_image, channels="RGB")

    resized = cv2.resize(opencv_image, (224, 224))
    resized = mobilenet_v2_preprocess_input(resized.astype(np.float32))
    img_reshape = np.expand_dims(resized, axis=0)

    if st.button("Generate Prediction"):
        interpreter.set_tensor(input_details[0]['index'], img_reshape)
        interpreter.invoke()
        prediction = interpreter.get_tensor(output_details[0]['index'])

        class_id = np.argmax(prediction)
        confidence = np.max(prediction) * 100

        st.success(f"{label_dict[class_id]} ({confidence:.2f}%)")
