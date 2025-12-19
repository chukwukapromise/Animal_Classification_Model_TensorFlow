import numpy as np
import streamlit as st
import tensorflow as tf
import json
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_v2_preprocess_input

# Load model
interpreter = tf.lite.Interpreter(model_path="saved_model/animal_class.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# App title and description
st.title("Animal Classifier")
st.text(
    "This model has been trained on a diverse dataset covering 90 different animal species, "
    "enabling accurate and fast predictions. The application is designed with a simple, "
    "user-friendly interface and is optimized for real-time inference."
)
# File uploader 
uploaded_file = st.file_uploader(
    "Choose an image file to get started",
    type=["png", "jpg", "jpeg"]
)

# Label mapping
label_dict = {
    0: 'antelope', 1: 'badger', 2: 'bat', 3: 'bear',
    4: 'bee', 5: 'beetle', 6: 'bison', 7: 'boar',
    8: 'butterfly', 9: 'cat', 10: 'caterpillar',
    11: 'chimpanzee', 12: 'cockroach', 13: 'cow',
    14: 'coyote', 15: 'crab', 16: 'crow', 17: 'deer',
    18: 'dog', 19: 'dolphin', 20: 'donkey',
    21: 'dragon fly', 22: 'duck', 23: 'eagle',
    24: 'elephant', 25: 'flamingo', 26: 'fly',
    27: 'fox', 28: 'goat', 29: 'goldfish',
    30: 'goose', 31: 'gorilla', 32: 'grasshopper',
    33: 'hamster', 34: 'hare', 35: 'hedgehog',
    36: 'hippopotamus', 37: 'hornbill', 38: 'horse',
    39: 'hummingbird', 40: 'hyena', 41: 'jellyfish',
    42: 'kangaroo', 43: 'koala', 44: 'ladybugs',
    45: 'leopard', 46: 'lion', 47: 'lizard',
    48: 'lobster', 49: 'mosquito', 50: 'moth',
    51: 'mouse', 52: 'octopus', 53: 'okapi',
    54: 'orangutan', 55: 'otter', 56: 'owl',
    57: 'ox', 58: 'oyster', 59: 'panda',
    60: 'parrot', 61: 'pelecaniformes', 62: 'penguin',
    63: 'pig', 64: 'pigeon', 65: 'porcupine',
    66: 'possum', 67: 'raccoon', 68: 'rat',
    69: 'reindeer', 70: 'rhinoceros', 71: 'sandpiper',
    72: 'seahorse', 73: 'seal', 74: 'shark',
    75: 'sheep', 76: 'snake', 77: 'sparrow',
    78: 'squid', 79: 'squirrel', 80: 'star',
    81: 'swan', 82: 'tiger', 83: 'turkey',
    84: 'turtle', 85: 'whale', 86: 'wolf',
    87: 'wombat', 88: 'woodpecker', 89: 'zebra'
}

@st.cache_data
def load_animal_descriptions():
    with open("animal_descriptions.json", "r") as f:
        return json.load(f)

animal_info = load_animal_descriptions()

# Prediction
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, use_column_width=True)
    
    resized = image.resize((224, 224))
    
    img_array = np.array(resized).astype(np.float32)
    img_array = mobilenet_v2_preprocess_input(img_array)
    
    img_reshape = np.expand_dims(img_array, axis=0)

    if st.button("Generate Prediction"):
        
        # Inference
        interpreter.set_tensor(input_details[0]['index'], img_reshape)
        interpreter.invoke()
        prediction = interpreter.get_tensor(output_details[0]['index'])

        class_id = np.argmax(prediction)
        confidence = np.max(prediction) * 100

        predicted_label = label_dict[class_id]
        info = animal_info.get(predicted_label)

        # Result
        st.success(f"{predicted_label.title()} ({confidence:.2f}%)")

        # Description
        if info:
            with st.expander("Learn more about this animal"):
                st.write(info["summary"])
                st.write(f"Habitat: {info['habitat']}")
                st.write(f"Diet: {info['diet']}")
                st.write(f"Lifespan: {info['lifespan']}")
        else:
            st.warning("No description available for this animal.")
