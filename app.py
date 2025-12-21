# Imports 
import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import requests
import re
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# Page config
st.set_page_config(
    page_title="Animal Classifier",
    page_icon="🐾",
    layout="wide"
)

# Session state
if "image" not in st.session_state:
    st.session_state.image = None

if "prediction" not in st.session_state:
    st.session_state.prediction = None

# Load model 
@st.cache_resource
def load_model():
    interpreter = tf.lite.Interpreter(
        model_path="model/animal_class.tflite"
    )
    interpreter.allocate_tensors()
    return interpreter

interpreter = load_model()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Wikipedia helpers
def get_clean_wikipedia_text(query):
    url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "prop": "extracts",
        "explaintext": True,
        "exsectionformat": "wiki",
        "titles": query,
        "format": "json"
    }

    headers = {"User-Agent": "AnimalClassifier/1.0"}
    response = requests.get(url, params=params, headers=headers, timeout=8)
    data = response.json()

    pages = data["query"]["pages"]
    page = next(iter(pages.values()))
    text = page.get("extract", "")

    if not text:
        return None

    stop_sections = [
        "\n== References ==",
        "\n== External links ==",
        "\n== See also ==",
        "\n== Further reading =="
    ]

    for section in stop_sections:
        if section in text:
            text = text.split(section)[0]

    return text.strip()

def render_wikipedia_text(text):
    lines = text.split("\n")

    for line in lines:
        line = line.strip()
        if not line:
            continue

        match = re.match(r"^(=+)\s*(.*?)\s*\1$", line)

        if match:
            level = len(match.group(1))
            heading = match.group(2)

            if level == 1:
                st.header(heading)
            elif level == 2:
                st.subheader(heading)
            elif level == 3:
                st.markdown(f"**{heading}**")
            elif level == 4:
                st.markdown(f"***{heading}***")
            elif level == 5:
                st.markdown(f"*{heading}*")
            else:
                st.markdown(
                    f"<small><i>{heading}</i></small>",
                    unsafe_allow_html=True
                )
        else:
            st.write(line)

# Labels
label_dict = {
    0: 'antelope', 1: 'badger', 2: 'bat', 3: 'bear',
    4: 'bee', 5: 'beetle', 6: 'bison', 7: 'boar',
    8: 'butterfly', 9: 'cat', 10: 'caterpillar',
    11: 'chimpanzee', 12: 'cockroach', 13: 'cow',
    14: 'coyote', 15: 'crab', 16: 'crow', 17: 'deer',
    18: 'dog', 19: 'dolphin', 20: 'donkey',
    21: 'dragonfly', 22: 'duck', 23: 'eagle',
    24: 'elephant', 25: 'flamingo', 26: 'fly',
    27: 'fox', 28: 'goat', 29: 'goldfish',
    30: 'goose', 31: 'gorilla', 32: 'grasshopper',
    33: 'hamster', 34: 'hare', 35: 'hedgehog',
    36: 'hippopotamus', 37: 'hornbill', 38: 'horse',
    39: 'hummingbird', 40: 'hyena', 41: 'jellyfish',
    42: 'kangaroo', 43: 'koala', 44: 'ladybug',
    45: 'leopard', 46: 'lion', 47: 'lizard',
    48: 'lobster', 49: 'mosquito', 50: 'moth',
    51: 'mouse', 52: 'octopus', 53: 'okapi',
    54: 'orangutan', 55: 'otter', 56: 'owl',
    57: 'ox', 58: 'oyster', 59: 'panda',
    60: 'parrot', 61: 'pelican', 62: 'penguin',
    63: 'pig', 64: 'pigeon', 65: 'porcupine',
    66: 'possum', 67: 'raccoon', 68: 'rat',
    69: 'reindeer', 70: 'rhinoceros', 71: 'sandpiper',
    72: 'seahorse', 73: 'seal', 74: 'shark',
    75: 'sheep', 76: 'snake', 77: 'sparrow',
    78: 'squid', 79: 'squirrel', 80: 'starfish',
    81: 'swan', 82: 'tiger', 83: 'turkey',
    84: 'turtle', 85: 'whale', 86: 'wolf',
    87: 'wombat', 88: 'woodpecker', 89: 'zebra'
}

# Layout
left, center, right = st.columns([1, 3, 1])

with center:
    st.title("Animal Classifier")
    st.write(
        "This model has been trained on a diverse dataset covering 90 different animal species, "
        "enabling accurate and fast predictions."
    )

    uploaded_file = st.file_uploader(
        "Upload an image file",
        type=["jpg", "jpeg", "png", "webp"]
    )

    # Store image
    if uploaded_file:
        st.session_state.image = Image.open(uploaded_file).convert("RGB")
        st.session_state.prediction = None

    # Display image persistently
    if st.session_state.image:
        st.image(st.session_state.image, use_container_width=True)

        if st.button("Generate Prediction"):
            img = st.session_state.image.resize((224, 224))
            img_array = np.array(img).astype(np.float32)
            img_array = preprocess_input(img_array)
            img_array = np.expand_dims(img_array, axis=0)

            interpreter.set_tensor(input_details[0]["index"], img_array)
            interpreter.invoke()
            output = interpreter.get_tensor(output_details[0]["index"])

            class_id = int(np.argmax(output))
            confidence = float(np.max(output) * 100)
            animal = label_dict[class_id]

            st.session_state.prediction = {
                "animal": animal,
                "confidence": confidence
            }

    # Display prediction + Wikipedia
    if st.session_state.prediction:
        animal = st.session_state.prediction["animal"]
        confidence = st.session_state.prediction["confidence"]

        st.success(f"{animal.title()} ({confidence:.2f}%)")

        article_text = get_clean_wikipedia_text(animal)
        if article_text:
            render_wikipedia_text(article_text)
        else:
            st.warning("Article could not be retrieved.")
