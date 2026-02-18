import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input

model = VGG16(
    weights="imagenet",
    include_top=False,
    pooling="max",
    input_shape=(224, 224, 3)
)

def extract_features(img_path):
    img = Image.open(img_path).resize((224, 224)).convert("RGB")
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    features = model.predict(img_array, verbose=0)
    return features.flatten()
