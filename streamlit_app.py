# import streamlit as st
# import cv2
# import torch

# from streamlit.logger import get_logger

# LOGGER = get_logger(__name__)

# # Load YOLOv8 model
# @st.cache(allow_output_mutation=True)
# def load_model():
#     model = torch.hub.load('yolo-model.pt', 'model', pretrained=True)
#     return model

    

# def run():
#     st.set_page_config(
#         page_title="Hello",
#         page_icon="👋",
#         )
#     # Streamlit app
#     st.title("YOLOv8 Object Detection")

#     uploaded_file = st.file_uploader("Choose an image...", type="jpg")

#     if uploaded_file is not None:
#         image = cv2.imread(uploaded_file.name)[:, :, ::-1]  # BGR to RGB
#         model = load_model()
#         results = model(image)
        
#         # Display results
#         st.image(image, caption="Uploaded Image", use_column_width=True)
#         st.write("Detected Objects:")
#         st.json(results.xyxy[0].numpy().tolist())

# if __name__ == "__main__":
#     run()


# Copyright (c) Streamlit Inc. (2018-2022) Snowflake Inc. (2022)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import streamlit as st
from streamlit.logger import get_logger
from PIL import Image
import numpy as np

LOGGER = get_logger(__name__)


def run():
    st.set_page_config(
        page_title="Hello",
        page_icon="👋",
    )

    st.title("YOLOv8 Object Detection")

    img_file_buffer = st.file_uploader("Upload an image")
    if img_file_buffer is not None:
        image = Image.open(img_file_buffer)
        img_array = np.array(image) # if you want to pass it to OpenCV
        st.image(image, caption="The caption", use_column_width=True)


if __name__ == "__main__":
    run()
