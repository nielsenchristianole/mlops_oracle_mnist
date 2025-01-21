import os

import pandas as pd
import requests
import streamlit as st
from streamlit_drawable_canvas import st_canvas
from google.cloud import run_v2
import cv2


def get_backend_url():
    """Get the URL of the backend service."""
    parent = "projects/my-personal-mlops-project/locations/europe-west1"
    client = run_v2.ServicesClient()
    services = client.list_services(parent=parent)
    for service in services:
        if service.name.split("/")[-1] == "production-model":
            return service.uri
    return os.environ.get("BACKEND", None)


def classify_image(image, backend):
    """Send the image to the backend for classification."""
    predict_url = f"{backend}/predict"
    response = requests.post(predict_url, files={"image": image}, timeout=10)
    if response.status_code == 200:
        return response.json()
    return None


def main() -> None:
    """Main function of the Streamlit frontend."""
    backend = get_backend_url()
    if backend is None:
        msg = "Backend service not found"
        raise ValueError(msg)

    SIZE = 192

    st.markdown('<style>body{color: Black; background-color: White}</style>', unsafe_allow_html=True)

    st.title('Chinese Oracle Bone Character Recognizer')
    st.write("Predict on one of the following chinese bone characters")

    st.image("docs/images/chinese_characters.png", use_container_width =True)
    st.write("[Mordern] Big: 大, Sun: 日, Moon 月, Cattle 牛, Next 翌, Field 田, Not 勿, Arrow 矢, Time 巳, Wood: 木")

    tab1, tab2 = st.tabs(["Upload Image", "Draw Image"])

    with tab1:
        uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
        
    with tab2:  
        col1, col2, col3 = st.columns(3)
        with col1:
            mode = st.checkbox("Draw (or Delete)?", True)
            canvas_result = st_canvas(
                                fill_color='#000000',
                                stroke_width=20,
                                stroke_color='#FFFFFF',
                                background_color='#000000',
                                width=SIZE,
                                height=SIZE,
                                drawing_mode="freedraw" if mode else "transform",
                                key='canvas')

        with col2:
            if canvas_result.image_data is not None:
                img = cv2.resize(canvas_result.image_data.astype('uint8'), (28, 28))
                rescaled = cv2.resize(img, (SIZE, SIZE), interpolation=cv2.INTER_NEAREST)
                st.write('Pre-processed')
                st.image(rescaled)
                
        with col3:
            if st.button('Predict'):
                test_x = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                image = uploaded_file.read()
                result = classify_image(image, backend=backend)

                if result is not None:
                    prediction = result["prediction"]
                    probabilities = result["probabilities"]

                    # show the image and prediction
                    st.image(image, caption="Uploaded Image")
                    st.write("Prediction:", prediction)

                    # make a nice bar chart
                    data = {"Class": [f"Class {i}" for i in range(10)], "Probability": probabilities}
                    df = pd.DataFrame(data)
                    df.set_index("Class", inplace=True)
                    st.bar_chart(df, y="Probability")
                else:
                    st.write("Failed to get prediction")


if __name__ == "__main__":
    main()