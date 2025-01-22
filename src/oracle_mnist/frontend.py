import bentoml
import cv2
import numpy as np
import streamlit as st

# from google.cloud import run_v2
from PIL import Image
from streamlit_drawable_canvas import st_canvas

# def get_backend_url():
#     """Get the URL of the backend service."""
#     parent = "projects/my-personal-mlops-project/locations/europe-west1"
#     client = run_v2.ServicesClient()
#     services = client.list_services(parent=parent)
#     for service in services:
#         if service.name.split("/")[-1] == "production-model":
#             return service.uri
#     return os.environ.get("BACKEND", None)


def classify_image(images: np.ndarray, backend):
    """Send the image to the backend for classification."""
    images = images.tolist()
    # predict_url = f"{backend}/predict"
    client = bentoml.SyncHTTPClient(backend)
    response = client.predict(im=images)
    print(response)
    if response.status_code == 200:
        return response.json()
    return None


def main() -> None:
    """Main function of the Streamlit frontend."""
    backend = "https://localhost:6060"  # get_backend_url()
    if backend is None:
        msg = "Backend service not found"
        raise ValueError(msg)

    st.markdown(
        "<style>body{color: Black; background-color: White}</style>",
        unsafe_allow_html=True,
    )

    st.title("Chinese Oracle Bone Character Recognizer")
    st.write("Predict on one of the following chinese bone characters")

    st.image("docs/images/chinese_characters.png", use_container_width=True)
    st.write(
        "[Mordern] Big: 大, Sun: 日,\
             Moon 月, Cattle 牛, Next 翌,\
            Field 田, Not 勿, Arrow 矢, Time 巳, Wood: 木"
    )

    tab1, tab2 = st.tabs(["Upload Image", "Draw Image"])

    with tab1:
        uploaded_file = st.file_uploader(
            "Please upload an image",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True,
        )
        if uploaded_file is not None:
            if st.button("Predict", key="predict"):
                imgs = [cv2.resize(np.array(Image.open(file)), (28, 28)) for file in uploaded_file]
                imgs = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in imgs]
                imgs = [np.repeat(img[None, ...], 3, axis=0) for img in imgs]
                imgs = np.array(imgs)

                results = classify_image(imgs, backend=backend)
                print("results: " + results)

            for file in uploaded_file:
                st.image(file, caption="Uploaded Image", use_container_width=True)

        else:
            st.write("Please upload an image")

    with tab2:
        col1, col2 = st.columns(2)

        with col1:
            mode = st.checkbox("Draw (or Delete)?", True)
            canvas_result = st_canvas(
                fill_color="#000000",
                stroke_width=20,
                stroke_color="#FFFFFF",
                background_color="#000000",
                width=192,
                height=192,
                drawing_mode="freedraw" if mode else "transform",
                key="canvas",
            )
            if st.button("Predict", key="predict2"):
                img = cv2.resize(canvas_result.image_data.astype("uint8"), (28, 28))
                imgs = img[None, None, ...]
                imgs = np.repeat(imgs, 3, axis=1)

                results = classify_image(imgs, backend=backend)
                print("results: " + results)

        with col2:
            if canvas_result.image_data is not None:
                img = cv2.resize(canvas_result.image_data.astype("uint8"), (28, 28))
                rescaled = cv2.resize(img, (192, 192), interpolation=cv2.INTER_NEAREST)
                st.write("Pre-processed")
                st.image(rescaled)


if __name__ == "__main__":
    main()
