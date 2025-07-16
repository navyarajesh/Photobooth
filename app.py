
import streamlit as st
from streamlit_camera_input_live import camera_input
from datetime import datetime
import cv2
import numpy as np

def apply_grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def apply_sepia(img):
    kernel = np.array([[0.272, 0.534, 0.131],
                       [0.349, 0.686, 0.168],
                       [0.393, 0.769, 0.189]])
    sepia = cv2.transform(img, kernel)
    sepia = np.clip(sepia, 0, 255)
    return sepia.astype(np.uint8)

def apply_cartoon(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.medianBlur(gray, 5)
    edges = cv2.adaptiveThreshold(blur, 255,
                                  cv2.ADAPTIVE_THRESH_MEAN_C,
                                  cv2.THRESH_BINARY, 9, 9)
    color = cv2.bilateralFilter(img, 9, 300, 300)
    cartoon = cv2.bitwise_and(color, color, mask=edges)
    return cartoon

def overlay_image(bg, overlay, x, y, scale=1.0):
    overlay = cv2.resize(overlay, (0, 0), fx=scale, fy=scale)
    h, w, _ = overlay.shape
    if y + h > bg.shape[0] or x + w > bg.shape[1]:
        return bg

    for i in range(h):
        for j in range(w):
            if overlay[i, j][3] != 0:
                bg[y + i, x + j] = overlay[i, j][:3]
    return bg

st.set_page_config(page_title="Photobooth", layout="centered")
st.title("📸 iPad-Friendly Digital Photobooth")

img = camera_input("Take a photo with your iPad!")
filter_type = st.selectbox("Choose a filter", ["None", "Grayscale", "Sepia", "Cartoon"])
add_overlay = st.checkbox("Add Sunglasses Overlay")

if img:
    file_bytes = np.asarray(bytearray(img.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)

    if filter_type == "Grayscale":
        image = apply_grayscale(image)
    elif filter_type == "Sepia":
        image = apply_sepia(image)
    elif filter_type == "Cartoon":
        image = apply_cartoon(image)

    if add_overlay:
        overlay_img = cv2.imread("overlays/sunglasses.png", cv2.IMREAD_UNCHANGED)
        if overlay_img is not None:
            image = overlay_image(image, overlay_img, x=100, y=80, scale=0.5)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"photo_{timestamp}.jpg"
    cv2.imwrite(filename, image)

    st.success(f"Photo saved as {filename}")
    st.image(image, channels="BGR", caption="Filtered Photo")
    st.download_button("Download Photo", data=cv2.imencode(".jpg", image)[1].tobytes(), file_name=filename)
