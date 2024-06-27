import cv2 
from PIL import Image
import numpy as np
import streamlit as st 

MODEL = "model/MobileNetSSD_deploy.caffemodel"
PROTOTXT = "model/MobileNetSSD_deploy.prototxt.txt"

def process_image(image):
    blob = cv2.dnn.blobFromImage(
        cv2.resize(image, (300,300)), 0.007843, (300, 300), 127.5
    )
    net = cv2.dnn.readNetFromCaffe(PROTOTXT, MODEL)
    net.setInput(blob)
    detections = net.forward()
    return detections 

def annotate_image(
        image, detections, confidence_threhold=0.5
    ):
    # loop over the detections
    (h,w) = image.shape[:2]
    for i in np.arange(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > confidence_threhold:
            #Extract the index of the class label from the 'detections', then compute the 
            #(x-y)-coordinate of the bounding box for the object
            box = detections[0, 0, i, 3:7]*np.array([w, h, w, h])
            (start_x, start_y, end_x, end_y) = box.astype("int")
            cv2.rectangle(image, (start_x, start_y), (end_x, end_y), 70, 2)
    return image

def main():
    st.title("Object Detection for Images")
    file = st.file_uploader("Choose A File", type = ['jpg','png','jpeg'])
    if file is not None:
        st.image(file, caption="Uploaded Image")

        if st.button("Detect Object"):
            image = Image.open(file)
            image = np.array(image)
            detections = process_image(image)
            processed_image = annotate_image(image, detections)
            st.image(processed_image, caption="Processed Image")

if __name__ == "__main__":
    main()