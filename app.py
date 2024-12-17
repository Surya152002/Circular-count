import streamlit as st
import cv2
import numpy as np
from PIL import Image

# Function to count circles in an image frame
def count_circles(frame):
    # Convert to grayscale and blur to remove noise
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    
    # Hough Circle Transform to detect circles
    circles = cv2.HoughCircles(
        gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=50,
        param1=50, param2=30, minRadius=10, maxRadius=100
    )
    
    circle_count = 0
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            # Draw the circle
            cv2.circle(frame, (i[0], i[1]), i[2], (0, 255, 0), 3)
            circle_count += 1
            
    return frame, circle_count

# Streamlit UI
def main():
    st.title("Circle Counter with Camera Input")
    st.write("Use your device camera to count and highlight circles in real-time or upload an image.")

    # Option to choose between Camera and Image upload
    option = st.radio("Choose Input Method:", ["Camera", "Upload Image"])

    # If user selects Camera
    if option == "Camera":
        st.write("Starting Camera Stream...")
        camera_stream = st.empty()  # Stream placeholder
        stop_button = st.button("Stop Camera")

        # Access camera using OpenCV
        cap = cv2.VideoCapture(0)  # 0 is the default camera
        
        while cap.isOpened() and not stop_button:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to capture video!")
                break

            # Count circles in the frame
            processed_frame, circle_count = count_circles(frame)
            
            # Convert the frame for display in Streamlit
            frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            camera_stream.image(frame_rgb, caption=f"Circles Detected: {circle_count}", use_column_width=True)

        cap.release()
        st.write("Camera Stopped.")

    # If user selects Upload Image
    elif option == "Upload Image":
        uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)

            # Convert to OpenCV format
            frame = np.array(image)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            # Count circles
            processed_image, circle_count = count_circles(frame)

            # Display processed image
            processed_image_rgb = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)
            st.image(processed_image_rgb, caption=f"Circles Detected: {circle_count}", use_column_width=True)
            st.success(f"Number of circles detected: {circle_count}")

if __name__ == "__main__":
    main()
